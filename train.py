"""Train the CNN-LSTM action recogniser, then evaluate the best checkpoint on the test split.

    python train.py
    python train.py --set model.backbone=mobilenet_v3_large --set train.output_dir=runs/mnv3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from sportsar.config import add_config_args, load_config
from sportsar.data.dataset import build_datasets, class_weights
from sportsar.engine import (ModelEMA, build_optimizer, build_scheduler, evaluate, make_loader, make_scaler,
                             predict, train_one_epoch)
from sportsar.metrics import classification_metrics, format_report, plot_confusion_matrix, plot_history
from sportsar.models import build_model
from sportsar.utils import count_parameters, get_device, save_json, set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    t, d = cfg["train"], cfg["data"]
    set_seed(cfg["seed"])
    device = get_device(args.device)
    amp = bool(t["amp"]) and device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    out_dir = Path(t["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, out_dir / "config.json")

    datasets, classes = build_datasets(cfg)
    print(f"Classes ({d['label_mode']}): {classes}")
    print({k: len(v) for k, v in datasets.items()})
    if not len(datasets["train"]) or not len(datasets["val"]):
        raise SystemExit("Empty train or val split - check prepare_data.py output.")
    loaders = {
        "train": make_loader(datasets["train"], t["batch_size"], True, d["num_workers"], cfg["seed"]),
        "val": make_loader(datasets["val"], t["batch_size"], False, d["num_workers"]),
        "test": make_loader(datasets["test"], t["batch_size"], False, d["num_workers"]),
    }

    model = build_model(cfg, len(classes)).to(device)
    total, trainable = count_parameters(model)
    print(f"Model: {cfg['model']['backbone']} + LSTM | params {total / 1e6:.2f}M (trainable {trainable / 1e6:.2f}M) | device {device}")

    weights = class_weights(datasets["train"].labels, len(classes)).to(device) if t["class_weighted_loss"] else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(t["label_smoothing"]))
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, len(loaders["train"]), cfg)
    scaler = make_scaler(amp)
    ema = ModelEMA(model, float(t["ema_decay"])) if float(t.get("ema_decay", 0) or 0) > 0 else None

    history, best_acc, best_epoch, bad_epochs = [], -1.0, 0, 0
    for epoch in range(1, int(t["epochs"]) + 1):
        tr = train_one_epoch(model, loaders["train"], optimizer, scheduler, scaler, criterion, device, amp,
                             float(t["grad_clip"]) if t["grad_clip"] else None,
                             float(t.get("mixup_alpha", 0)), float(t.get("mixup_prob", 0)), ema)
        va = evaluate(model, loaders["val"], criterion, device, amp)
        cand, tag = model, "raw"
        if ema is not None:  # keep whichever weights (raw or EMA) validate better this epoch
            va_ema = evaluate(ema.module, loaders["val"], criterion, device, amp)
            if va_ema["acc"] >= va["acc"]:
                va, cand, tag = va_ema, ema.module, "ema"
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["acc"], "val_loss": va["loss"],
               "val_acc": va["acc"], "weights": tag, "lr": optimizer.param_groups[0]["lr"],
               "epoch_time_s": tr["time_s"]}
        history.append(row)
        improved = va["acc"] > best_acc
        print(f"epoch {epoch:3d} | train loss {tr['loss']:.4f} acc {tr['acc'] * 100:5.1f}% | "
              f"val loss {va['loss']:.4f} acc {va['acc'] * 100:5.1f}% ({tag}) | {tr['time_s']:.0f}s"
              + ("  *best*" if improved else ""))
        ckpt = {"model": cand.state_dict(), "config": cfg, "classes": classes, "epoch": epoch, "val_acc": va["acc"]}
        torch.save(ckpt, out_dir / "last.pt")
        if improved:
            best_acc, best_epoch, bad_epochs = va["acc"], epoch, 0
            torch.save(ckpt, out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= int(t["patience"]):
                print(f"Early stopping: no val improvement for {bad_epochs} epochs.")
                break
        save_json(history, out_dir / "history.json")

    save_json(history, out_dir / "history.json")
    plot_history(history, out_dir / "training_curves.png")

    # ---------------- final test evaluation with the best checkpoint
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=False)["model"])
    t0 = time.perf_counter()
    res = predict(model, loaders["test"], device, amp)
    wall = time.perf_counter() - t0
    metrics = classification_metrics(res["labels"], res["probs"], classes)
    metrics.update({
        "best_epoch": best_epoch, "best_val_acc": best_acc,
        "model_clips_per_s": res["clips_per_s"], "model_frames_per_s": res["frames_per_s"],
        "end_to_end_clips_per_s_incl_loading": len(res["labels"]) / wall if wall else 0.0,
    })
    save_json(metrics, out_dir / "test_metrics.json")
    plot_confusion_matrix(metrics["confusion_matrix"], classes, out_dir / "confusion_matrix.png",
                          title=f"Test confusion matrix ({d['label_mode']})")
    print("\n=== TEST (single view) ===")
    print(format_report(metrics, classes))
    print(f"Throughput: {res['clips_per_s']:.1f} clips/s = {res['frames_per_s']:.0f} frames/s (model only, "
          f"batch {t['batch_size']}, {d['num_frames']} frames/clip)")
    print(f"\nArtifacts in {out_dir}/ : best.pt, history.json, training_curves.png, test_metrics.json, confusion_matrix.png")


if __name__ == "__main__":
    main()
