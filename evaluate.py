"""Evaluate a checkpoint on the test split, optionally with test-time augmentation
(2 temporal offsets x horizontal flip = 4 views averaged).

    python evaluate.py --checkpoint runs/cnn_lstm/best.pt --tta
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sportsar.config import apply_overrides
from sportsar.data.dataset import build_datasets
from sportsar.engine import make_loader, predict
from sportsar.metrics import classification_metrics, format_report, plot_confusion_matrix
from sportsar.models import load_checkpoint
from sportsar.utils import get_device, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--tta", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out-dir", default=None, help="defaults to the checkpoint folder")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override config stored in the checkpoint (e.g. data.cache_dir=...)")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    model, cfg, classes = load_checkpoint(args.checkpoint, device)
    cfg = apply_overrides(cfg, args.overrides)
    amp = device.type == "cuda"
    out_dir = Path(args.out_dir or Path(args.checkpoint).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    views = [(0.25, False), (0.75, False), (0.25, True), (0.75, True)] if args.tta else [(0.5, False)]
    probs, labels, fps = None, None, []
    for pos, flip in views:
        datasets, _ = build_datasets(cfg, eval_position=pos, eval_flip=flip)
        ds = datasets[args.split]
        ds.eval_flip = flip
        loader = make_loader(ds, args.batch_size, False, cfg["data"]["num_workers"])
        res = predict(model, loader, device, amp)
        probs = res["probs"] if probs is None else probs + res["probs"]
        labels = res["labels"]
        fps.append(res["frames_per_s"])
    probs = probs / len(views)

    metrics = classification_metrics(labels, probs, classes)
    metrics.update({"split": args.split, "views": len(views),
                    "model_frames_per_s": float(np.mean(fps)),
                    "model_clips_per_s": float(np.mean(fps)) / cfg["data"]["num_frames"] / len(views)})
    tag = f"{args.split}{'_tta' if args.tta else ''}"
    save_json(metrics, out_dir / f"eval_{tag}.json")
    plot_confusion_matrix(metrics["confusion_matrix"], classes, out_dir / f"confusion_matrix_{tag}.png",
                          title=f"{args.split} confusion matrix ({len(views)} view{'s' if len(views) > 1 else ''})")
    print(f"=== {args.split.upper()} | {len(views)} view(s) ===")
    print(format_report(metrics, classes))
    print(f"Model throughput: {metrics['model_frames_per_s']:.0f} frames/s per view; "
          f"{metrics['model_clips_per_s']:.1f} clips/s with {len(views)}-view TTA")
    print(f"Saved {out_dir / f'eval_{tag}.json'}")


if __name__ == "__main__":
    main()
