"""Training / evaluation loops (AMP, warmup + cosine schedule, gradient clipping)."""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import sync


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle,
                      worker_init_fn=seed_worker, generator=g,
                      persistent_workers=num_workers > 0)


def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    t = cfg["train"]
    groups = [{"params": model.head_parameters(), "lr": float(t["lr"])}]
    backbone = model.backbone_parameters()
    if backbone:
        groups.append({"params": backbone, "lr": float(t["backbone_lr"])})
    return torch.optim.AdamW(groups, weight_decay=float(t["weight_decay"]))


def build_scheduler(optimizer, steps_per_epoch: int, cfg: dict):
    t = cfg["train"]
    total = max(1, steps_per_epoch * int(t["epochs"]))
    warmup = steps_per_epoch * int(t["warmup_epochs"])

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=enabled and device.type == "cuda")


class ModelEMA:
    """Exponential moving average of the weights; the averaged copy usually generalises better."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))  # short warm-up
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device, amp: bool,
                    grad_clip: float | None, mixup_alpha: float = 0.0, mixup_prob: float = 0.0,
                    ema: ModelEMA | None = None) -> dict:
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    start = time.perf_counter()
    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lam, perm = 1.0, None
        if mixup_alpha > 0 and np.random.rand() < mixup_prob:
            # mixup: blend two clips (on uint8 -> float) and their labels
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(clips.size(0), device=device)
            clips = (clips.float() * lam + clips[perm].float() * (1 - lam)).round().to(torch.uint8)
        with autocast(device, amp):
            logits = model(clips)
            loss = criterion(logits, labels)
            if perm is not None:
                loss = lam * loss + (1 - lam) * criterion(logits, labels[perm])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        n = labels.size(0)
        total_loss += loss.item() * n
        correct += (logits.argmax(1) == labels).sum().item()
        seen += n
    return {"loss": total_loss / max(seen, 1), "acc": correct / max(seen, 1),
            "time_s": time.perf_counter() - start}


@torch.no_grad()
def predict(model, loader, device, amp: bool = True) -> dict:
    """Run the model over a loader; returns probabilities, labels and pure-model throughput."""
    model.eval()
    probs, labels = [], []
    model_time, frames = 0.0, 0
    for clips, y in loader:
        clips = clips.to(device, non_blocking=True)
        sync(device)
        t0 = time.perf_counter()
        with autocast(device, amp):
            logits = model(clips)
        sync(device)
        model_time += time.perf_counter() - t0
        frames += clips.shape[0] * clips.shape[1]
        probs.append(torch.softmax(logits.float(), 1).cpu())
        labels.append(y)
    probs_t = torch.cat(probs).numpy() if probs else np.zeros((0, 0))
    labels_t = torch.cat(labels).numpy() if labels else np.zeros(0, dtype=int)
    n_clips = len(labels_t)
    return {
        "probs": probs_t,
        "labels": labels_t,
        "clips_per_s": n_clips / model_time if model_time else 0.0,
        "frames_per_s": frames / model_time if model_time else 0.0,
    }


def evaluate(model, loader, criterion, device, amp: bool = True) -> dict:
    out = predict(model, loader, device, amp)
    if len(out["labels"]) == 0:
        return {"loss": float("nan"), "acc": float("nan")}
    probs = torch.from_numpy(out["probs"]).clamp_min(1e-8)
    labels = torch.from_numpy(out["labels"]).long()
    loss = nn.functional.nll_loss(probs.log(), labels).item()
    acc = float((out["probs"].argmax(1) == out["labels"]).mean())
    return {"loss": loss, "acc": acc, **{k: out[k] for k in ("clips_per_s", "frames_per_s")}}
