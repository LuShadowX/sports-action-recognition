"""Frame-rate benchmark.

1. Model-only throughput on synthetic clips (batch 1 latency + batched throughput, FP32 vs AMP).
2. Optional real-video streaming FPS (decode + preprocess + CNN/LSTM [+ detection & tracking]).

    python benchmark_fps.py --checkpoint runs/cnn_lstm/best.pt --video some_clip.avi
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from sportsar.engine import autocast
from sportsar.inference import StreamingActionRecognizer
from sportsar.models import load_checkpoint
from sportsar.utils import get_device, save_json, sync


@torch.no_grad()
def bench_model(model, device, batch: int, frames: int, size: int, amp: bool, iters: int, warmup: int = 5) -> dict:
    x = torch.randint(0, 256, (batch, frames, 3, size, size), dtype=torch.uint8, device=device)
    for _ in range(warmup):
        with autocast(device, amp):
            model(x)
    sync(device)
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with autocast(device, amp):
            model(x)
        sync(device)
        lat.append(time.perf_counter() - t0)
    lat = np.array(lat)
    return {"batch": batch, "amp": amp, "latency_ms_p50": float(np.median(lat) * 1e3),
            "latency_ms_p95": float(np.percentile(lat, 95) * 1e3),
            "clips_per_s": float(batch / lat.mean()), "frames_per_s": float(batch * frames / lat.mean())}


def bench_stream(model, classes, cfg, device, video: str, track: bool, max_frames: int) -> dict:
    rec = StreamingActionRecognizer(model, classes, cfg, device, amp=device.type == "cuda")
    detector = tracker = None
    if track:
        from sportsar.tracking.detector import PersonDetector
        from sportsar.tracking.tracker import SortTracker
        detector, tracker = PersonDetector(device), SortTracker()
    cap = cv2.VideoCapture(video)
    n, t_total = 0, 0.0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        rec.update(frame)
        if detector is not None:
            tracker.update(detector(frame))
        sync(device)
        t_total += time.perf_counter() - t0
        n += 1
    cap.release()
    return {"video": video, "frames": n, "tracking": track, "stride": rec.stride,
            "processing_fps": n / t_total if t_total else 0.0}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video", default=None)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    model, cfg, classes = load_checkpoint(args.checkpoint, device)
    T, S = cfg["data"]["num_frames"], cfg["data"]["crop_size"]
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    print(f"Device: {gpu} | backbone {cfg['model']['backbone']} | {T} frames @ {S}px")

    results = {"device": gpu, "backbone": cfg["model"]["backbone"], "num_frames": T, "crop": S, "model": []}
    amps = [False, True] if device.type == "cuda" else [False]
    print(f"\n{'batch':>5} {'amp':>5} {'p50 ms':>9} {'p95 ms':>9} {'clips/s':>9} {'frames/s':>10}")
    for amp in amps:
        for b in (1, 8):
            r = bench_model(model, device, b, T, S, amp, args.iters)
            results["model"].append(r)
            print(f"{b:>5} {str(amp):>5} {r['latency_ms_p50']:>9.1f} {r['latency_ms_p95']:>9.1f} "
                  f"{r['clips_per_s']:>9.1f} {r['frames_per_s']:>10.0f}")

    if args.video:
        results["stream"] = []
        for track in (False, True):
            r = bench_stream(model, classes, cfg, device, args.video, track, args.max_frames)
            results["stream"].append(r)
            print(f"\nStreaming {'action+tracking' if track else 'action only    '}: "
                  f"{r['processing_fps']:.1f} FPS over {r['frames']} frames (CNN every {r['stride']} frames)")

    out = Path(args.out or Path(args.checkpoint).parent / "fps_benchmark.json")
    save_json(results, out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
