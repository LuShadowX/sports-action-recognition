"""Download (optional), index, split and frame-cache the UCF101 sports subset.

    python prepare_data.py --download                 # Kaggle via kagglehub
    python prepare_data.py --data-root /path/to/UCF-101
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sportsar.config import add_config_args, load_config
from sportsar.data.classes import FINE_CLASSES
from sportsar.data.ucf101 import assign_splits, index_videos, split_summary
from sportsar.data.video_io import read_video_frames
from sportsar.utils import save_json

DEFAULT_KAGGLE_SLUG = "matthewjansen/ucf101-action-recognition"


def _cache_one(args):
    video_path, out_path, n_frames, size = args
    if Path(out_path).exists():
        return video_path, None
    try:
        frames = read_video_frames(video_path, n_frames, size)
        tmp = str(out_path) + ".tmp.npy"
        np.save(tmp, frames)
        os.replace(tmp, out_path)
        return video_path, None
    except Exception as e:  # corrupt clip: report and skip
        return video_path, str(e)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--data-root", help="folder containing UCF101 videos (overrides data.root)")
    p.add_argument("--download", action="store_true", help="download UCF101 from Kaggle with kagglehub")
    p.add_argument("--kaggle-slug", default=DEFAULT_KAGGLE_SLUG)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    p.add_argument("--max-per-class", type=int, default=None, help="cap clips per class (quick experiments)")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    d = cfg["data"]

    root = args.data_root or d["root"]
    if args.download:
        import kagglehub
        root = kagglehub.dataset_download(args.kaggle_slug)
        print(f"Downloaded {args.kaggle_slug} -> {root}")

    print(f"Indexing videos under {root} ...")
    records = index_videos(root, FINE_CLASSES)
    found = {r.fine_class for r in records}
    missing = sorted(set(FINE_CLASSES) - found)
    if not records:
        raise SystemExit(f"No UCF101 sports clips found under {root}. Check the path / download.")
    if missing:
        print(f"WARNING: classes not found in dataset: {missing}")

    if args.max_per_class:
        kept, per = [], {}
        for r in records:
            if per.get(r.fine_class, 0) < args.max_per_class:
                kept.append(r)
                per[r.fine_class] = per.get(r.fine_class, 0) + 1
        records = kept

    assign_splits(records, d["split"], d["val_fraction"], cfg["seed"])

    cache_dir = Path(d["cache_dir"])
    frames_dir = cache_dir / f"frames_{d['cache_frames']}x{d['frame_size']}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for r in records:
        r.cache_file = str((frames_dir / f"{r.clip_id}.npy").relative_to(cache_dir))
        jobs.append((r.video_path, str(cache_dir / r.cache_file), d["cache_frames"], d["frame_size"]))

    print(f"Caching {len(jobs)} clips ({d['cache_frames']} frames @ {d['frame_size']}px) with {args.workers} workers ...")
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_cache_one, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="clip"):
            video_path, err = fut.result()
            if err:
                errors.append((video_path, err))
    if errors:
        print(f"Skipped {len(errors)} unreadable clips, e.g. {errors[:3]}")
        bad = {v for v, _ in errors}
        records = [r for r in records if r.video_path not in bad]

    manifest = {
        "data_root": str(root),
        "cache_frames": d["cache_frames"],
        "frame_size": d["frame_size"],
        "split": d["split"],
        "records": [r.to_dict() for r in records],
    }
    save_json(manifest, cache_dir / "manifest.json")
    print("\nClips per split / action:")
    for split, counts in split_summary(records).items():
        print(f"  {split:<5} {sum(counts.values()):>5}  {counts}")
    print(f"\nManifest written to {cache_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
