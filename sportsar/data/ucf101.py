"""Index UCF101 videos from any mirror layout and build leakage-free splits.

UCF101 file names encode class, *group* (same actors / same original video) and clip:
``v_TennisSwing_g08_c03.avi``. Clips from the same group are near-duplicates, so the
official splits hold out whole groups for testing. We follow that protocol and also
carve the validation set out of *training groups*, so train / val / test never share actors.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .classes import FINE_TO_COARSE

UCF_NAME_RE = re.compile(r"v_(?P<cls>[A-Za-z]+)_g(?P<group>\d+)_c(?P<clip>\d+)", re.IGNORECASE)
VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".mov", ".webm", ".mpg", ".mpeg"}

# Official UCF101 test groups per split (train = remaining groups 1..25).
OFFICIAL_TEST_GROUPS = {1: range(1, 8), 2: range(8, 15), 3: range(15, 22)}
NUM_GROUPS = 25


@dataclass
class ClipRecord:
    clip_id: str
    video_path: str
    fine_class: str
    coarse_action: str
    group: int | None
    split: str = ""
    cache_file: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def index_videos(root: str | Path, fine_classes: list[str] | None = None) -> list[ClipRecord]:
    """Recursively find videos of the requested UCF101 classes under ``root``."""
    fine_classes = fine_classes or sorted(FINE_TO_COARSE)
    lookup = {c.lower(): c for c in fine_classes}
    records: list[ClipRecord] = []
    seen: set[str] = set()
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in VIDEO_EXTS or not path.is_file():
            continue
        match = UCF_NAME_RE.search(path.stem)
        if match:
            raw_cls, group = match["cls"], int(match["group"])
            clip_id = f"v_{raw_cls}_g{int(match['group']):02d}_c{int(match['clip']):02d}"
        else:  # non-standard naming: fall back to the parent folder as the class
            raw_cls, group, clip_id = path.parent.name, None, path.stem
        fine = lookup.get(raw_cls.lower())
        if fine is None or clip_id.lower() in seen:  # some mirrors ship duplicates
            continue
        seen.add(clip_id.lower())
        records.append(ClipRecord(clip_id, str(path), fine, FINE_TO_COARSE[fine], group))
    return records


def _stable_fraction(text: str) -> float:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def assign_splits(records: list[ClipRecord], split_id: int = 1, val_fraction: float = 0.15,
                  seed: int = 42) -> list[ClipRecord]:
    if split_id not in OFFICIAL_TEST_GROUPS:
        raise ValueError(f"split must be 1, 2 or 3, got {split_id}")
    test_groups = set(OFFICIAL_TEST_GROUPS[split_id])
    train_groups = sorted(set(range(1, NUM_GROUPS + 1)) - test_groups)
    n_val = max(1, round(len(train_groups) * val_fraction))
    val_groups = set(random.Random(seed).sample(train_groups, n_val))

    test_frac = len(test_groups) / NUM_GROUPS
    for r in records:
        if r.group is not None:
            r.split = "test" if r.group in test_groups else "val" if r.group in val_groups else "train"
        else:  # unknown group: deterministic hash split with the same proportions
            u = _stable_fraction(f"{seed}:{r.clip_id}")
            r.split = "test" if u < test_frac else "val" if u < test_frac + (1 - test_frac) * val_fraction else "train"
    return records


def split_summary(records: list[ClipRecord], key: str = "coarse_action") -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for r in records:
        summary.setdefault(r.split, {}).setdefault(getattr(r, key), 0)
        summary[r.split][getattr(r, key)] += 1
    return {s: dict(sorted(v.items())) for s, v in sorted(summary.items())}
