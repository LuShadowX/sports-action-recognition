"""PyTorch dataset over cached clips.

Each cached clip is a uint8 ``[cache_frames, S, S, 3]`` array. At training time we use
TSN-style segment sampling (split the cache into ``num_frames`` segments and draw one
random frame per segment) plus clip-consistent spatial augmentation. Clips are returned
as uint8 ``[T, 3, H, W]``; normalisation happens on the GPU inside the model.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from ..utils import load_json
from .classes import class_names, label_key


def segment_indices(n_available: int, num_frames: int, position: float | None,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """One index per temporal segment. ``position`` in [0,1) picks a fixed spot inside each
    segment (evaluation); ``None`` samples randomly (training)."""
    if n_available < num_frames:
        return np.linspace(0, n_available - 1, num_frames).round().astype(int)
    edges = np.linspace(0, n_available, num_frames + 1)
    lo, width = edges[:-1], np.diff(edges)
    if position is None:
        rng = rng or np.random.default_rng()
        offsets = rng.random(num_frames) * width
    else:
        offsets = np.full(num_frames, position) * width
    return np.minimum((lo + offsets).astype(int), n_available - 1)


class ClipDataset(Dataset):
    def __init__(self, records: list[dict], cache_root: str | Path, classes: list[str], key: str,
                 num_frames: int, crop_size: int, train: bool, eval_position: float = 0.5,
                 eval_flip: bool = False, aug: dict | None = None):
        self.records = records
        self.cache_root = Path(cache_root)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.key = key
        self.num_frames = num_frames
        self.crop = crop_size
        self.train = train
        self.eval_position = eval_position
        self.eval_flip = eval_flip
        aug = aug or {}
        self.scale_min = float(aug.get("scale_min", 0.8))
        self.rotate_deg = float(aug.get("rotate_deg", 0.0))
        self.erase_prob = float(aug.get("erase_prob", 0.0))
        self.temporal_min = float(aug.get("temporal_min", 1.0))

    def __len__(self) -> int:
        return len(self.records)

    @property
    def labels(self) -> list[int]:
        return [self.class_to_idx[r[self.key]] for r in self.records]

    def __getitem__(self, i: int):
        rec = self.records[i]
        arr = np.load(self.cache_root / rec["cache_file"], mmap_mode="r")
        if self.train:
            # speed / extent jitter: sample from a random sub-window covering temporal_min..100% of the clip
            n = len(arr)
            span = max(self.num_frames, int(round(n * np.random.uniform(self.temporal_min, 1.0))))
            span = min(span, n)
            start = np.random.randint(0, n - span + 1)
            idx = start + segment_indices(span, self.num_frames, None)
        else:
            idx = segment_indices(len(arr), self.num_frames, self.eval_position)
        frames = np.ascontiguousarray(arr[idx])  # T,S,S,3 uint8
        clip = torch.from_numpy(frames).permute(0, 3, 1, 2)  # T,3,S,S
        clip = self._augment(clip) if self.train else self._center(clip)
        return clip.contiguous(), self.class_to_idx[rec[self.key]]

    def _center(self, clip: torch.Tensor) -> torch.Tensor:
        s = clip.shape[-1]
        o = (s - self.crop) // 2
        clip = clip[..., o:o + self.crop, o:o + self.crop]
        return clip.flip(-1) if self.eval_flip else clip

    def _augment(self, clip: torch.Tensor) -> torch.Tensor:
        s = clip.shape[-1]
        # random-resized-style crop (same box for every frame keeps motion consistent)
        scale = np.random.uniform(self.scale_min, 1.0)
        c = max(self.crop, int(round(s * scale)))
        top, left = np.random.randint(0, s - c + 1, size=2)
        clip = clip[..., top:top + c, left:left + c]
        if c != self.crop:
            clip = torch.nn.functional.interpolate(clip.float(), size=(self.crop, self.crop),
                                                   mode="bilinear", align_corners=False)
        clip = clip.float()
        if np.random.rand() < 0.5:
            clip = clip.flip(-1)
        if self.rotate_deg > 0:
            angle = float(np.random.uniform(-self.rotate_deg, self.rotate_deg))
            clip = TF.rotate(clip, angle, interpolation=TF.InterpolationMode.BILINEAR)
        if np.random.rand() < self.erase_prob:
            # one occluding patch, same place in every frame
            h = int(self.crop * np.random.uniform(0.15, 0.4))
            w = int(self.crop * np.random.uniform(0.15, 0.4))
            y, x = np.random.randint(0, self.crop - h + 1), np.random.randint(0, self.crop - w + 1)
            clip[..., y:y + h, x:x + w] = float(np.random.uniform(0, 255))
        # brightness / contrast / saturation jitter, shared across frames
        b = np.random.uniform(-20, 20)
        ct = np.random.uniform(0.8, 1.2)
        sat = np.random.uniform(0.8, 1.2)
        gray = clip.mean(dim=1, keepdim=True)
        clip = gray + (clip - gray) * sat
        mean = clip.mean()
        clip = (clip - mean) * ct + mean + b
        return clip.clamp_(0, 255).to(torch.uint8)


def load_manifest(cache_dir: str | Path) -> dict:
    path = Path(cache_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python prepare_data.py` first.")
    return load_json(path)


def build_datasets(cfg: dict, eval_position: float = 0.5, eval_flip: bool = False):
    d = cfg["data"]
    manifest = load_manifest(d["cache_dir"])
    classes = class_names(d["label_mode"])
    key = label_key(d["label_mode"])
    common = dict(cache_root=d["cache_dir"], classes=classes, key=key,
                  num_frames=d["num_frames"], crop_size=d["crop_size"])
    by_split = {s: [r for r in manifest["records"] if r["split"] == s] for s in ("train", "val", "test")}
    datasets = {
        "train": ClipDataset(by_split["train"], train=True, aug=d.get("aug"), **common),
        "val": ClipDataset(by_split["val"], train=False, eval_position=eval_position, **common),
        "test": ClipDataset(by_split["test"], train=False, eval_position=eval_position,
                            eval_flip=eval_flip, **common),
    }
    return datasets, classes


def class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    n = len(labels)
    return torch.tensor([n / (num_classes * max(counts.get(c, 0), 1)) for c in range(num_classes)],
                        dtype=torch.float32)
