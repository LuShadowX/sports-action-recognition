"""Clip-level and streaming (real-time) action recognition."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from .data.dataset import segment_indices
from .data.video_io import center_crop, read_video_frames, resize_square
from .engine import autocast


@dataclass
class Prediction:
    label: str
    confidence: float
    probs: np.ndarray


@torch.no_grad()
def predict_clip(model, video_path: str, cfg: dict, classes: list[str], device: torch.device,
                 tta: bool = True) -> Prediction:
    """Classify a whole short clip exactly the way the test set is evaluated."""
    d = cfg["data"]
    frames = read_video_frames(video_path, d["cache_frames"], d["frame_size"])
    frames = center_crop(frames, d["crop_size"])
    views = []
    for pos in ((0.25, 0.75) if tta else (0.5,)):
        clip = frames[segment_indices(len(frames), d["num_frames"], pos)]
        views.append(clip)
        if tta:
            views.append(clip[:, :, ::-1])
    x = torch.from_numpy(np.ascontiguousarray(np.stack(views))).permute(0, 1, 4, 2, 3).to(device)
    with autocast(device, True):
        logits = model(x)
    probs = torch.softmax(logits.float(), 1).mean(0).cpu().numpy()
    i = int(probs.argmax())
    return Prediction(classes[i], float(probs[i]), probs)


class StreamingActionRecognizer:
    """Runs the CNN on every ``stride``-th frame, keeps the last ``window`` features and
    re-runs only the (cheap) LSTM head, so cost per frame is ~1/stride of a CNN pass."""

    def __init__(self, model, classes: list[str], cfg: dict, device: torch.device,
                 stride: int | None = None, window: int | None = None, ema: float | None = None,
                 amp: bool = True):
        d, inf = cfg["data"], cfg.get("inference", {})
        self.model = model.eval()
        self.classes = classes
        self.device = device
        self.frame_size = d["frame_size"]
        self.crop = d["crop_size"]
        self.stride = stride or inf.get("stream_stride", 8)
        self.window = window or d["num_frames"]
        self.ema = inf.get("ema", 0.6) if ema is None else ema
        self.amp = amp
        self.min_feats = max(4, self.window // 2)
        self.feats: deque = deque(maxlen=self.window)
        self.frame_idx = 0
        self.smoothed: np.ndarray | None = None
        self.current: Prediction | None = None

    def reset(self) -> None:
        self.feats.clear()
        self.frame_idx = 0
        self.smoothed = None
        self.current = None

    @torch.no_grad()
    def update(self, frame_bgr: np.ndarray) -> Prediction | None:
        self.frame_idx += 1
        if (self.frame_idx - 1) % self.stride:
            return self.current
        img = center_crop(resize_square(frame_bgr, self.frame_size), self.crop)
        x = torch.from_numpy(np.ascontiguousarray(img)).to(self.device).permute(2, 0, 1)[None, None]
        with autocast(self.device, self.amp):
            f = self.model.encode_frames(x)[0, 0]
            self.feats.append(f)
            if len(self.feats) < self.min_feats:
                return self.current
            logits, _ = self.model.classify_features(torch.stack(list(self.feats))[None])
        probs = torch.softmax(logits.float(), 1)[0].cpu().numpy()
        self.smoothed = probs if self.smoothed is None else self.ema * self.smoothed + (1 - self.ema) * probs
        i = int(self.smoothed.argmax())
        self.current = Prediction(self.classes[i], float(self.smoothed[i]), self.smoothed.copy())
        return self.current
