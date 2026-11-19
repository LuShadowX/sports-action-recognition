from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def resize_square(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR frame -> RGB ``size x size`` (full field of view kept, like at training time)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def read_video_frames(path: str | Path, num_frames: int, size: int) -> np.ndarray:
    """Decode a clip and return ``num_frames`` frames spread uniformly over its whole length.

    Frames are resized while decoding so memory stays small. Returns uint8 ``[N, size, size, 3]`` RGB.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(resize_square(frame, size))
    cap.release()
    if not frames:
        raise IOError(f"No decodable frames in: {path}")
    idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    return np.stack([frames[i] for i in idx])


def center_crop(frames: np.ndarray, crop: int) -> np.ndarray:
    """Center-crop ``[..., H, W, C]`` arrays."""
    h, w = frames.shape[-3], frames.shape[-2]
    top, left = (h - crop) // 2, (w - crop) // 2
    return frames[..., top:top + crop, left:left + crop, :]


def video_info(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info
