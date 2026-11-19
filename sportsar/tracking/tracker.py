"""SORT-style multi-object tracker: constant-velocity Kalman filter + Hungarian IoU matching."""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def _box_to_z(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)], dtype=np.float64)


def _x_to_box(x: np.ndarray) -> np.ndarray:
    s, r = max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class KalmanBoxTrack:
    """State: [cx, cy, area, aspect, vx, vy, v_area]; measurement: [cx, cy, area, aspect]."""

    F = np.eye(7)
    F[0, 4] = F[1, 5] = F[2, 6] = 1.0
    H = np.eye(4, 7)

    def __init__(self, box: np.ndarray, track_id: int, trail_len: int = 60):
        self.x = np.zeros(7)
        self.x[:4] = _box_to_z(box)
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01
        self.Q[6, 6] *= 0.01
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.id = track_id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.score = float(box[4]) if len(box) > 4 else 1.0
        self.trail: deque = deque(maxlen=trail_len)

    def predict(self) -> np.ndarray:
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.box

    def update(self, det: np.ndarray) -> None:
        z = _box_to_z(det)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.hits += 1
        self.time_since_update = 0
        self.score = float(det[4]) if len(det) > 4 else self.score

    @property
    def box(self) -> np.ndarray:
        return _x_to_box(self.x)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the player touches the court plane."""
        b = self.box
        return float((b[0] + b[2]) / 2), float(b[3])


class SortTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_hits: int = 3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[KalmanBoxTrack] = []
        self._next_id = 1
        self.frame_count = 0

    def update(self, dets: np.ndarray) -> list[KalmanBoxTrack]:
        """``dets``: ``[N,5]`` (x1,y1,x2,y2,score). Returns confirmed tracks seen this frame."""
        self.frame_count += 1
        predicted = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        valid = np.all(np.isfinite(predicted), axis=1)
        self.tracks = [t for t, v in zip(self.tracks, valid) if v]
        predicted = predicted[valid]

        matched_t, matched_d = set(), set()
        if len(self.tracks) and len(dets):
            iou = iou_matrix(predicted, dets[:, :4])
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:
                    self.tracks[r].update(dets[c])
                    matched_t.add(r)
                    matched_d.add(c)

        for i, det in enumerate(dets):
            if i not in matched_d:
                self.tracks.append(KalmanBoxTrack(det, self._next_id))
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        active = [t for t in self.tracks if t.time_since_update == 0 and
                  (t.hits >= self.min_hits or self.frame_count <= self.min_hits)]
        for t in active:
            t.trail.append(t.foot_point)
        return active
