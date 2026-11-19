"""Court calibration (homography), zone assignment, minimap and movement analytics.

Calibrate by giving the 4 image-pixel corners of the playing area in this order:
far-left, far-right, near-right, near-left. They map to court metres
(0,0), (W,0), (W,L), (0,L). Each player's foot point is projected onto the court plane,
so zones, distances and speeds are measured in real metres, independent of camera angle.

Without corners the frame itself is used as the plane ("image" preset, units = pixels).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CourtSpec:
    name: str
    width: float                 # across the court (x)
    length: float                # along the court, far baseline -> near baseline (y)
    x_edges: list[float]
    y_edges: list[float]
    col_names: list[str]
    row_names: list[str]
    unit: str = "m"
    lines: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)


def _thirds(w: float) -> list[float]:
    return [0.0, w / 3, 2 * w / 3, w]


def court_preset(name: str, frame_wh: tuple[int, int] | None = None) -> CourtSpec:
    lr = ["Left", "Center", "Right"]
    if name == "tennis":
        W, L = 10.97, 23.77
        return CourtSpec("tennis", W, L, _thirds(W), [0, 5.485, L / 2, L - 5.485, L], lr,
                         ["Far back court", "Far service box", "Near service box", "Near back court"],
                         lines=[((0, L / 2), (W, L / 2)), ((1.37, 5.485), (W - 1.37, 5.485)),
                                ((1.37, L - 5.485), (W - 1.37, L - 5.485)), ((1.37, 0), (1.37, L)),
                                ((W - 1.37, 0), (W - 1.37, L)), ((W / 2, 5.485), (W / 2, L - 5.485))])
    if name == "badminton":
        W, L = 6.10, 13.40
        return CourtSpec("badminton", W, L, _thirds(W), [0, 2.23, 4.72, L / 2, L - 4.72, L - 2.23, L], lr,
                         ["Far rear", "Far mid", "Far front", "Near front", "Near mid", "Near rear"],
                         lines=[((0, L / 2), (W, L / 2)), ((0, L / 2 - 1.98), (W, L / 2 - 1.98)),
                                ((0, L / 2 + 1.98), (W, L / 2 + 1.98))])
    if name == "volleyball":
        W, L = 9.0, 18.0
        return CourtSpec("volleyball", W, L, _thirds(W), [0, 6, 9, 12, 18], lr,
                         ["Far back row", "Far front row", "Near front row", "Near back row"],
                         lines=[((0, 9), (W, 9)), ((0, 6), (W, 6)), ((0, 12), (W, 12))])
    if name == "basketball":
        W, L = 15.0, 28.0
        return CourtSpec("basketball", W, L, [0, 5.05, 9.95, W], [0, 5.8, L / 2, L - 5.8, L], lr,
                         ["Far paint", "Far half", "Near half", "Near paint"],
                         lines=[((0, L / 2), (W, L / 2)), ((5.05, 0), (5.05, 5.8)), ((9.95, 0), (9.95, 5.8)),
                                ((5.05, 5.8), (9.95, 5.8)), ((5.05, L), (5.05, L - 5.8)),
                                ((9.95, L), (9.95, L - 5.8)), ((5.05, L - 5.8), (9.95, L - 5.8))])
    if name == "image":
        if frame_wh is None:
            raise ValueError("The 'image' preset needs the frame size")
        W, L = float(frame_wh[0]), float(frame_wh[1])
        return CourtSpec("image", W, L, _thirds(W), _thirds(L), lr, ["Top", "Middle", "Bottom"], unit="px")
    raise ValueError(f"Unknown court preset {name!r} (tennis|badminton|volleyball|basketball|image)")


def parse_corners(text: str) -> np.ndarray:
    """``"x1,y1;x2,y2;x3,y3;x4,y4"`` -> float32 ``[4,2]`` (far-left, far-right, near-right, near-left)."""
    pts = [tuple(float(v) for v in p.split(",")) for p in text.strip().split(";") if p.strip()]
    if len(pts) != 4 or any(len(p) != 2 for p in pts):
        raise ValueError("court corners must be 4 'x,y' pairs separated by ';'")
    return np.array(pts, dtype=np.float32)


class CourtZoneMapper:
    def __init__(self, spec: CourtSpec, image_corners: np.ndarray | None = None, margin: float = 1.5):
        self.spec = spec
        self.calibrated = image_corners is not None
        court_corners = np.array([[0, 0], [spec.width, 0], [spec.width, spec.length], [0, spec.length]],
                                 dtype=np.float32)
        img = image_corners if image_corners is not None else court_corners
        self.H = cv2.getPerspectiveTransform(img.astype(np.float32), court_corners)
        self.H_inv = np.linalg.inv(self.H)
        self.image_corners = img.astype(np.float32)
        self.margin = margin if spec.unit == "m" else 0.0  # players may stand behind the baseline

    # ------------------------------------------------------------ geometry
    def to_court(self, pts_px) -> np.ndarray:
        pts = np.asarray(pts_px, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, pts_court) -> np.ndarray:
        pts = np.asarray(pts_court, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    def in_play_area(self, pt_court) -> bool:
        x, y = pt_court
        m = self.margin
        return -m <= x <= self.spec.width + m and -m * 3 <= y <= self.spec.length + m * 3

    def zone_of(self, pt_court) -> tuple[int, str]:
        """Zone index (row * n_cols + col) and readable name; -1 / 'Out' if off court."""
        if not self.in_play_area(pt_court):
            return -1, "Out"
        x = float(np.clip(pt_court[0], 0, self.spec.width - 1e-6))
        y = float(np.clip(pt_court[1], 0, self.spec.length - 1e-6))
        col = int(np.searchsorted(self.spec.x_edges, x, side="right") - 1)
        row = int(np.searchsorted(self.spec.y_edges, y, side="right") - 1)
        col = min(max(col, 0), len(self.spec.col_names) - 1)
        row = min(max(row, 0), len(self.spec.row_names) - 1)
        return row * len(self.spec.col_names) + col, f"{self.spec.row_names[row]} / {self.spec.col_names[col]}"

    @property
    def zone_names(self) -> list[str]:
        return [f"{r} / {c}" for r in self.spec.row_names for c in self.spec.col_names]

    # ------------------------------------------------------------- drawing
    def draw_overlay(self, frame: np.ndarray, highlight: dict[int, tuple[int, int, int]] | None = None,
                     alpha: float = 0.25) -> np.ndarray:
        s = self.spec
        overlay = frame.copy()
        for zi, color in (highlight or {}).items():
            r, c = divmod(zi, len(s.col_names))
            quad = [(s.x_edges[c], s.y_edges[r]), (s.x_edges[c + 1], s.y_edges[r]),
                    (s.x_edges[c + 1], s.y_edges[r + 1]), (s.x_edges[c], s.y_edges[r + 1])]
            cv2.fillPoly(overlay, [self.to_image(quad).astype(np.int32)], color)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        grid_color = (255, 255, 255) if self.calibrated else (200, 200, 200)
        for x in s.x_edges:
            p = self.to_image([(x, 0), (x, s.length)]).astype(int)
            cv2.line(frame, tuple(p[0]), tuple(p[1]), grid_color, 1, cv2.LINE_AA)
        for y in s.y_edges:
            p = self.to_image([(0, y), (s.width, y)]).astype(int)
            cv2.line(frame, tuple(p[0]), tuple(p[1]), grid_color, 1, cv2.LINE_AA)
        if self.calibrated:
            cv2.polylines(frame, [self.image_corners.astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
        return frame

    def draw_minimap(self, players: list[tuple[int, tuple[float, float], list]], height: int = 220,
                     colors: dict[int, tuple[int, int, int]] | None = None) -> np.ndarray:
        """Top-down court view. ``players``: (track_id, court_xy, court_trail)."""
        s = self.spec
        pad = 12
        scale = (height - 2 * pad) / (s.length + 2 * self.margin * 3)
        width = int(s.width * scale + 2 * pad + 2 * self.margin * scale)
        img = np.full((height, width, 3), (60, 110, 60), np.uint8)
        ox, oy = pad + self.margin * scale, pad + self.margin * 3 * scale

        def P(x, y):
            return int(ox + x * scale), int(oy + y * scale)

        cv2.rectangle(img, P(0, 0), P(s.width, s.length), (255, 255, 255), 2)
        for a, b in s.lines:
            cv2.line(img, P(*a), P(*b), (230, 230, 230), 1, cv2.LINE_AA)
        for x in s.x_edges[1:-1]:
            cv2.line(img, P(x, 0), P(x, s.length), (120, 160, 120), 1)
        for y in s.y_edges[1:-1]:
            cv2.line(img, P(0, y), P(s.width, y), (120, 160, 120), 1)
        for tid, xy, trail in players:
            color = (colors or {}).get(tid, (0, 200, 255))
            pts = [P(*p) for p in trail]
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)
            cv2.circle(img, P(*xy), 5, color, -1, cv2.LINE_AA)
            cv2.putText(img, str(tid), (P(*xy)[0] + 6, P(*xy)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return img


class ZoneAnalytics:
    """Per-player time-in-zone, zone transitions, distance covered and speed."""

    def __init__(self, mapper: CourtZoneMapper, fps: float, max_step: float | None = None):
        self.mapper = mapper
        self.fps = fps
        # ignore physically impossible jumps (ID switches / detector noise): ~12 m/s sprint ceiling
        self.max_step = max_step if max_step is not None else (12.0 / fps if mapper.spec.unit == "m" else np.inf)
        self.zone_frames: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.distance: dict[int, float] = defaultdict(float)
        self.last_pos: dict[int, np.ndarray] = {}
        self.last_zone: dict[int, str] = {}
        self.transitions: dict[int, int] = defaultdict(int)
        self.frames_seen: dict[int, int] = defaultdict(int)
        self.trails: dict[int, list] = defaultdict(list)
        self.heat_points: list[tuple[float, float]] = []

    def update(self, track_id: int, court_xy: np.ndarray) -> tuple[int, str]:
        zi, zname = self.mapper.zone_of(court_xy)
        self.zone_frames[track_id][zname] += 1
        self.frames_seen[track_id] += 1
        if track_id in self.last_pos:
            step = float(np.linalg.norm(court_xy - self.last_pos[track_id]))
            if step <= self.max_step:
                self.distance[track_id] += step
        if self.last_zone.get(track_id) not in (None, zname):
            self.transitions[track_id] += 1
        self.last_pos[track_id] = np.asarray(court_xy, dtype=np.float64)
        self.last_zone[track_id] = zname
        trail = self.trails[track_id]
        trail.append((float(court_xy[0]), float(court_xy[1])))
        if len(trail) > 90:
            del trail[0]
        if zi >= 0:
            self.heat_points.append((float(court_xy[0]), float(court_xy[1])))
        return zi, zname

    def summary(self, min_frames: int = 5) -> dict:
        unit = self.mapper.spec.unit
        out = {}
        for tid, n in sorted(self.frames_seen.items()):
            if n < min_frames:
                continue
            secs = n / self.fps
            zones = {z: round(c / self.fps, 2) for z, c in
                     sorted(self.zone_frames[tid].items(), key=lambda kv: -kv[1])}
            out[str(tid)] = {
                "time_on_screen_s": round(secs, 2),
                "dominant_zone": next(iter(zones), None),
                "time_in_zone_s": zones,
                "zone_transitions": self.transitions[tid],
                f"distance_{unit}": round(self.distance[tid], 2),
                f"avg_speed_{unit}_per_s": round(self.distance[tid] / secs, 2) if secs else 0.0,
            }
        return out

    def save_heatmap(self, path: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        s = self.mapper.spec
        fig, ax = plt.subplots(figsize=(4, 4 * s.length / s.width if s.width else 6))
        if self.heat_points:
            pts = np.array(self.heat_points)
            ax.hist2d(pts[:, 0], pts[:, 1], bins=[24, int(24 * s.length / s.width) or 24],
                      range=[[0, s.width], [0, s.length]], cmap="hot")
        for x in s.x_edges:
            ax.axvline(x, color="cyan", lw=0.6)
        for y in s.y_edges:
            ax.axhline(y, color="cyan", lw=0.6)
        ax.set_xlim(0, s.width)
        ax.set_ylim(s.length, 0)
        ax.set_aspect("equal")
        ax.set_title(f"Player occupancy ({s.name})")
        ax.set_xlabel(s.unit)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
