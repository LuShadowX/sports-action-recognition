"""Annotate a match video: action recognition + player tracking + court-zone analytics.

    # any clip, zones on the image plane
    python infer_video.py --checkpoint runs/cnn_lstm/best.pt --video match.mp4

    # calibrated tennis court (corners: far-left; far-right; near-right; near-left, in pixels)
    python infer_video.py --checkpoint runs/cnn_lstm/best.pt --video match.mp4 \
        --court tennis --court-corners "410,180;870,180;1150,650;130,650" --court-only

Outputs: annotated MP4, summary JSON (clip prediction, action timeline, per-player zone stats),
per-frame tracks CSV and a court occupancy heatmap.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from sportsar.data.video_io import video_info
from sportsar.inference import StreamingActionRecognizer, predict_clip
from sportsar.models import load_checkpoint
from sportsar.tracking.court import CourtZoneMapper, ZoneAnalytics, court_preset, parse_corners
from sportsar.tracking.detector import PersonDetector
from sportsar.tracking.tracker import SortTracker
from sportsar.utils import get_device, save_json, sync

PALETTE = [(255, 99, 71), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128)]


def track_color(tid: int) -> tuple[int, int, int]:
    return PALETTE[tid % len(PALETTE)]


def draw_action_panel(frame, pred, classes, fps: float) -> None:
    """Action label, per-class probability bars and FPS; sized relative to the frame width."""
    s = float(np.clip(frame.shape[1] / 900, 0.4, 1.6))
    S = lambda v: int(round(v * s))  # noqa: E731
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, S(2))
    row = S(18)
    h = S(34) + row * (len(classes) + 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (S(8), S(8)), (S(270), S(8) + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    title = f"{pred.label.upper()} {pred.confidence * 100:.0f}%" if pred else "warming up..."
    cv2.putText(frame, title, (S(16), S(30)), font, 0.65 * s, (255, 255, 255), thick, cv2.LINE_AA)
    for i, c in enumerate(classes):
        y = S(48) + row * i
        p = float(pred.probs[i]) if pred else 0.0
        cv2.rectangle(frame, (S(110), y - S(10)), (S(110) + int(S(150) * p), y + S(2)),
                      (0, 200, 120) if pred and c == pred.label else (150, 150, 150), -1)
        cv2.putText(frame, c[:14], (S(16), y), font, 0.42 * s, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.1f} FPS", (S(16), S(48) + row * len(classes)), font, 0.5 * s,
                (0, 255, 255), 1, cv2.LINE_AA)


def merge_timeline(labels: list[tuple[float, str]]) -> list[dict]:
    segs: list[dict] = []
    for t, lab in labels:
        if segs and segs[-1]["action"] == lab:
            segs[-1]["end_s"] = round(t, 2)
        else:
            segs.append({"action": lab, "start_s": round(t, 2), "end_s": round(t, 2)})
    return segs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", default=None, help="output .mp4 (default outputs/<video>_annotated.mp4)")
    p.add_argument("--court", default="image", help="tennis|badminton|volleyball|basketball|image")
    p.add_argument("--court-corners", default=None, help='"x,y;x,y;x,y;x,y" far-left, far-right, near-right, near-left')
    p.add_argument("--court-only", action="store_true", help="ignore people outside the court (crowd, umpires)")
    p.add_argument("--max-players", type=int, default=0, help="keep only the N most confident detections (0 = all)")
    p.add_argument("--no-track", action="store_true", help="action recognition only")
    p.add_argument("--detector", default="fasterrcnn_mobilenet_v3_large_fpn",
                   help="or fasterrcnn_resnet50_fpn_v2 (slower, more accurate)")
    p.add_argument("--det-thresh", type=float, default=0.6)
    p.add_argument("--min-box-area", type=float, default=0.0015,
                   help="ignore person boxes smaller than this fraction of the frame (lower it for aerial views)")
    p.add_argument("--stride", type=int, default=None, help="CNN every k-th frame (default from config)")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    model, cfg, classes = load_checkpoint(args.checkpoint, device)
    info = video_info(args.video)
    fps_in = info["fps"] if 1 <= info["fps"] <= 240 else 25.0
    out_path = Path(args.out or f"outputs/{Path(args.video).stem}_annotated.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clip_pred = predict_clip(model, args.video, cfg, classes, device, tta=True)
    print(f"Whole-clip prediction: {clip_pred.label} ({clip_pred.confidence * 100:.1f}%)")

    recognizer = StreamingActionRecognizer(model, classes, cfg, device, stride=args.stride)
    corners = parse_corners(args.court_corners) if args.court_corners else None
    if corners is None and args.court != "image":
        print(f"No --court-corners given: falling back to image-plane zones instead of '{args.court}'.")
    spec = court_preset(args.court if corners is not None else "image", (info["width"], info["height"]))
    mapper = CourtZoneMapper(spec, corners)
    analytics = ZoneAnalytics(mapper, fps_in)
    detector = tracker = None
    if not args.no_track:
        detector = PersonDetector(device, args.detector, args.det_thresh, args.min_box_area)
        tracker = SortTracker(max_age=int(fps_in))

    cap = cv2.VideoCapture(args.video)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in,
                             (info["width"], info["height"]))
    csv_f = open(out_path.with_suffix(".tracks.csv"), "w", newline="")
    rows = csv.writer(csv_f)
    rows.writerow(["frame", "time_s", "track_id", "x1", "y1", "x2", "y2", "court_x", "court_y", "zone", "action"])

    timeline, timings = [], {"action": 0.0, "tracking": 0.0, "total": 0.0}
    fps_ema, n = None, 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        t_start = time.perf_counter()
        pred = recognizer.update(frame)
        sync(device)
        t_action = time.perf_counter()

        tracks = []
        if detector is not None:
            dets = detector(frame)
            if args.court_only and len(dets):
                feet = np.stack([(dets[:, 0] + dets[:, 2]) / 2, dets[:, 3]], 1)
                dets = dets[[mapper.in_play_area(pt) for pt in mapper.to_court(feet)]]
            if args.max_players and len(dets) > args.max_players:
                dets = dets[np.argsort(-dets[:, 4])[:args.max_players]]
            tracks = tracker.update(dets)
        t_track = time.perf_counter()

        time_s = n / fps_in
        action = pred.label if pred else ""
        if pred:
            timeline.append((time_s, pred.label))
        highlight, minimap_players = {}, []
        for trk in tracks:
            court_xy = mapper.to_court([trk.foot_point])[0]
            zi, zname = analytics.update(trk.id, court_xy)
            color = track_color(trk.id)
            if zi >= 0:
                highlight[zi] = color
            minimap_players.append((trk.id, tuple(court_xy), analytics.trails[trk.id]))
            b = trk.box
            rows.writerow([n, f"{time_s:.3f}", trk.id, *[f"{v:.1f}" for v in b], f"{court_xy[0]:.2f}",
                           f"{court_xy[1]:.2f}", zname, action])

        vis = mapper.draw_overlay(frame, highlight)
        for trk in tracks:
            b = trk.box.astype(int)
            color = track_color(trk.id)
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), color, 2)
            zname = analytics.last_zone.get(trk.id, "")
            label = f"P{trk.id} | {zname}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (b[0], b[1] - th - 6), (b[0] + tw + 4, b[1]), color, -1)
            cv2.putText(vis, label, (b[0] + 2, b[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(vis, tuple(int(v) for v in trk.foot_point), 4, color, -1)

        t_end = time.perf_counter()
        inst = 1.0 / max(t_end - t_start, 1e-6)
        fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst
        draw_action_panel(vis, pred, classes, fps_ema)
        if detector is not None:
            mm = mapper.draw_minimap(minimap_players, height=min(220, info["height"] // 2),
                                     colors={t.id: track_color(t.id) for t in tracks})
            mh, mw = mm.shape[:2]
            if mw < info["width"] and mh < info["height"]:
                vis[-mh - 8:-8, -mw - 8:-8] = mm
        writer.write(vis)

        timings["action"] += t_action - t_start
        timings["tracking"] += t_track - t_action
        timings["total"] += t_end - t_start
        n += 1

    cap.release()
    writer.release()
    csv_f.close()

    stream_probs = np.zeros(len(classes))
    for _, lab in timeline:
        stream_probs[classes.index(lab)] += 1
    summary = {
        "video": args.video,
        "video_info": info,
        "frames_processed": n,
        "clip_prediction": {"action": clip_pred.label, "confidence": round(clip_pred.confidence, 4),
                            "probs": {c: round(float(v), 4) for c, v in zip(classes, clip_pred.probs)}},
        "streaming_action_share": {c: round(float(v / max(len(timeline), 1)), 4) for c, v in zip(classes, stream_probs)},
        "action_timeline": merge_timeline(timeline),
        "court": {"preset": spec.name, "calibrated": mapper.calibrated, "unit": spec.unit, "zones": mapper.zone_names},
        "players": analytics.summary(min_frames=int(fps_in // 2)),
        "speed": {
            "processing_fps": round(n / timings["total"], 2) if timings["total"] else 0.0,
            "action_ms_per_frame": round(1e3 * timings["action"] / max(n, 1), 2),
            "tracking_ms_per_frame": round(1e3 * timings["tracking"] / max(n, 1), 2),
            "cnn_stride": recognizer.stride,
        },
    }
    save_json(summary, out_path.with_suffix(".summary.json"))
    if detector is not None:
        analytics.save_heatmap(str(out_path.with_suffix(".heatmap.png")))
    print(f"Processed {n} frames at {summary['speed']['processing_fps']} FPS "
          f"(action {summary['speed']['action_ms_per_frame']} ms + tracking {summary['speed']['tracking_ms_per_frame']} ms per frame)")
    print(f"Players tracked: {len(summary['players'])}")
    print(f"Saved {out_path} (+ .summary.json, .tracks.csv{', .heatmap.png' if detector else ''})")


if __name__ == "__main__":
    main()
