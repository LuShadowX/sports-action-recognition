"""Player (person) detection with a COCO-pretrained torchvision detector."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torchvision

PERSON_LABEL = 1  # COCO category id for "person" in torchvision detectors


class PersonDetector:
    def __init__(self, device: torch.device, model_name: str = "fasterrcnn_mobilenet_v3_large_fpn",
                 score_thresh: float = 0.6, min_box_area: float = 0.0015):
        self.device = device
        self.score_thresh = score_thresh
        self.min_box_area = min_box_area  # fraction of the frame area
        self.model = torchvision.models.get_model(model_name, weights="DEFAULT",
                                                  box_score_thresh=score_thresh).to(device).eval()

    @torch.no_grad()
    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return ``[N, 5]`` array of (x1, y1, x2, y2, score) person boxes in pixels."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float().div_(255.0)
        out = self.model([x])[0]
        keep = (out["labels"] == PERSON_LABEL) & (out["scores"] >= self.score_thresh)
        boxes = out["boxes"][keep].cpu().numpy()
        scores = out["scores"][keep].cpu().numpy()
        if len(boxes):
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (w * h)
            m = area >= self.min_box_area
            boxes, scores = boxes[m], scores[m]
        return np.concatenate([boxes, scores[:, None]], axis=1) if len(boxes) else np.zeros((0, 5), np.float32)
