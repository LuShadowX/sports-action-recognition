"""2D-CNN frame encoder + LSTM temporal model.

    clip [B,T,3,H,W] --CNN (shared weights, per frame)--> [B,T,D]
                     --LSTM (bi-directional, 2 layers)--> [B,T,H']
                     --temporal attention pooling------> [B,H']
                     --classifier---------------------> [B,num_classes]

``encode_frames`` and ``classify_features`` are exposed separately so streaming inference
can run the CNN once per new frame and re-use cached features for the LSTM window.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int, list[nn.Module]]:
    """Return (feature extractor, feature dim, stages ordered from LAST to FIRST)."""
    weights = "DEFAULT" if pretrained else None
    m = torchvision.models.get_model(name, weights=weights)
    if name.startswith("resnet"):
        dim = m.fc.in_features
        m.fc = nn.Identity()
        stages = [m.layer4, m.layer3, m.layer2, m.layer1, nn.ModuleList([m.conv1, m.bn1])]
    elif name.startswith("mobilenet_v3"):
        dim = m.classifier[0].in_features
        m.classifier = nn.Identity()
        stages = list(reversed(list(m.features)))
    elif name.startswith("efficientnet"):
        dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
        stages = list(reversed(list(m.features)))
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    return m, dim, stages


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)  # B,T
        return (x * w.unsqueeze(-1)).sum(1), w


class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet18", pretrained: bool = True,
                 trainable_backbone_stages: int = 1, hidden_size: int = 256, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.4, pooling: str = "attention"):
        super().__init__()
        self.encoder, feat_dim, stages = build_backbone(backbone, pretrained)
        self._set_trainable(stages, trainable_backbone_stages)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        self.feat_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(feat_dim, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.pooling = pooling
        self.attn = TemporalAttentionPool(out_dim) if pooling == "attention" else None
        self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Dropout(dropout), nn.Linear(out_dim, num_classes))
        self.feat_dim = feat_dim

    # ------------------------------------------------------------------ setup
    def _set_trainable(self, stages: list[nn.Module], n: int) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = n < 0
        if n > 0:
            for stage in stages[:n]:
                for p in stage.parameters():
                    p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        # frozen BatchNorm layers keep their ImageNet running statistics
        for m in self.encoder.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(
                    p.requires_grad for p in m.parameters()):
                m.eval()
        return self

    def backbone_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def head_parameters(self):
        enc = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in enc and p.requires_grad]

    # ---------------------------------------------------------------- forward
    def encode_frames(self, clip: torch.Tensor) -> torch.Tensor:
        """uint8 or float(0-1) ``[B,T,3,H,W]`` -> per-frame features ``[B,T,D]``."""
        b, t = clip.shape[:2]
        x = clip.flatten(0, 1)
        x = x.float().div_(255.0) if x.dtype == torch.uint8 else x.float()
        x = (x - self.mean) / self.std
        return self.encoder(x).reshape(b, t, -1)

    def classify_features(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        seq, (h_n, _) = self.lstm(self.feat_drop(feats))
        weights = None
        if self.pooling == "attention":
            pooled, weights = self.attn(seq)
        elif self.pooling == "mean":
            pooled = seq.mean(1)
        elif self.pooling == "last":
            pooled = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.lstm.bidirectional else h_n[-1]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return self.head(pooled.float()), weights

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.encode_frames(clip))[0]


def build_model(cfg: dict, num_classes: int, pretrained: bool | None = None) -> CNNLSTM:
    m = cfg["model"]
    return CNNLSTM(
        num_classes=num_classes,
        backbone=m["backbone"],
        pretrained=m["pretrained"] if pretrained is None else pretrained,
        trainable_backbone_stages=m["trainable_backbone_stages"],
        hidden_size=m["hidden_size"],
        num_layers=m["num_layers"],
        bidirectional=m["bidirectional"],
        dropout=m["dropout"],
        pooling=m["pooling"],
    )


def load_checkpoint(path: str, device: torch.device) -> tuple[CNNLSTM, dict, list[str]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg, classes = ckpt["config"], ckpt["classes"]
    model = build_model(cfg, len(classes), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, classes
