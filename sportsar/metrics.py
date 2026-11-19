from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             f1_score, top_k_accuracy_score)


def classification_metrics(labels: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    preds = probs.argmax(1)
    k = len(classes)
    out = {
        "num_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=list(range(k)), zero_division=0)),
        "per_class": classification_report(labels, preds, labels=list(range(k)), target_names=classes,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(k))).tolist(),
    }
    if k > 5:  # top-3 is trivial with few classes
        out["top3_accuracy"] = float(top_k_accuracy_score(labels, probs, k=3, labels=list(range(k))))
    return out


def format_report(metrics: dict, classes: list[str]) -> str:
    lines = [f"Accuracy : {metrics['accuracy'] * 100:.2f}%  ({metrics['num_samples']} clips)",
             f"Macro-F1 : {metrics['macro_f1'] * 100:.2f}%"]
    if "top3_accuracy" in metrics:
        lines.append(f"Top-3 acc: {metrics['top3_accuracy'] * 100:.2f}%")
    lines.append(f"\n{'class':<20}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for c in classes:
        r = metrics["per_class"][c]
        lines.append(f"{c:<20}{r['precision']:>10.3f}{r['recall']:>10.3f}{r['f1-score']:>10.3f}{int(r['support']):>10d}")
    return "\n".join(lines)


def plot_confusion_matrix(cm, classes: list[str], path: str | Path, title: str = "Confusion matrix") -> None:
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    size = max(5, 0.6 * len(classes) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{norm[i, j] * 100:.0f}%\n({int(cm[i, j])})", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: list[dict], path: str | Path) -> None:
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train")
    axes[0].plot(ep, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(ep, [h["train_acc"] * 100 for h in history], label="train")
    axes[1].plot(ep, [h["val_acc"] * 100 for h in history], label="val")
    axes[1].set_title("Accuracy (%)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
