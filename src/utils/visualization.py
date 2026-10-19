"""Training curves, confusion matrix, and ROC plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


def plot_training_history(
    history: dict,
    save_path: str | Path,
    title: str = "Training History",
) -> None:
    """Plot loss/accuracy from a Keras history dict or CSV-loaded dict."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history.get("loss", [])) + 1)

    axes[0].plot(epochs, history.get("loss", []), label="train")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    acc_key = "accuracy" if "accuracy" in history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"
    if acc_key in history:
        axes[1].plot(epochs, history[acc_key], label="train")
    if val_acc_key in history:
        axes[1].plot(epochs, history[val_acc_key], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str] | None,
    save_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    labels = sorted(set(int(x) for x in y_true) | set(int(x) for x in y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    names = None
    if class_names is not None:
        names = [class_names[i] if i < len(class_names) else str(i) for i in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.4), max(6, len(labels) * 0.35)))
    sns.heatmap(
        cm,
        annot=len(labels) <= 20,
        fmt="d",
        cmap="Blues",
        xticklabels=names if names else labels,
        yticklabels=names if names else labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    if names and len(names) > 15:
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_ovr(
    y_true: Sequence[int],
    y_proba: np.ndarray,
    save_path: str | Path,
    title: str = "ROC (One-vs-Rest)",
) -> float:
    """Plot micro-average ROC; return micro-average AUC."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    classes = np.arange(y_proba.shape[1])
    y_bin = label_binarize(y_true, classes=classes)
    if y_bin.ndim == 1:
        y_bin = np.column_stack([1 - y_bin, y_bin])

    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"micro-average AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return float(roc_auc)
