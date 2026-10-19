"""Evaluate trained model: Accuracy/Precision/Recall/F1/AUC + plots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.config_loader import ensure_dirs, load_config, resolve_path
from src.utils.logger import get_logger
from src.utils.visualization import plot_confusion_matrix, plot_roc_ovr, plot_training_history

logger = get_logger("evaluate")


def _load_test(cfg: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with open(resolve_path(cfg["paths"]["splits_file"]), "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(resolve_path(cfg["paths"]["label_map"]), "r", encoding="utf-8") as f:
        labels = json.load(f)

    inv = {int(k): v for k, v in labels["label_to_name"].items()}
    class_names = [inv[i] for i in range(len(inv))]

    size = tuple(cfg["preprocessing"]["image_size"])
    xs, ys = [], []
    for item in data["splits"]["test"]:
        path = resolve_path(item["path"])
        img = tf.io.read_file(str(path))
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, size)
        xs.append(img.numpy().astype(np.float32))
        ys.append(int(item["label"]))
    return np.stack(xs), np.array(ys, dtype=np.int32), class_names


def _resolve_model_path(cfg: dict, architecture: str) -> Path:
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    best = models_dir / f"{architecture}_best.keras"
    final = models_dir / f"{architecture}_final.keras"
    if best.exists():
        return best
    if final.exists():
        return final
    raise FileNotFoundError(
        f"No trained model for {architecture}. Expected {best} or {final}."
    )


def evaluate(cfg: dict | None = None, architecture: str | None = None) -> dict:
    if cfg is None:
        cfg = load_config()
    ensure_dirs(cfg)
    arch = (architecture or cfg["model"]["architecture"]).lower()

    from src.models.load_model import load_keras_model

    model_path = _resolve_model_path(cfg, arch)
    logger.info("Loading model %s", model_path)
    model = load_keras_model(model_path)

    x_test, y_test, class_names = _load_test(cfg)
    logger.info("Test samples: %s", len(y_test))

    proba = model.predict(x_test, batch_size=int(cfg["training"]["batch_size"]), verbose=0)
    y_pred = np.argmax(proba, axis=1)

    metrics = {
        "architecture": arch,
        "model_path": str(model_path),
        "num_test_samples": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }

    try:
        metrics["auc_ovr_macro"] = float(
            roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        )
    except ValueError as exc:
        metrics["auc_ovr_macro"] = None
        metrics["auc_note"] = (
            f"AUC unavailable ({exc}). Often happens if a class is missing in the test set."
        )

    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names,
        reports_dir / f"{arch}_confusion_matrix.png",
        title=f"{arch} Confusion Matrix",
    )
    micro_auc = plot_roc_ovr(
        y_test,
        proba,
        reports_dir / f"{arch}_roc.png",
        title=f"{arch} ROC (micro)",
    )
    metrics["auc_micro"] = micro_auc

    # Training curves if history exists
    hist_json = resolve_path(cfg["paths"]["logs_dir"]) / f"{arch}_history.json"
    if hist_json.exists():
        with open(hist_json, "r", encoding="utf-8") as f:
            history = json.load(f)
        plot_training_history(
            history,
            reports_dir / f"{arch}_training_curves.png",
            title=f"{arch} Training History",
        )

    # Classification report
    report_txt = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )
    with open(reports_dir / f"{arch}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    metrics_path = reports_dir / f"{arch}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Markdown snippet
    md_path = reports_dir / "evaluation_report.md"
    _append_or_write_report(md_path, metrics, report_txt)

    logger.info(
        "Results %s | Acc=%.4f F1=%.4f AUC_micro=%.4f",
        arch,
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics["auc_micro"],
    )
    return metrics


def _append_or_write_report(md_path: Path, metrics: dict, report_txt: str) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    block = f"""
## Model: `{metrics['architecture']}`

| Metric | Value |
|--------|------:|
| Test samples | {metrics['num_test_samples']} |
| Accuracy | {metrics['accuracy']:.4f} |
| Precision (macro) | {metrics['precision_macro']:.4f} |
| Recall (macro) | {metrics['recall_macro']:.4f} |
| F1 (macro) | {metrics['f1_macro']:.4f} |
| AUC (micro) | {metrics.get('auc_micro')} |
| AUC (macro OVR) | {metrics.get('auc_ovr_macro')} |

Model file: `{metrics['model_path']}`

### Classification report

```
{report_txt}
```

---
"""
    header = """# Evaluation Report / تقرير التقييم

> **Honesty note:** Metrics below are produced only after a real training+evaluation run
> on prepared data. If this section is empty or marked pending, no fabricated numbers
> were inserted.

"""
    if not md_path.exists():
        md_path.write_text(header + block, encoding="utf-8")
    else:
        existing = md_path.read_text(encoding="utf-8")
        # Replace section for same architecture if present
        marker = f"## Model: `{metrics['architecture']}`"
        if marker in existing:
            # crude replace: keep header, drop old section
            parts = existing.split("## Model:")
            kept = [parts[0]]
            for p in parts[1:]:
                if not p.lstrip().startswith(f"`{metrics['architecture']}`"):
                    kept.append("## Model:" + p)
            existing = "".join(kept)
        md_path.write_text(existing.rstrip() + "\n" + block, encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--arch", default=None)
    args = parser.parse_args()
    evaluate(load_config(args.config), architecture=args.arch)


if __name__ == "__main__":
    main()
