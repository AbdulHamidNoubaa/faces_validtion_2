"""Keras callbacks for training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf


def build_callbacks(cfg: dict[str, Any], model_name: str = "mobilenetv2") -> list:
    models_dir = Path(cfg["paths"]["models_dir"])
    # resolved by caller typically; keep relative-safe
    from src.utils.config_loader import resolve_path

    models_dir = resolve_path(cfg["paths"]["models_dir"])
    logs_dir = resolve_path(cfg["paths"]["logs_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = models_dir / f"{model_name}_best.keras"
    csv_path = logs_dir / f"{model_name}_history.csv"

    train_cfg = cfg["training"]
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(train_cfg["early_stopping_patience"]),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(train_cfg["reduce_lr_factor"]),
            patience=int(train_cfg["reduce_lr_patience"]),
            min_lr=float(train_cfg["min_lr"]),
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_path), append=True),
    ]
    return callbacks
