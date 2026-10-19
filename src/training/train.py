"""Train closed-set face classifier with transfer learning + fine-tuning."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.models.baseline_models import build_model, compile_model
from src.training.callbacks import build_callbacks
from src.utils.config_loader import ensure_dirs, load_config, project_root, resolve_path
from src.utils.logger import get_logger
from src.utils.visualization import plot_training_history

logger = get_logger("train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _load_split_arrays(cfg: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    splits_path = resolve_path(cfg["paths"]["splits_file"])
    with open(splits_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["splits"][split]
    if not items:
        raise RuntimeError(f"Split '{split}' is empty. Run build_dataset_splits first.")

    size = tuple(cfg["preprocessing"]["image_size"])
    images = []
    labels = []
    for item in items:
        path = resolve_path(item["path"])
        # images already enhanced BGR on disk; convert to RGB float
        img = tf.io.read_file(str(path))
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, size)
        images.append(img.numpy().astype(np.float32))
        labels.append(int(item["label"]))

    x = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int32)
    return x, y


def _make_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    cfg: dict,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 1024), seed=int(cfg["project"]["seed"]))

    if augment and cfg["preprocessing"]["augmentation"].get("enabled", True):
        from src.preprocessing.augmentation import make_keras_augmentation

        aug = make_keras_augmentation(cfg)

        def _aug(image, label):
            image = aug(image, training=True)
            return image, label

        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def train(cfg: dict | None = None, architecture: str | None = None) -> Path:
    if cfg is None:
        cfg = load_config()
    ensure_dirs(cfg)
    set_seed(int(cfg["project"]["seed"]))

    arch = (architecture or cfg["model"]["architecture"]).lower()
    logger.info("Training architecture=%s", arch)

    splits_path = resolve_path(cfg["paths"]["splits_file"])
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Missing {splits_path}. Run: python -m src.data_preparation.build_dataset_splits"
        )

    with open(splits_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    num_classes = int(meta["num_classes"])

    logger.info("Loading train/val arrays...")
    x_train, y_train = _load_split_arrays(cfg, "train")
    x_val, y_val = _load_split_arrays(cfg, "val")
    logger.info(
        "train=%s val=%s classes=%s",
        len(y_train),
        len(y_val),
        num_classes,
    )

    bs = int(cfg["training"]["batch_size"])
    train_ds = _make_dataset(x_train, y_train, bs, shuffle=True, augment=True, cfg=cfg)
    val_ds = _make_dataset(x_val, y_val, bs, shuffle=False, augment=False, cfg=cfg)

    # Phase 1: frozen base + head
    model = build_model(arch, num_classes, cfg, training_phase="head")
    model = compile_model(
        model,
        learning_rate=float(cfg["training"]["learning_rate"]),
        optimizer_name=str(cfg["training"]["optimizer"]),
    )
    callbacks = build_callbacks(cfg, model_name=arch)

    logger.info("Phase 1: training classification head...")
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg["training"]["epochs_head"]),
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2: fine-tune
    history = {k: list(v) for k, v in hist1.history.items()}
    if cfg["model"]["fine_tune"].get("enabled", True):
        logger.info("Phase 2: fine-tuning last layers...")
        # Rebuild with same weights but unfrozen layers — transfer weights
        ft_model = build_model(arch, num_classes, cfg, training_phase="finetune")
        ft_model.set_weights(model.get_weights())
        ft_model = compile_model(
            ft_model,
            learning_rate=float(cfg["model"]["fine_tune"]["fine_tune_learning_rate"]),
            optimizer_name=str(cfg["training"]["optimizer"]),
        )
        callbacks_ft = build_callbacks(cfg, model_name=arch)
        hist2 = ft_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=int(cfg["training"]["epochs_finetune"]),
            callbacks=callbacks_ft,
            verbose=1,
        )
        for k, v in hist2.history.items():
            history.setdefault(k, [])
            history[k].extend(list(v))
        model = ft_model

    # Save final + history
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    final_path = models_dir / f"{arch}_final.keras"
    model.save(final_path)

    hist_json = resolve_path(cfg["paths"]["logs_dir"]) / f"{arch}_history.json"
    with open(hist_json, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_training_history(
        history,
        resolve_path(cfg["paths"]["reports_dir"]) / f"{arch}_training_curves.png",
        title=f"{arch} Training History",
    )

    # Also persist num_classes sidecar
    meta_out = {
        "architecture": arch,
        "num_classes": num_classes,
        "final_model": str(final_path.relative_to(project_root())).replace("\\", "/"),
        "best_checkpoint": str((models_dir / f"{arch}_best.keras").relative_to(project_root())).replace("\\", "/"),
    }
    with open(models_dir / f"{arch}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    logger.info("Saved model -> %s", final_path)
    return final_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train face recognition model")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--arch", default=None, help="Override architecture")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, architecture=args.arch)


if __name__ == "__main__":
    main()
