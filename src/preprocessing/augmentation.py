"""Training-time data augmentation utilities."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import tensorflow as tf


def augment_bgr(
    image_bgr: np.ndarray,
    cfg: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Lightweight OpenCV augmentation for numpy pipelines / demos."""
    if rng is None:
        rng = np.random.default_rng()
    aug = cfg["preprocessing"]["augmentation"]
    if not aug.get("enabled", True):
        return image_bgr

    img = image_bgr.copy()
    rot = float(aug.get("rotation_range", 10))
    if rot > 0:
        angle = float(rng.uniform(-rot, rot))
        H, W = img.shape[:2]
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)

    if aug.get("horizontal_flip", True) and rng.random() < 0.5:
        img = cv2.flip(img, 1)

    delta = float(aug.get("brightness_delta", 0.15))
    if delta > 0:
        factor = float(rng.uniform(1.0 - delta, 1.0 + delta))
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    noise_std = float(aug.get("noise_std", 5.0))
    if noise_std > 0:
        noise = rng.normal(0, noise_std, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def make_keras_augmentation(cfg: dict[str, Any]) -> tf.keras.Sequential:
    """Keras preprocessing layers used inside tf.data map (RGB inputs)."""
    aug = cfg["preprocessing"]["augmentation"]
    layers = []
    if aug.get("horizontal_flip", True):
        layers.append(tf.keras.layers.RandomFlip("horizontal"))
    rot = float(aug.get("rotation_range", 10)) / 360.0
    if rot > 0:
        layers.append(tf.keras.layers.RandomRotation(rot))
    delta = float(aug.get("brightness_delta", 0.15))
    if delta > 0:
        # RandomBrightness exists in TF>=2.9; fall back to contrast jitter otherwise
        if hasattr(tf.keras.layers, "RandomBrightness"):
            layers.append(tf.keras.layers.RandomBrightness(delta))
        else:
            layers.append(tf.keras.layers.RandomContrast(delta))
    return tf.keras.Sequential(layers, name="augmentation")
