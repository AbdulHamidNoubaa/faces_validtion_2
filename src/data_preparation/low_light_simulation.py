"""Artificial low-light simulation for academic experiments."""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np


def simulate_low_light(
    image_bgr: np.ndarray,
    brightness_range: Sequence[float] = (0.15, 0.45),
    noise_std: float = 8.0,
    gamma_dark: Sequence[float] = (1.8, 2.8),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Darken an image to simulate low-light CCTV / ATM conditions.

    Steps: multiplicative brightness scale -> gamma darkening -> additive noise.
    Returns uint8 BGR image of the same shape.
    """
    if rng is None:
        rng = np.random.default_rng()

    img = image_bgr.astype(np.float32)
    scale = float(rng.uniform(brightness_range[0], brightness_range[1]))
    img = img * scale

    gamma = float(rng.uniform(gamma_dark[0], gamma_dark[1]))
    img = 255.0 * np.power(np.clip(img / 255.0, 0.0, 1.0), gamma)

    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, img.shape).astype(np.float32)
        img = img + noise

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_brightness_factor(image_bgr: np.ndarray, factor: float) -> np.ndarray:
    """Deterministic brightness scaling for demo UI (factor in [0, 1])."""
    factor = float(np.clip(factor, 0.05, 1.0))
    out = image_bgr.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def estimate_brightness(image_bgr: np.ndarray) -> float:
    """Mean luminance in [0, 255]."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))
