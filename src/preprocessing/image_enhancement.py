"""Low-light image enhancement: CLAHE+Gamma (default) and Retinex (stretch)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def apply_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_gamma(image_bgr: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """gamma > 1 brightens mid-tones after CLAHE for dark faces."""
    gamma = float(max(gamma, 1e-6))
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image_bgr, table)


def apply_clahe_gamma(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    gamma: float = 1.2,
) -> np.ndarray:
    return apply_gamma(apply_clahe(image_bgr, clip_limit, tile_grid_size), gamma)


def apply_single_scale_retinex(image_bgr: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    """
    Simplified Single-Scale Retinex (stretch goal).
    Operates in log domain; output rescaled to uint8.
    """
    img = image_bgr.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    blur = np.maximum(blur, 1.0)
    retinex = np.log(img) - np.log(blur)
    # Per-channel rescale to [0, 255]
    out = np.zeros_like(image_bgr, dtype=np.uint8)
    for c in range(3):
        ch = retinex[:, :, c]
        ch = ch - ch.min()
        denom = ch.max() if ch.max() > 0 else 1.0
        out[:, :, c] = (ch / denom * 255).astype(np.uint8)
    return out


def enhance_image(image_bgr: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Dispatch enhancement based on config.preprocessing.enhancement_method."""
    method = str(cfg["preprocessing"].get("enhancement_method", "clahe_gamma")).lower()
    if method in ("none", "off", "disabled"):
        return image_bgr
    if method == "clahe_gamma":
        c = cfg["preprocessing"]["clahe"]
        g = cfg["preprocessing"]["gamma"]
        return apply_clahe_gamma(
            image_bgr,
            clip_limit=float(c["clip_limit"]),
            tile_grid_size=tuple(c["tile_grid_size"]),
            gamma=float(g["value"]),
        )
    if method == "retinex":
        sigma = float(cfg["preprocessing"]["retinex"]["sigma"])
        return apply_single_scale_retinex(image_bgr, sigma=sigma)
    raise ValueError(f"Unknown enhancement_method: {method}")
