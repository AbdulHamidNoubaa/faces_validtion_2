"""Unified face preprocessing (same steps as LFW dataset pipeline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.data_preparation.low_light_simulation import simulate_low_light
from src.preprocessing.face_detection_alignment import detect_and_align_face
from src.preprocessing.image_enhancement import enhance_image


def process_face_image_bgr(
    image_bgr: np.ndarray,
    cfg: dict[str, Any],
    apply_low_light: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Detect → optional low-light sim → align → CLAHE/Gamma → BGR uint8 224×224.

    Returns (processed_bgr, face_detected).
    """
    if rng is None:
        rng = np.random.default_rng()

    img = image_bgr.copy()
    if apply_low_light:
        ll = cfg["dataset"]["low_light"]
        img = simulate_low_light(
            img,
            brightness_range=ll["brightness_range"],
            noise_std=float(ll["noise_std"]),
            gamma_dark=ll["gamma_dark"],
            rng=rng,
        )

    aligned = detect_and_align_face(img, cfg)
    face_detected = aligned is not None
    if aligned is None:
        size = tuple(cfg["preprocessing"]["image_size"])
        aligned = cv2.resize(img, size)

    enhanced = enhance_image(aligned, cfg)
    return enhanced, face_detected


def process_image_file(
    src: Path,
    dest: Path,
    cfg: dict[str, Any],
    apply_low_light: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[bool, bool]:
    """Read src image, write processed BGR to dest. Returns (ok, face_detected)."""
    img = cv2.imread(str(src))
    if img is None:
        return False, False
    out, face_detected = process_face_image_bgr(img, cfg, apply_low_light, rng)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dest), out)
    return bool(ok), face_detected
