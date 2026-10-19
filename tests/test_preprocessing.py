"""Unit tests for preprocessing / low-light modules (no trained weights required)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preparation.low_light_simulation import (  # noqa: E402
    apply_brightness_factor,
    simulate_low_light,
)
from src.preprocessing.image_enhancement import (  # noqa: E402
    apply_clahe_gamma,
    apply_single_scale_retinex,
    enhance_image,
)
from src.utils.config_loader import load_config  # noqa: E402


@pytest.fixture
def sample_bgr() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(160, 160, 3), dtype=np.uint8)


def test_simulate_low_light_darkens(sample_bgr):
    rng = np.random.default_rng(0)
    dark = simulate_low_light(sample_bgr, brightness_range=(0.2, 0.2), noise_std=0.0, rng=rng)
    assert dark.shape == sample_bgr.shape
    assert dark.mean() < sample_bgr.mean()


def test_brightness_factor(sample_bgr):
    out = apply_brightness_factor(sample_bgr, 0.3)
    assert out.dtype == np.uint8
    assert out.mean() < sample_bgr.mean()


def test_clahe_gamma_preserves_shape(sample_bgr):
    out = apply_clahe_gamma(sample_bgr)
    assert out.shape == sample_bgr.shape
    assert out.dtype == np.uint8


def test_retinex_shape(sample_bgr):
    out = apply_single_scale_retinex(sample_bgr, sigma=15.0)
    assert out.shape == sample_bgr.shape


def test_enhance_dispatch_from_config(sample_bgr):
    cfg = load_config()
    cfg["preprocessing"]["enhancement_method"] = "clahe_gamma"
    out = enhance_image(sample_bgr, cfg)
    assert out.shape == sample_bgr.shape

    cfg["preprocessing"]["enhancement_method"] = "retinex"
    out2 = enhance_image(sample_bgr, cfg)
    assert out2.shape == sample_bgr.shape


def test_config_loads():
    cfg = load_config()
    assert "model" in cfg
    assert cfg["model"]["architecture"] == "mobilenetv2"
