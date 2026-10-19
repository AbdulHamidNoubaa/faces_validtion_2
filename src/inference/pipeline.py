"""Face verification inference pipeline (shared by Django web app and CLI)."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.data_preparation.low_light_simulation import apply_brightness_factor
from src.models.load_model import load_keras_model
from src.preprocessing.face_detection_alignment import detect_and_align_face
from src.preprocessing.image_enhancement import enhance_image
from src.utils.config_loader import load_config, resolve_path

_lock = threading.Lock()
_cache: dict[str, tuple[Any, dict, dict]] = {}


@dataclass
class VerificationResult:
    identity: str
    confidence: float
    accepted: bool
    decision_ar: str
    decision_en: str
    latency_ms: float
    architecture: str
    threshold: float
    brightness: float
    face_detected: bool
    preprocessed_rgb: np.ndarray | None = None
    match_source: str = "classifier"
    employee_code: str | None = None
    matched_photo_id: int | None = None


def get_available_architectures() -> list[str]:
    return ["mobilenetv2", "mobilenetv1", "resnet50", "vgg16"]


def _resolve_model_path(cfg: dict, architecture: str):
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    for name in (f"{architecture}_best.keras", f"{architecture}_final.keras"):
        path = models_dir / name
        if path.exists():
            return path
    return models_dir / f"{architecture}_best.keras"


def load_artifacts(architecture: str, cfg: dict | None = None):
    """Load and cache Keras model + label map for an architecture."""
    if cfg is None:
        cfg = load_config()
    arch = architecture.lower()
    if arch in _cache:
        return _cache[arch]

    with _lock:
        if arch in _cache:
            return _cache[arch]
        model_path = _resolve_model_path(cfg, arch)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained weights for '{arch}' at {model_path.parent}. "
                f"Run: python -m src.training.train --arch {arch}"
            )
        model = load_keras_model(model_path)
        label_path = resolve_path(cfg["paths"]["label_map"])
        with open(label_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        _cache[arch] = (model, label_map, cfg)
        return _cache[arch]


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Use JPG or PNG.")
    return img


def preprocess_frame(image_bgr: np.ndarray, cfg: dict) -> tuple[np.ndarray | None, bool]:
    aligned = detect_and_align_face(image_bgr, cfg)
    face_detected = aligned is not None
    if aligned is None:
        size = tuple(cfg["preprocessing"]["image_size"])
        aligned = cv2.resize(image_bgr, size)
    enhanced = enhance_image(aligned, cfg)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB).astype(np.float32)
    return rgb, face_detected


def _decide(proba: np.ndarray, label_map: dict, threshold: float) -> dict:
    idx = int(np.argmax(proba))
    conf = float(proba[idx])
    name = label_map["label_to_name"].get(str(idx), f"id_{idx}")
    accepted = conf >= threshold
    return {
        "identity": name,
        "confidence": conf,
        "accepted": accepted,
        "decision_ar": "مقبول ✓" if accepted else "مرفوض ✗",
        "decision_en": "ACCEPTED" if accepted else "REJECTED",
    }


def verify_image_bytes(
    file_bytes: bytes,
    architecture: str = "mobilenetv2",
    threshold: float | None = None,
    brightness: float = 1.0,
    cfg: dict | None = None,
) -> VerificationResult:
    """Full pipeline: decode → low-light sim → preprocess → predict."""
    if cfg is None:
        cfg = load_config()
    if threshold is None:
        threshold = float(cfg["demo"]["default_threshold"])

    t0 = time.perf_counter()
    image_bgr = decode_uploaded_image(file_bytes)
    image_bgr = apply_brightness_factor(image_bgr, brightness)

    model, label_map, cfg = load_artifacts(architecture, cfg)
    rgb, face_detected = preprocess_frame(image_bgr, cfg)
    if rgb is None:
        raise ValueError("Preprocessing failed.")

    batch = np.expand_dims(rgb, axis=0)
    proba = model.predict(batch, verbose=0)[0]
    decision = _decide(proba, label_map, threshold)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return VerificationResult(
        identity=decision["identity"],
        confidence=decision["confidence"],
        accepted=decision["accepted"],
        decision_ar=decision["decision_ar"],
        decision_en=decision["decision_en"],
        latency_ms=latency_ms,
        architecture=architecture.lower(),
        threshold=threshold,
        brightness=brightness,
        face_detected=face_detected,
        preprocessed_rgb=rgb,
    )


def rgb_to_jpeg_bytes(rgb: np.ndarray, quality: int = 85) -> bytes:
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()
