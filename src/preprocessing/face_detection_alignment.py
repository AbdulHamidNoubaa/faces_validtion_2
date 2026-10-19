"""Face detection and alignment (MediaPipe Tasks + OpenCV Haar fallback)."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.config_loader import project_root

_mp_detector = None
_haar = None

_MP_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_HAAR_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)


def _ensure_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest


def _mp_model_path() -> Path:
    return _ensure_file(
        _MP_MODEL_URL,
        project_root() / "outputs" / "models" / "blaze_face_short_range.tflite",
    )


def _haar_path() -> Path:
    # OpenCV 4 ships cascades; OpenCV 5 may not — fall back to download
    bundled = Path(getattr(cv2, "data").haarcascades) / "haarcascade_frontalface_default.xml"
    if bundled.exists():
        return bundled
    return _ensure_file(
        _HAAR_URL,
        project_root() / "outputs" / "models" / "haarcascade_frontalface_default.xml",
    )


def _get_mediapipe_tasks(min_conf: float = 0.5):
    global _mp_detector
    if _mp_detector is not None:
        return _mp_detector

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base = mp_python.BaseOptions(model_asset_path=str(_mp_model_path()))
    options = vision.FaceDetectorOptions(
        base_options=base,
        min_detection_confidence=float(min_conf),
    )
    _mp_detector = vision.FaceDetector.create_from_options(options)
    return _mp_detector


def _get_haar():
    global _haar
    if _haar is None:
        _haar = cv2.CascadeClassifier(str(_haar_path()))
        if _haar.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {_haar_path()}")
    return _haar


def _clamp_box(x: int, y: int, w: int, h: int, W: int, H: int, margin: float = 0.2):
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + margin)
    x1 = int(max(0, cx - side / 2))
    y1 = int(max(0, cy - side / 2))
    x2 = int(min(W, cx + side / 2))
    y2 = int(min(H, cy + side / 2))
    return x1, y1, x2, y2


def detect_face_mediapipe(
    image_bgr: np.ndarray,
    min_conf: float = 0.5,
) -> tuple[int, int, int, int] | None:
    import mediapipe as mp

    H, W = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if hasattr(mp, "solutions"):
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=float(min_conf),
        )
        results = detector.process(rgb)
        if not results.detections:
            return None
        bb = results.detections[0].location_data.relative_bounding_box
        x = int(bb.xmin * W)
        y = int(bb.ymin * H)
        w = int(bb.width * W)
        h = int(bb.height * H)
        return _clamp_box(x, y, w, h, W, H)

    detector = _get_mediapipe_tasks(min_conf)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = detector.detect(mp_img)
    if not result.detections:
        return None
    box = result.detections[0].bounding_box
    return _clamp_box(int(box.origin_x), int(box.origin_y), int(box.width), int(box.height), W, H)


def detect_face_haar(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = _get_haar().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    H, W = image_bgr.shape[:2]
    return _clamp_box(int(x), int(y), int(w), int(h), W, H)


def detect_and_align_face(image_bgr: np.ndarray, cfg: dict[str, Any]) -> np.ndarray | None:
    """
    Detect face, crop with margin, resize to config image_size.
    Returns BGR uint8 or None if no face found.
    """
    size = tuple(cfg["preprocessing"]["image_size"])
    detector = str(cfg["preprocessing"].get("face_detector", "mediapipe")).lower()
    min_conf = float(cfg["preprocessing"].get("mediapipe_min_confidence", 0.5))

    box = None
    if detector == "mediapipe":
        try:
            box = detect_face_mediapipe(image_bgr, min_conf)
        except Exception:
            box = None
        if box is None:
            try:
                box = detect_face_haar(image_bgr)
            except Exception:
                box = None
    else:
        try:
            box = detect_face_haar(image_bgr)
        except Exception:
            box = None
        if box is None:
            try:
                box = detect_face_mediapipe(image_bgr, min_conf)
            except Exception:
                box = None

    if box is None:
        return None

    x1, y1, x2, y2 = box
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
