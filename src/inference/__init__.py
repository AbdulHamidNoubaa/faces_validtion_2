"""Inference pipeline for web and API consumers."""

from src.inference.pipeline import (
    VerificationResult,
    decode_uploaded_image,
    get_available_architectures,
    load_artifacts,
    preprocess_frame,
    rgb_to_jpeg_bytes,
    verify_image_bytes,
)

__all__ = [
    "VerificationResult",
    "decode_uploaded_image",
    "get_available_architectures",
    "load_artifacts",
    "preprocess_frame",
    "rgb_to_jpeg_bytes",
    "verify_image_bytes",
]
