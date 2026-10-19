"""Extract L2-normalized embeddings from trained classifiers."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.models.load_model import load_keras_model
from src.utils.config_loader import load_config, resolve_path

_cache: dict[str, Any] = {}


def _embedding_model(full_model):
    import tensorflow as tf
    layer = full_model.get_layer("embedding")
    return tf.keras.Model(full_model.input, layer.output, name="embedding_extractor")


def get_embedding_model(architecture: str, cfg: dict | None = None):
    if architecture in _cache:
        return _cache[architecture]
    if cfg is None:
        cfg = load_config()
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    for name in (f"{architecture}_best.keras", f"{architecture}_final.keras"):
        path = models_dir / name
        if path.exists():
            full = load_keras_model(path)
            emb = _embedding_model(full)
            _cache[architecture] = emb
            return emb
    raise FileNotFoundError(f"No model weights for {architecture}")


def embedding_from_rgb_float(
    rgb_float: np.ndarray,
    architecture: str = "mobilenetv2",
    cfg: dict | None = None,
) -> np.ndarray:
    """rgb_float: H×W×3 float32 (0–255 scale as in training pipeline)."""
    if rgb_float.ndim == 3:
        batch = np.expand_dims(rgb_float, axis=0)
    else:
        batch = rgb_float
    model = get_embedding_model(architecture, cfg)
    vec = model.predict(batch, verbose=0)[0]
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm


def embedding_to_json(vec: np.ndarray) -> str:
    return json.dumps(vec.tolist())


def embedding_from_json(raw: str) -> np.ndarray:
    return np.array(json.loads(raw), dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
