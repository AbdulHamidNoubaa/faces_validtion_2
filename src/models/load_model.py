"""Load saved Keras models with project custom layers registered."""

from __future__ import annotations

from pathlib import Path


def load_keras_model(path: str | Path):
    """Lazy-loads TensorFlow only when a model is actually needed."""
    import tensorflow as tf
    from src.models.preprocess_layer import KerasAppPreprocess

    kwargs = {
        "filepath": str(path),
        "custom_objects": {"KerasAppPreprocess": KerasAppPreprocess},
        "compile": False,
    }
    try:
        return tf.keras.models.load_model(**kwargs, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(**kwargs)
