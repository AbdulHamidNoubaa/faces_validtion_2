"""Reusable Keras preprocess layer (serializable)."""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="bfr")
class KerasAppPreprocess(tf.keras.layers.Layer):
    """Apply a keras.applications preprocess_input inside the graph."""

    def __init__(self, backend: str = "mobilenet_v2", **kwargs):
        super().__init__(**kwargs)
        self.backend = backend

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        b = self.backend
        if b == "mobilenet_v2":
            return tf.keras.applications.mobilenet_v2.preprocess_input(x)
        if b == "mobilenet":
            return tf.keras.applications.mobilenet.preprocess_input(x)
        if b == "resnet50":
            return tf.keras.applications.resnet50.preprocess_input(x)
        if b == "vgg16":
            return tf.keras.applications.vgg16.preprocess_input(x)
        raise ValueError(f"Unknown preprocess backend: {b}")

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"backend": self.backend})
        return cfg
