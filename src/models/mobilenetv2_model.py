"""MobileNetV2 backbone with transfer-learning classification head."""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from src.models.preprocess_layer import KerasAppPreprocess


def build_mobilenetv2_classifier(
    num_classes: int,
    cfg: dict[str, Any],
    training_phase: str = "head",
) -> tf.keras.Model:
    """
    Build MobileNetV2 + GAP + Dense head for closed-set classification.

    training_phase:
      - 'head': freeze entire base
      - 'finetune': unfreeze last N layers
    """
    input_shape = tuple(cfg["model"]["input_shape"])
    dropout = float(cfg["model"].get("dropout", 0.3))
    dense_units = int(cfg["model"].get("dense_units", 256))

    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base.trainable = False

    if training_phase == "finetune" and cfg["model"]["fine_tune"].get("enabled", True):
        n = int(cfg["model"]["fine_tune"]["unfreeze_last_n_layers"])
        base.trainable = True
        for layer in base.layers[:-n]:
            layer.trainable = False

    inputs = layers.Input(shape=input_shape, name="image")
    x = KerasAppPreprocess(backend="mobilenet_v2", name="preprocess")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu", name="embedding")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2_face_cls")
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    optimizer_name: str = "adam",
) -> tf.keras.Model:
    opt_name = optimizer_name.lower()
    if opt_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
