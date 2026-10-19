"""Baseline architectures for academic comparison (MobileNetV1, ResNet50, VGG16)."""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet, ResNet50, VGG16

from src.models.mobilenetv2_model import build_mobilenetv2_classifier, compile_model
from src.models.preprocess_layer import KerasAppPreprocess


def _build_generic(
    name: str,
    base_factory,
    preprocess_backend: str,
    num_classes: int,
    cfg: dict[str, Any],
    training_phase: str = "head",
) -> tf.keras.Model:
    input_shape = tuple(cfg["model"]["input_shape"])
    dropout = float(cfg["model"].get("dropout", 0.3))
    dense_units = int(cfg["model"].get("dense_units", 256))

    base = base_factory(
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
    x = KerasAppPreprocess(backend=preprocess_backend, name="preprocess")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu", name="embedding")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs, outputs, name=f"{name}_face_cls")


def build_model(
    architecture: str,
    num_classes: int,
    cfg: dict[str, Any],
    training_phase: str = "head",
) -> tf.keras.Model:
    arch = architecture.lower()
    if arch == "mobilenetv2":
        return build_mobilenetv2_classifier(num_classes, cfg, training_phase)
    if arch in ("mobilenetv1", "mobilenet"):
        return _build_generic(
            "mobilenetv1",
            lambda **kw: MobileNet(**kw),
            "mobilenet",
            num_classes,
            cfg,
            training_phase,
        )
    if arch == "resnet50":
        return _build_generic(
            "resnet50",
            lambda **kw: ResNet50(**kw),
            "resnet50",
            num_classes,
            cfg,
            training_phase,
        )
    if arch == "vgg16":
        return _build_generic(
            "vgg16",
            lambda **kw: VGG16(**kw),
            "vgg16",
            num_classes,
            cfg,
            training_phase,
        )
    raise ValueError(f"Unsupported architecture: {architecture}")


__all__ = ["build_model", "compile_model"]
