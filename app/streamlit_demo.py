"""DEPRECATED: use Django web app (`python web/manage.py runserver`).

Legacy Streamlit banking checkpoint demo (upload / camera / low-light slider).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Streamlit runs this file without project root on PYTHONPATH
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from src.data_preparation.low_light_simulation import apply_brightness_factor
from src.preprocessing.face_detection_alignment import detect_and_align_face
from src.preprocessing.image_enhancement import enhance_image
from src.utils.config_loader import load_config, project_root, resolve_path


@st.cache_resource
def load_artifacts(arch: str):
    cfg = load_config()
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    model_path = models_dir / f"{arch}_best.keras"
    if not model_path.exists():
        model_path = models_dir / f"{arch}_final.keras"
    if not model_path.exists():
        return None, None, cfg, model_path

    from src.models.load_model import load_keras_model

    model = load_keras_model(model_path)
    with open(resolve_path(cfg["paths"]["label_map"]), "r", encoding="utf-8") as f:
        label_map = json.load(f)
    return model, label_map, cfg, model_path


def preprocess_frame(image_bgr: np.ndarray, cfg: dict) -> np.ndarray | None:
    aligned = detect_and_align_face(image_bgr, cfg)
    if aligned is None:
        size = tuple(cfg["preprocessing"]["image_size"])
        aligned = cv2.resize(image_bgr, size)
    enhanced = enhance_image(aligned, cfg)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB).astype(np.float32)
    return rgb


def decide(proba: np.ndarray, label_map: dict, threshold: float) -> dict:
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


def main() -> None:
    st.set_page_config(page_title="Banking Face Checkpoint", layout="wide")
    cfg = load_config()
    st.title(cfg["demo"]["title_ar"])
    st.caption(cfg["demo"]["title_en"])

    arch = st.sidebar.selectbox(
        "Model / النموذج",
        ["mobilenetv2", "mobilenetv1", "resnet50", "vgg16"],
        index=0,
    )
    threshold = st.sidebar.slider(
        "Confidence threshold / عتبة الثقة",
        0.1,
        0.99,
        float(cfg["demo"]["default_threshold"]),
        0.01,
    )
    brightness = st.sidebar.slider(
        "Simulate low light / محاكاة إضاءة منخفضة",
        0.1,
        1.0,
        1.0,
        0.05,
        help="1.0 = original brightness; lower values darken the input for stress-testing.",
    )

    model, label_map, cfg, model_path = load_artifacts(arch)
    if model is None:
        st.error(
            f"Model not found: `{model_path}`.\n\n"
            "Train first:\n"
            "```\npython -m src.training.train --arch mobilenetv2\n```"
        )
        st.stop()

    st.sidebar.success(f"Loaded: {model_path.name}")

    tab1, tab2 = st.tabs(["Upload image / رفع صورة", "Camera / كاميرا"])

    image_bgr = None
    with tab1:
        up = st.file_uploader("Face image", type=["jpg", "jpeg", "png"])
        if up is not None:
            file_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with tab2:
        cam = st.camera_input("Capture face")
        if cam is not None:
            file_bytes = np.asarray(bytearray(cam.getvalue()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.info("ارفع صورة أو التقط من الكاميرا لبدء التحقق.")
        return

    darkened = apply_brightness_factor(image_bgr, brightness)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Input")
        st.image(cv2.cvtColor(darkened, cv2.COLOR_BGR2RGB), use_container_width=True)

    t0 = time.perf_counter()
    face = preprocess_frame(darkened, cfg)
    if face is None:
        st.error("Face not detected / لم يُكتشف وجه.")
        return

    with col2:
        st.subheader("Preprocessed")
        st.image(face.astype(np.uint8), use_container_width=True)

    batch = np.expand_dims(face, axis=0)
    proba = model.predict(batch, verbose=0)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    result = decide(proba, label_map, threshold)

    with col3:
        st.subheader("Decision / القرار")
        if result["accepted"]:
            st.success(result["decision_ar"])
        else:
            st.error(result["decision_ar"])
        st.metric("Identity / الهوية", result["identity"])
        st.metric("Confidence / الثقة", f"{result['confidence']*100:.2f}%")
        st.metric("Latency / زمن المعالجة", f"{elapsed_ms:.1f} ms")

    st.progress(min(result["confidence"], 1.0))
    st.caption(
        "سيناريو محاكاة لنقطة تحقق مصرفية (صراف آلي / بوابة دخول). "
        "استخدم شريط الإضاءة لاختبار صلابة النظام تحت الإضاءة المنخفضة."
    )


if __name__ == "__main__":
    main()
