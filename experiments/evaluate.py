"""Cosine-gallery verification, masks, threshold sweep, and latency."""

from __future__ import annotations

import os
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from experiments.common import (
    CACHE,
    RESULTS,
    ROOT,
    SEED,
    THRESHOLD,
    TIMING_N,
    mean_std,
    safe_div,
    thresholds,
    write_csv,
    write_json,
)
from src.inference.embeddings import _embedding_model, cosine_similarity
from src.models.load_model import load_keras_model
from src.preprocessing.image_enhancement import enhance_image
from src.utils.config_loader import resolve_path

MASK_BGR = (190, 170, 150)
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _tf():
    import tensorflow as tf

    return tf


def set_tf_seed(seed: int = SEED) -> None:
    tf = _tf()
    tf.random.set_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def device_info() -> dict[str, Any]:
    tf = _tf()
    gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    cpus = [d.name for d in tf.config.list_physical_devices("CPU")]
    return {
        "tensorflow": tf.__version__,
        "gpus": gpus,
        "cpus": cpus,
        "runtime": "GPU" if gpus else "CPU",
        "built_with_cuda": bool(tf.test.is_built_with_cuda()),
    }


def resolve_weights(cfg: dict) -> Path:
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    for name in ("mobilenetv2_best.keras", "mobilenetv2_final.keras"):
        path = models_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No MobileNetV2 weights in {models_dir}")


def bgr_batch_to_rgb(images: np.ndarray) -> np.ndarray:
    rgb = np.empty(images.shape, dtype=np.float32)
    for index, image in enumerate(images):
        rgb[index] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    return vectors / norms


def embed_many(model, images_bgr: np.ndarray) -> np.ndarray:
    rgb = bgr_batch_to_rgb(images_bgr)
    vectors = model.predict(rgb, batch_size=32, verbose=0)
    return l2_normalize(vectors)


def decision_metrics(
    scores: np.ndarray,
    pred_ids: list[str],
    true_ids: list[str],
    genuine: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    pred = np.asarray(pred_ids)
    truth = np.asarray(true_ids)
    genuine = np.asarray(genuine, dtype=bool)
    impostor = ~genuine
    correct_accept = genuine & (scores >= threshold) & (pred == truth)
    tp = int(np.sum(correct_accept))
    fp = int(np.sum(impostor & (scores >= threshold)))
    fn = int(np.sum(genuine & ~correct_accept))
    tn = int(np.sum(impostor & (scores < threshold)))
    n_genuine = int(np.sum(genuine))
    n_impostor = int(np.sum(impostor))
    if tp + fn != n_genuine or fp + tn != n_impostor:
        raise RuntimeError(f"Confusion counts do not add up: tp={tp} fn={fn} fp={fp} tn={tn}")
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall / (precision + recall))
    y = genuine.astype(np.int32)
    auc = None
    auc_note = None
    if len(np.unique(y)) < 2:
        auc_note = "AUC undefined because the subset has only one class."
    else:
        auc = float(roc_auc_score(y, scores))
    return {
        "threshold": float(threshold),
        "n": int(scores.shape[0]),
        "n_genuine": n_genuine,
        "n_impostor": n_impostor,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_misidentified": int(np.sum(genuine & (scores >= threshold) & (pred != truth))),
        "accuracy": safe_div(tp + tn, scores.shape[0]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": safe_div(fp, n_impostor),
        "frr": safe_div(fn, n_genuine),
        "frr_score_only": safe_div(int(np.sum(genuine & (scores < threshold))), n_genuine),
        "auc": auc,
        "auc_note": auc_note,
    }


def score_pack(
    embeddings: np.ndarray,
    meta: list[dict],
    gallery: np.ndarray,
    gallery_ids: list[str],
) -> dict[str, Any]:
    sims = embeddings @ gallery.T
    best = np.argmax(sims, axis=1)
    scores = sims[np.arange(len(meta)), best]
    return {
        "scores": scores.astype(np.float64),
        "pred_ids": [gallery_ids[int(index)] for index in best],
        "true_ids": [item["identity"] for item in meta],
        "genuine": np.asarray([not item["held_out"] for item in meta], dtype=bool),
        "paths": [item["path"] for item in meta],
        "low_light": [bool(item["low_light"]) for item in meta],
    }


def subset_pack(pack: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    idx = np.asarray(indices, dtype=np.int64)
    return {
        "scores": pack["scores"][idx],
        "pred_ids": [pack["pred_ids"][int(i)] for i in idx],
        "true_ids": [pack["true_ids"][int(i)] for i in idx],
        "genuine": pack["genuine"][idx],
        "paths": [pack["paths"][int(i)] for i in idx],
        "low_light": [pack["low_light"][int(i)] for i in idx],
    }


def roc_arrays(genuine: np.ndarray, scores: np.ndarray) -> dict[str, np.ndarray] | None:
    y = genuine.astype(np.int32)
    if len(np.unique(y)) < 2:
        return None
    fpr, tpr, thr = roc_curve(y, scores, drop_intermediate=False)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thr}


def eer_score_only(genuine: np.ndarray, scores: np.ndarray) -> dict[str, Any] | None:
    curves = roc_arrays(genuine, scores)
    if curves is None:
        return None
    fpr = curves["fpr"]
    fnr = 1.0 - curves["tpr"]
    diff = np.abs(fpr - fnr)
    index = int(np.argmin(diff))
    threshold = curves["thresholds"][index]
    return {
        "eer": float((fpr[index] + fnr[index]) / 2.0),
        "far": float(fpr[index]),
        "frr": float(fnr[index]),
        "threshold": None if not np.isfinite(threshold) else float(threshold),
        "definition": "FRR is the share of genuine probes whose max cosine is below the threshold. Identity errors are not included.",
    }


def eer_identity_aware(rows: list[dict]) -> dict[str, Any]:
    usable = [row for row in rows if row["far"] is not None and row["frr"] is not None]
    if not usable:
        return {"grid_min_abs_gap": None, "interpolated": None}
    best = min(usable, key=lambda row: abs(row["far"] - row["frr"]))
    interpolated = None
    for left, right in zip(usable, usable[1:]):
        d0 = left["far"] - left["frr"]
        d1 = right["far"] - right["frr"]
        if d0 == 0:
            interpolated = {
                "threshold": left["threshold"],
                "eer": left["far"],
                "method": "exact_grid_point",
            }
            break
        if d0 * d1 < 0:
            weight = abs(d0) / (abs(d0) + abs(d1))
            far = left["far"] + weight * (right["far"] - left["far"])
            frr = left["frr"] + weight * (right["frr"] - left["frr"])
            interpolated = {
                "threshold": left["threshold"] + weight * (right["threshold"] - left["threshold"]),
                "eer": (far + frr) / 2.0,
                "far": far,
                "frr": frr,
                "method": "linear_between_grid_points",
            }
            break
    return {
        "grid_min_abs_gap": {
            "threshold": best["threshold"],
            "far": best["far"],
            "frr": best["frr"],
            "abs_gap": abs(best["far"] - best["frr"]),
            "accuracy": best["accuracy"],
        },
        "interpolated": interpolated,
        "definition": "FAR and FRR use the thesis decision: FRR counts a genuine probe unless cosine >= threshold and the identity matches.",
    }


def load_landmarker():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except Exception as exc:
        return None, f"mediapipe landmarker import failed: {exc}"

    import socket

    socket.setdefaulttimeout(60)
    dest = ROOT / "experiments" / "assets" / "face_landmarker.task"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size < 1000:
        try:
            urllib.request.urlretrieve(LANDMARKER_URL, dest)
        except Exception as exc:
            return None, f"landmarker download failed: {exc}"
    try:
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(dest)),
            num_faces=1,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        return {"mp": mp, "landmarker": landmarker, "path": str(dest)}, None
    except Exception as exc:
        return None, f"landmarker create failed: {exc}"


def apply_mask(image_bgr: np.ndarray, pack) -> tuple[np.ndarray, str]:
    image = image_bgr.copy()
    height, width = image.shape[:2]
    if pack is not None:
        try:
            mp = pack["mp"]
            rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = pack["landmarker"].detect(mp_image)
            landmarks = result.face_landmarks[0] if result.face_landmarks else None
            if landmarks and len(landmarks) > 1:
                nose_y = float(landmarks[1].y) * height
                points = []
                for landmark in landmarks:
                    x = int(np.clip(landmark.x * width, 0, width - 1))
                    y = int(np.clip(landmark.y * height, 0, height - 1))
                    if y >= nose_y:
                        points.append([x, y])
                if len(points) >= 3:
                    hull = cv2.convexHull(np.asarray(points, dtype=np.int32))
                    cv2.fillConvexPoly(image, hull, MASK_BGR)
                    return image, "mediapipe_facemesh_convex_hull_at_or_below_nose_tip"
        except Exception:
            image = image_bgr.copy()
    y0 = (height * 2) // 3
    cv2.rectangle(image, (0, y0), (width - 1, height - 1), MASK_BGR, thickness=-1)
    return image, "lower_third_rectangle"


def mask_many(images: np.ndarray, pack) -> tuple[np.ndarray, list[str]]:
    masked = []
    methods = []
    for image in images:
        out, method = apply_mask(image, pack)
        masked.append(out)
        methods.append(method)
    return np.stack(masked, axis=0), methods


def align_rgb_float(image_bgr: np.ndarray, cfg: dict) -> tuple[float, float, float, np.ndarray]:
    from src.preprocessing.face_detection_alignment import detect_and_align_face

    t0 = time.perf_counter()
    aligned = detect_and_align_face(image_bgr, cfg)
    if aligned is None:
        aligned = cv2.resize(image_bgr, tuple(cfg["preprocessing"]["image_size"]))
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, aligned


def run_timed(frame: np.ndarray, emb_model, gallery: np.ndarray, cfg: dict, enhance: bool) -> dict[str, float]:
    from src.preprocessing.face_detection_alignment import detect_and_align_face

    t0 = time.perf_counter()
    aligned = detect_and_align_face(frame, cfg)
    if aligned is None:
        aligned = cv2.resize(frame, tuple(cfg["preprocessing"]["image_size"]))
    t1 = time.perf_counter()
    if enhance:
        processed = enhance_image(aligned, cfg)
    else:
        processed = aligned
    t2 = time.perf_counter()
    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB).astype(np.float32)
    t3 = time.perf_counter()
    vector = emb_model.predict(rgb[None, ...], verbose=0)[0]
    vector = np.asarray(vector, dtype=np.float32)
    vector = vector / (np.linalg.norm(vector) + 1e-8)
    t4 = time.perf_counter()
    scores = gallery @ vector
    int(np.argmax(scores))
    t5 = time.perf_counter()
    resize_ms = (t1 - t0) * 1000.0
    enhance_ms = (t2 - t1) * 1000.0
    normalize_ms = (t3 - t2) * 1000.0
    embed_ms = (t4 - t3) * 1000.0
    match_ms = (t5 - t4) * 1000.0
    return {
        "resize_ms": resize_ms,
        "enhance_ms": enhance_ms,
        "normalize_ms": normalize_ms,
        "resize_normalize_ms": resize_ms + normalize_ms,
        "embed_ms": embed_ms,
        "match_ms": match_ms,
        "total_ms": (t5 - t0) * 1000.0,
    }


def summarize_stage(rows: list[dict], key: str) -> dict[str, Any]:
    stats = mean_std([row[key] for row in rows])
    return {"stage": key, **stats, "mean_ms": stats["mean"], "std_ms": stats["std"]}


def softmax_on_arrays(model, images_bgr: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    rgb = bgr_batch_to_rgb(images_bgr)
    proba = model.predict(rgb, batch_size=32, verbose=0)
    pred = np.argmax(proba, axis=1)
    correct = int(np.sum(pred == labels))
    return {
        "n": int(len(labels)),
        "correct": correct,
        "accuracy": float(correct / len(labels)),
    }


def softmax_on_stored(model, meta: list[dict], image_size: tuple[int, int]) -> dict[str, Any]:
    tf = _tf()
    images = []
    labels = []
    for item in meta:
        raw = tf.io.read_file(str(ROOT / item["path"]))
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        images.append(image.numpy().astype(np.float32))
        labels.append(int(item["label"]))
    batch = np.stack(images, axis=0)
    y = np.asarray(labels, dtype=np.int32)
    proba = model.predict(batch, batch_size=32, verbose=0)
    pred = np.argmax(proba, axis=1)
    correct = int(np.sum(pred == y))
    return {"n": int(len(y)), "correct": correct, "accuracy": float(correct / len(y))}


def evaluate(prep: dict) -> dict[str, Any]:
    set_tf_seed(SEED)
    cfg = prep["cfg"]
    weights = resolve_weights(cfg)
    print(f"Loading {weights}", flush=True)
    full_model = load_keras_model(weights)
    emb_model = _embedding_model(full_model)
    devices = device_info()
    meta = prep["test_meta"]
    labels = np.asarray([item["label"] for item in meta], dtype=np.int32)

    print("Embedding gallery and test images...", flush=True)
    gallery = embed_many(emb_model, prep["gallery_enhanced"])
    gallery_ids = [item["identity"] for item in prep["gallery_meta"]]
    # Confirm the project cosine helper matches the matrix product.
    cosine_delta = abs(
        cosine_similarity(gallery[0], gallery[min(1, len(gallery) - 1)])
        - float(np.dot(gallery[0], gallery[min(1, len(gallery) - 1)]))
    )
    emb_aligned = embed_many(emb_model, prep["test_aligned"])
    emb_enhanced = embed_many(emb_model, prep["test_enhanced"])
    embedding_dim = int(gallery.shape[1])

    nl_idx = [i for i, item in enumerate(meta) if not item["low_light"]]
    ll_idx = [i for i, item in enumerate(meta) if item["low_light"]]
    packs = {
        "E1": subset_pack(score_pack(emb_aligned, meta, gallery, gallery_ids), nl_idx),
        "E2": subset_pack(score_pack(emb_aligned, meta, gallery, gallery_ids), ll_idx),
        "E3": subset_pack(score_pack(emb_enhanced, meta, gallery, gallery_ids), ll_idx),
        "E3b": subset_pack(score_pack(emb_enhanced, meta, gallery, gallery_ids), nl_idx),
    }
    descriptions = {
        "E1": "NL test images, face aligned and resized, no CLAHE+Gamma",
        "E2": "LL test images, face aligned and resized, no CLAHE+Gamma",
        "E3": "Same LL aligned crop as E2, then CLAHE+Gamma",
        "E3b": "Same NL aligned crop as E1, then CLAHE+Gamma",
    }
    experiments = {}
    roc_store = {}
    score_rows = []
    for name, pack in packs.items():
        metrics = decision_metrics(pack["scores"], pack["pred_ids"], pack["true_ids"], pack["genuine"], THRESHOLD)
        metrics["description"] = descriptions[name]
        metrics["eer_score_only"] = eer_score_only(pack["genuine"], pack["scores"])
        experiments[name] = metrics
        curves = roc_arrays(pack["genuine"], pack["scores"])
        if curves is not None:
            roc_store[f"{name}_fpr"] = curves["fpr"]
            roc_store[f"{name}_tpr"] = curves["tpr"]
            roc_store[f"{name}_thr"] = curves["thresholds"]
        for index in range(len(pack["paths"])):
            score_rows.append(
                {
                    "experiment": name,
                    "path": pack["paths"][index],
                    "true_identity": pack["true_ids"][index],
                    "pred_identity": pack["pred_ids"][index],
                    "low_light": pack["low_light"][index],
                    "genuine": bool(pack["genuine"][index]),
                    "score": float(pack["scores"][index]),
                    "threshold": THRESHOLD,
                }
            )

    sweep_rows = []
    sweep_by_exp = {}
    for name in ("E2", "E3", "E1", "E3b"):
        pack = packs[name]
        rows = []
        for threshold in thresholds():
            metrics = decision_metrics(
                pack["scores"], pack["pred_ids"], pack["true_ids"], pack["genuine"], threshold
            )
            row = {
                "experiment": name,
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "far": metrics["far"],
                "frr": metrics["frr"],
                "frr_score_only": metrics["frr_score_only"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
            rows.append(row)
            sweep_rows.append(row)
        operating = next(row for row in rows if abs(row["threshold"] - THRESHOLD) < 1e-9)
        if abs(operating["accuracy"] - experiments[name]["accuracy"]) > 1e-12:
            raise RuntimeError(f"Sweep at 0.52 does not match {name}")
        sweep_by_exp[name] = {
            "rows": rows,
            "eer_identity_aware": eer_identity_aware(rows),
            "eer_score_only": experiments[name]["eer_score_only"],
        }

    print("Masking test faces...", flush=True)
    landmarker, landmarker_error = load_landmarker()
    nl_images = prep["test_aligned"][nl_idx]
    ll_images = prep["test_aligned"][ll_idx]
    nl_masked, nl_methods = mask_many(nl_images, landmarker)
    ll_masked, ll_methods = mask_many(ll_images, landmarker)
    ll_masked_enh = np.stack([enhance_image(image, cfg) for image in ll_masked], axis=0)
    masked_packs = {
        "M_NL": score_pack(embed_many(emb_model, nl_masked), [meta[i] for i in nl_idx], gallery, gallery_ids),
        "M_LL": score_pack(embed_many(emb_model, ll_masked), [meta[i] for i in ll_idx], gallery, gallery_ids),
        "M_LL_ENH": score_pack(embed_many(emb_model, ll_masked_enh), [meta[i] for i in ll_idx], gallery, gallery_ids),
    }
    masked_desc = {
        "M_NL": "Masked NL, no CLAHE+Gamma",
        "M_LL": "Masked LL, no CLAHE+Gamma",
        "M_LL_ENH": "Masked LL, then CLAHE+Gamma",
    }
    masked = {}
    for name, pack in masked_packs.items():
        metrics = decision_metrics(pack["scores"], pack["pred_ids"], pack["true_ids"], pack["genuine"], THRESHOLD)
        metrics["description"] = masked_desc[name]
        masked[name] = metrics
        for index in range(len(pack["paths"])):
            score_rows.append(
                {
                    "experiment": name,
                    "path": pack["paths"][index],
                    "true_identity": pack["true_ids"][index],
                    "pred_identity": pack["pred_ids"][index],
                    "low_light": pack["low_light"][index],
                    "genuine": bool(pack["genuine"][index]),
                    "score": float(pack["scores"][index]),
                    "threshold": THRESHOLD,
                }
            )

    import random

    sample_rng = random.Random(SEED)
    sample_nl = sample_rng.sample(nl_idx, 2)
    sample_ll = sample_rng.sample(ll_idx, 2)
    sample_images = []
    sample_paths = []
    sample_methods = []
    for index in sample_nl + sample_ll:
        local = nl_idx.index(index) if index in nl_idx else None
        if local is not None:
            masked_image, method = nl_masked[local], nl_methods[local]
            original = prep["test_aligned"][index]
        else:
            local = ll_idx.index(index)
            masked_image, method = ll_masked[local], ll_methods[local]
            original = prep["test_aligned"][index]
        sample_images.append(np.stack([original, masked_image], axis=0))
        sample_paths.append(meta[index]["path"])
        sample_methods.append(method)
    np.savez_compressed(
        CACHE / "samples_masked.npz",
        images=np.stack(sample_images, axis=0),
        paths=np.array(sample_paths),
        methods=np.array(sample_methods),
    )
    method_counts = dict(Counter(nl_methods + ll_methods))

    print("Measuring latency...", flush=True)
    order = sorted(range(len(meta)), key=lambda i: meta[i]["path"])[:TIMING_N]
    frames = [prep["test_frames"][i] for i in order]
    for _ in range(10):
        run_timed(frames[0], emb_model, gallery, cfg, enhance=True)
    with_rows = []
    without_rows = []
    for number, frame in enumerate(frames, start=1):
        with_rows.append(run_timed(frame, emb_model, gallery, cfg, enhance=True))
        without_rows.append(run_timed(frame, emb_model, gallery, cfg, enhance=False))
        if number % 50 == 0:
            print(f"  timed {number}/{len(frames)}", flush=True)

    latency = {
        "n": len(frames),
        "warmup_runs_not_counted": 10,
        "warmup_path": "full pipeline with CLAHE+Gamma on the first sorted test frame",
        "selection": "first 200 test paths after alphabetical sort",
        "paths": [meta[i]["path"] for i in order],
        "runtime": devices["runtime"],
        "with_enhancement": {
            "resize_ms": summarize_stage(with_rows, "resize_ms"),
            "clahe_gamma_ms": summarize_stage(with_rows, "enhance_ms"),
            "normalize_ms": summarize_stage(with_rows, "normalize_ms"),
            "resize_normalize_ms": summarize_stage(with_rows, "resize_normalize_ms"),
            "embedding_ms": summarize_stage(with_rows, "embed_ms"),
            "matching_ms": summarize_stage(with_rows, "match_ms"),
            "total_ms": summarize_stage(with_rows, "total_ms"),
        },
        "without_enhancement": {
            "resize_ms": summarize_stage(without_rows, "resize_ms"),
            "normalize_ms": summarize_stage(without_rows, "normalize_ms"),
            "resize_normalize_ms": summarize_stage(without_rows, "resize_normalize_ms"),
            "embedding_ms": summarize_stage(without_rows, "embed_ms"),
            "matching_ms": summarize_stage(without_rows, "match_ms"),
            "total_ms": summarize_stage(without_rows, "total_ms"),
        },
        "notes": [
            "resize_ms includes face detection because that is where the system resizes the crop to 224.",
            "normalize_ms is BGR to RGB float32 in the 0-255 range. MobileNet preprocessing stays inside the model.",
            "CLAHE+Gamma is the enhancement that exists in the project. There is no separate denoising filter.",
            "Low-light simulation is not part of latency; it belongs to dataset construction.",
        ],
    }

    image_size = tuple(cfg["preprocessing"]["image_size"])
    # tf.image.resize expects (height, width). Config stores [224, 224].
    print("Softmax sanity check on the stored test JPEGs...", flush=True)
    softmax = {
        "replay_enhanced_in_memory": softmax_on_arrays(full_model, prep["test_enhanced"], labels),
        "stored_jpeg_tf_loader": softmax_on_stored(full_model, meta, (image_size[1], image_size[0])),
        "note": "Closed-set argmax accuracy of the softmax head. This is not the cosine protocol.",
    }
    saved_metrics_path = ROOT / "outputs" / "reports" / "mobilenetv2_metrics.json"
    if saved_metrics_path.exists():
        import json

        softmax["previously_saved_file"] = json.loads(saved_metrics_path.read_text(encoding="utf-8"))
        softmax["previously_saved_file_path"] = str(saved_metrics_path.relative_to(ROOT)).replace("\\", "/")

    if roc_store:
        np.savez_compressed(CACHE / "roc_curves.npz", **roc_store)
    write_csv(RESULTS / "scores.csv", score_rows)
    write_csv(RESULTS / "threshold_sweep.csv", sweep_rows)
    write_csv(
        RESULTS / "table_4_3_metrics.csv",
        [{"experiment": name, **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}} for name, metrics in experiments.items()],
    )
    write_csv(
        RESULTS / "table_4_4_masked.csv",
        [{"experiment": name, **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}} for name, metrics in masked.items()],
    )
    latency_rows = []
    for condition, block in (
        ("with_clahe_gamma", latency["with_enhancement"]),
        ("without_clahe_gamma", latency["without_enhancement"]),
    ):
        for stage, stats in block.items():
            latency_rows.append({"condition": condition, "stage": stage, "n": stats["n"], "mean_ms": stats["mean_ms"], "std_ms": stats["std_ms"]})
    write_csv(RESULTS / "table_4_5_latency.csv", latency_rows)

    return {
        "model_path": str(weights),
        "model_bytes": weights.stat().st_size,
        "model_param_count": int(full_model.count_params()),
        "embedding_dim_runtime": embedding_dim,
        "cosine_helper_abs_delta": float(cosine_delta),
        "devices": devices,
        "experiments": experiments,
        "masked": masked,
        "mask_method_counts": method_counts,
        "mask_landmarker_error": landmarker_error,
        "mask_color_bgr": list(MASK_BGR),
        "mask_rule": "Convex hull of FaceMesh landmarks at or below the nose tip. If landmarks are missing, a filled rectangle on the lower third.",
        "threshold_sweep": sweep_by_exp,
        "latency": latency,
        "softmax_sanity": softmax,
        "test_set_sha256": prep["replay"]["test_paths_sha256"],
    }
