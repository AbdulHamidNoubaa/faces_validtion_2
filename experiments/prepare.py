"""Replay the dataset build and prepare paired images for the experiments."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiments.common import (
    CACHE,
    RESULTS,
    ROOT,
    SEED,
    find_source_line,
    load_project_config,
    mean_std,
    scan_for_token,
    write_csv,
    write_json,
)
from src.data_preparation.build_dataset_splits import _list_person_images, _stratified_split
from src.data_preparation.low_light_simulation import simulate_low_light
from src.data_preparation.download_or_load_dataset import load_or_download
from src.preprocessing.face_detection_alignment import detect_face_haar, detect_face_mediapipe
from src.preprocessing.image_enhancement import enhance_image
from src.utils.config_loader import resolve_path


class ReplayError(RuntimeError):
    pass


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ReplayError(f"Could not read image: {path}")
    return image


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_splits(cfg: dict) -> tuple[Path, dict]:
    path = resolve_path(cfg["paths"]["splits_file"])
    payload = json_load(path)
    return path, payload


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def source_stem(processed_name: str) -> str:
    stem = Path(processed_name).stem
    if stem.endswith("_nl") or stem.endswith("_ll"):
        return stem.rsplit("_", 1)[0]
    return stem


def dataset_stats(cfg: dict | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_project_config()
    splits_path, data = load_splits(cfg)
    label_path = resolve_path(cfg["paths"]["label_map"])
    labels = json_load(label_path) if label_path.exists() else {}

    per_split: dict[str, dict[str, int]] = {}
    per_identity: Counter[str] = Counter()
    per_identity_split: dict[str, Counter[str]] = defaultdict(Counter)
    suffix_mismatches = 0
    rows: list[dict[str, Any]] = []

    for split in ("train", "val", "test"):
        items = data["splits"][split]
        nl = ll = 0
        for item in items:
            flag = bool(item["low_light"])
            name = Path(item["path"]).name
            suffix_ll = name.endswith("_ll.jpg") or name.endswith("_ll.jpeg") or name.endswith("_ll.png")
            if flag != suffix_ll:
                suffix_mismatches += 1
            if flag:
                ll += 1
            else:
                nl += 1
            per_identity[item["identity"]] += 1
            per_identity_split[split][item["identity"]] += 1
        per_split[split] = {"nl": nl, "ll": ll, "total": nl + ll, "listed": len(items)}

    identity_rows = []
    for identity in sorted(per_identity):
        identity_rows.append(
            {
                "identity": identity,
                "total": per_identity[identity],
                "train": per_identity_split["train"][identity],
                "val": per_identity_split["val"][identity],
                "test": per_identity_split["test"][identity],
            }
        )
    write_csv(RESULTS / "per_identity_counts.csv", identity_rows)

    def range_stats(counter: Counter[str]) -> dict[str, Any]:
        values = list(counter.values())
        if not values:
            return {"n_identities": 0, "min": None, "max": None, "mean": None}
        return {
            "n_identities": len(values),
            "min": int(min(values)),
            "max": int(max(values)),
            "mean": float(np.mean(values)),
            "sum": int(sum(values)),
        }

    recounted = {split: per_split[split]["total"] for split in ("train", "val", "test")}
    stored_counts = data.get("counts", {})
    summary = {
        "splits_file": str(splits_path.relative_to(ROOT)).replace("\\", "/"),
        "num_classes_field": data.get("num_classes"),
        "label_map_size": len(labels.get("name_to_label", {})),
        "stored_counts": stored_counts,
        "recounted_counts": recounted,
        "counts_match_file_field": stored_counts == recounted,
        "per_split": per_split,
        "grand_total": int(sum(recounted.values())),
        "images_per_identity_overall": range_stats(per_identity),
        "images_per_identity_by_split": {
            split: range_stats(per_identity_split[split]) for split in ("train", "val", "test")
        },
        "suffix_flag_mismatches": suffix_mismatches,
        "low_light_simulation_field": data.get("low_light_simulation"),
    }
    write_json(RESULTS / "dataset_stats.json", summary)
    write_csv(
        RESULTS / "table_3_1_dataset.csv",
        [
            {
                "split": split,
                "nl": per_split[split]["nl"],
                "ll": per_split[split]["ll"],
                "total": per_split[split]["total"],
            }
            for split in ("train", "val", "test")
        ]
        + [
            {
                "split": "all",
                "nl": sum(per_split[s]["nl"] for s in per_split),
                "ll": sum(per_split[s]["ll"] for s in per_split),
                "total": sum(per_split[s]["total"] for s in per_split),
            }
        ],
    )
    return summary


def leakage_check(cfg: dict | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_project_config()
    _, data = load_splits(cfg)
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    md5_groups: dict[str, list[str]] = defaultdict(list)
    missing_processed = []

    for split, items in data["splits"].items():
        for item in items:
            rel = item["path"]
            by_source[source_stem(rel)].append({"split": split, "path": rel, "low_light": item["low_light"]})
            full = ROOT / rel
            if not full.exists():
                missing_processed.append(rel)
                continue
            md5_groups[file_md5(full)].append(f"{split}:{rel}")

    cross_split = []
    duplicated_source = []
    both_nl_and_ll = []
    for source, entries in by_source.items():
        splits = sorted({entry["split"] for entry in entries})
        if len(entries) > 1:
            duplicated_source.append({"source": source, "entries": entries})
        if len(splits) > 1:
            cross_split.append({"source": source, "splits": splits, "entries": entries})
        tags = {bool(entry["low_light"]) for entry in entries}
        if tags == {True, False}:
            both_nl_and_ll.append({"source": source, "entries": entries})

    duplicate_md5 = [
        {"md5": digest, "paths": paths}
        for digest, paths in md5_groups.items()
        if len(paths) > 1
    ]

    raw_root = load_or_download(cfg)
    raw_hashes: dict[str, list[str]] = defaultdict(list)
    raw_files = [
        path
        for path in raw_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    for path in raw_files:
        raw_hashes[file_md5(path)].append(str(path.relative_to(ROOT)).replace("\\", "/"))
    duplicate_raw = [
        {"md5": digest, "paths": paths} for digest, paths in raw_hashes.items() if len(paths) > 1
    ]

    report = {
        "n_source_stems": len(by_source),
        "n_processed_missing": len(missing_processed),
        "missing_processed_examples": missing_processed[:20],
        "same_source_in_more_than_one_split": cross_split,
        "n_same_source_across_splits": len(cross_split),
        "same_source_more_than_once": duplicated_source,
        "n_duplicated_sources": len(duplicated_source),
        "same_source_has_both_nl_and_ll": both_nl_and_ll,
        "n_sources_with_both_nl_and_ll": len(both_nl_and_ll),
        "duplicate_processed_md5": duplicate_md5,
        "n_duplicate_processed_md5": len(duplicate_md5),
        "n_raw_images_hashed": len(raw_files),
        "duplicate_raw_md5": duplicate_raw,
        "n_duplicate_raw_md5": len(duplicate_raw),
        "leakage_found": bool(cross_split or duplicate_md5 or duplicate_raw or both_nl_and_ll),
    }
    write_json(RESULTS / "leakage.json", report)
    return report


def training_info(cfg: dict | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_project_config()
    import json

    history_path = resolve_path(cfg["paths"]["logs_dir"]) / "mobilenetv2_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else None
    epochs_ran = len(history["loss"]) if history and "loss" in history else None
    learning_rates = history.get("learning_rate") if history else None
    ft_lr = float(cfg["model"]["fine_tune"]["fine_tune_learning_rate"])
    ft_start = None
    if learning_rates:
        for index, value in enumerate(learning_rates):
            if float(value) <= ft_lr * 1.05:
                ft_start = index
                break
    head_ran = ft_start if ft_start is not None else epochs_ran
    ft_ran = None if epochs_ran is None or head_ran is None else epochs_ran - head_ran

    model_py = ROOT / "src" / "models" / "mobilenetv2_model.py"
    train_py = ROOT / "src" / "training" / "train.py"
    callbacks_py = ROOT / "src" / "training" / "callbacks.py"
    config_path = ROOT / "config" / "config.yaml"
    sources = {
        "loss": find_source_line(model_py, 'loss="sparse_categorical_crossentropy"'),
        "embedding_layer": find_source_line(model_py, 'name="embedding"'),
        "optimizer_adam": find_source_line(model_py, "optimizers.Adam"),
        "epochs_fit_head": find_source_line(train_py, 'epochs=int(cfg["training"]["epochs_head"])'),
        "epochs_fit_ft": find_source_line(train_py, 'epochs=int(cfg["training"]["epochs_finetune"])'),
        "early_stopping": find_source_line(callbacks_py, "EarlyStopping"),
        "config_optimizer": find_source_line(config_path, 'optimizer: "adam"'),
        "config_lr": find_source_line(config_path, "learning_rate: 0.001"),
        "config_batch": find_source_line(config_path, "batch_size:"),
        "config_dense": find_source_line(config_path, "dense_units:"),
        "config_ft_lr": find_source_line(config_path, "fine_tune_learning_rate:"),
    }
    best = resolve_path(cfg["paths"]["models_dir"]) / "mobilenetv2_best.keras"
    final = resolve_path(cfg["paths"]["models_dir"]) / "mobilenetv2_final.keras"
    info = {
        "optimizer": cfg["training"]["optimizer"],
        "learning_rate_head": float(cfg["training"]["learning_rate"]),
        "learning_rate_finetune": ft_lr,
        "batch_size": int(cfg["training"]["batch_size"]),
        "epochs_head_config": int(cfg["training"]["epochs_head"]),
        "epochs_finetune_config": int(cfg["training"]["epochs_finetune"]),
        "epochs_recorded_in_history": epochs_ran,
        "history_file": str(history_path.relative_to(ROOT)).replace("\\", "/") if history_path.exists() else None,
        "inferred_head_epochs_ran": head_ran,
        "inferred_finetune_epochs_ran": ft_ran,
        "recorded_learning_rates": learning_rates,
        "loss": "sparse_categorical_crossentropy",
        "embedding_units_config": int(cfg["model"]["dense_units"]),
        "dropout": float(cfg["model"]["dropout"]),
        "freeze_base": bool(cfg["model"]["freeze_base"]),
        "unfreeze_last_n_layers": int(cfg["model"]["fine_tune"]["unfreeze_last_n_layers"]),
        "early_stopping_patience": int(cfg["training"]["early_stopping_patience"]),
        "reduce_lr_patience": int(cfg["training"]["reduce_lr_patience"]),
        "reduce_lr_factor": float(cfg["training"]["reduce_lr_factor"]),
        "min_lr": float(cfg["training"]["min_lr"]),
        "seed": int(cfg["project"]["seed"]),
        "input_shape": list(cfg["model"]["input_shape"]),
        "sources": sources,
        "best_model_exists": best.exists(),
        "best_model_bytes": best.stat().st_size if best.exists() else None,
        "final_model_exists": final.exists(),
        "final_model_bytes": final.stat().st_size if final.exists() else None,
        "best_model": str(best.relative_to(ROOT)).replace("\\", "/") if best.exists() else None,
    }
    write_json(RESULTS / "training_settings.json", info)
    return info


def code_facts(cfg: dict | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_project_config()
    facts = {
        "enhancement_method": cfg["preprocessing"]["enhancement_method"],
        "clahe": cfg["preprocessing"]["clahe"],
        "gamma": cfg["preprocessing"]["gamma"],
        "retinex_sigma_not_default_path": cfg["preprocessing"]["retinex"],
        "low_light": cfg["dataset"]["low_light"],
        "image_size": cfg["preprocessing"]["image_size"],
        "face_detector": cfg["preprocessing"]["face_detector"],
        "demo_default_threshold": cfg["demo"]["default_threshold"],
        "evaluation_confidence_threshold": cfg["evaluation"]["confidence_threshold"],
        "enrollment_match_threshold": cfg["enrollment"]["match_threshold"],
        "token_0_52_hits_in_system_code": scan_for_token("0.52"),
        "readme_data": "data/README_DATA.md",
        "pipeline_order_in_code": [
            "optional simulate_low_light on the full BGR frame",
            "detect_and_align_face (crop + resize to image_size); if no face, cv2.resize",
            "enhance_image: CLAHE on the L channel then gamma LUT",
        ],
    }
    write_json(RESULTS / "code_facts.json", facts)
    return facts


def face_box(image_bgr: np.ndarray, cfg: dict) -> tuple[int, int, int, int] | None:
    detector = str(cfg["preprocessing"].get("face_detector", "mediapipe")).lower()
    min_conf = float(cfg["preprocessing"].get("mediapipe_min_confidence", 0.5))

    def mediapipe_box():
        try:
            return detect_face_mediapipe(image_bgr, min_conf)
        except Exception:
            return None

    def haar_box():
        try:
            return detect_face_haar(image_bgr)
        except Exception:
            return None

    if detector == "mediapipe":
        box = mediapipe_box()
        return haar_box() if box is None else box
    box = haar_box()
    return mediapipe_box() if box is None else box


def crop_resize(image_bgr: np.ndarray, box, size: tuple[int, int]) -> tuple[np.ndarray, bool]:
    if box is None:
        return cv2.resize(image_bgr, size), False
    x1, y1, x2, y2 = box
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return cv2.resize(image_bgr, size), False
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA), True


def psnr_bgr(reference: np.ndarray, other: np.ndarray) -> float | None:
    diff = reference.astype(np.float64) - other.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    if mse <= 1e-12:
        return None
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def ssim_gray(reference_bgr: np.ndarray, other_bgr: np.ndarray) -> tuple[float, str]:
    gray_a = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(other_bgr, cv2.COLOR_BGR2GRAY)
    from skimage.metrics import structural_similarity

    value = structural_similarity(
        gray_a,
        gray_b,
        data_range=255,
        win_size=11,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    return float(value), "skimage.metrics.structural_similarity grayscale win_size=11 gaussian sigma=1.5"


def choose_held_out(test_items: list[dict]) -> dict[str, Any]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for item in test_items:
        by_id[item["identity"]].append(item)
    eligible = []
    for identity in sorted(by_id):
        nl = sum(1 for item in by_id[identity] if not item["low_light"])
        ll = sum(1 for item in by_id[identity] if item["low_light"])
        if nl >= 1 and ll >= 1:
            eligible.append(identity)
    if len(eligible) < 5:
        raise ReplayError(
            f"Need at least 5 identities with both an NL and an LL test image; found {len(eligible)}."
        )
    rng = random_mod()
    held = sorted(rng.sample(eligible, 5))
    return {"eligible_count": len(eligible), "eligible": eligible, "held_out": held, "selector": "random.Random(42).sample on sorted eligible identities"}


def random_mod():
    import random

    return random.Random(SEED)


def choose_gallery_clean(splits: dict, held_out: set[str]) -> list[dict]:
    import random

    rng = random.Random(SEED)
    identities = sorted({item["identity"] for items in splits.values() for item in items} - held_out)
    selected: list[dict] = []
    for identity in identities:
        train_items = [item for item in splits["train"] if item["identity"] == identity]
        val_items = [item for item in splits["val"] if item["identity"] == identity]

        def shuffled(pool: list[dict]) -> list[dict]:
            copied = list(pool)
            rng.shuffle(copied)
            return copied

        ordered = (
            shuffled([item for item in train_items if not item["low_light"]])
            + shuffled([item for item in val_items if not item["low_light"]])
            + shuffled([item for item in train_items if item["low_light"]])
            + shuffled([item for item in val_items if item["low_light"]])
        )
        take = ordered[:5]
        for item in take:
            selected.append(
                {
                    "path": item["path"],
                    "identity": item["identity"],
                    "label": int(item["label"]),
                    "low_light": bool(item["low_light"]),
                    "split": "train" if item in train_items else "val",
                }
            )
    return selected


def align_for_recognition(image_bgr: np.ndarray, cfg: dict) -> tuple[np.ndarray, bool]:
    from src.preprocessing.face_detection_alignment import detect_and_align_face

    aligned = detect_and_align_face(image_bgr, cfg)
    if aligned is None:
        size = tuple(cfg["preprocessing"]["image_size"])
        return cv2.resize(image_bgr, size), False
    return aligned, True


def prepare(cfg: dict | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_project_config()
    splits_path, data = load_splits(cfg)
    held_info = choose_held_out(data["splits"]["test"])
    held_out = set(held_info["held_out"])
    gallery_meta = choose_gallery_clean(data["splits"], held_out)
    gallery_paths = {item["path"] for item in gallery_meta}
    test_path_set = {item["path"] for item in data["splits"]["test"]}
    if gallery_paths & test_path_set:
        raise ReplayError("Gallery contains a test-split path.")
    gallery_ids = {item["identity"] for item in gallery_meta}
    missing_ids = ({item["identity"] for item in data["splits"]["test"]} - held_out) - gallery_ids
    if missing_ids:
        raise ReplayError(f"Enrolled identities with no gallery photo: {sorted(missing_ids)}")
    test_items = data["splits"]["test"]
    test_paths = [item["path"] for item in test_items]
    test_index = {path: index for index, path in enumerate(test_paths)}
    gallery_index = {item["path"]: index for index, item in enumerate(gallery_meta)}

    import random

    sample_rng = random.Random(SEED)
    ll_test_paths = [item["path"] for item in test_items if item["low_light"]]
    if len(ll_test_paths) < 6:
        raise ReplayError(f"Need 6 LL test images for the figure; found {len(ll_test_paths)}.")
    sample_paths = sample_rng.sample(ll_test_paths, 6)

    expected: dict[str, dict] = {}
    for split, items in data["splits"].items():
        for item in items:
            expected[item["path"]] = {**item, "split": split}

    raw_root = load_or_download(cfg)
    seed = int(cfg["project"]["seed"])
    rng = np.random.default_rng(seed)
    people = _list_person_images(raw_root)
    min_imgs = int(cfg["dataset"]["min_images_per_identity"])
    people = {key: value for key, value in people.items() if len(value) >= min_imgs}
    max_ids = cfg["dataset"].get("max_identities")
    if max_ids is not None:
        ordered = sorted(people.items(), key=lambda kv: len(kv[1]), reverse=True)
        people = dict(ordered[: int(max_ids)])
    label_map = {name: idx for idx, name in enumerate(sorted(people.keys()))}
    train_r = float(cfg["dataset"]["train_ratio"])
    val_r = float(cfg["dataset"]["val_ratio"])
    ll_cfg = cfg["dataset"]["low_light"]
    ll_enabled = bool(ll_cfg.get("enabled", True))
    ll_frac = float(ll_cfg.get("fraction", 0.5))
    size = tuple(cfg["preprocessing"]["image_size"])

    n_test = len(test_paths)
    n_gallery = len(gallery_meta)
    test_aligned = np.zeros((n_test, size[1], size[0], 3), dtype=np.uint8)
    test_enhanced = np.zeros_like(test_aligned)
    test_detected = np.zeros(n_test, dtype=bool)
    gallery_enhanced = np.zeros((n_gallery, size[1], size[0], 3), dtype=np.uint8)
    gallery_detected = np.zeros(n_gallery, dtype=bool)
    test_frames: list[np.ndarray | None] = [None] * n_test
    seen: set[str] = set()
    quality_rows: list[dict[str, Any]] = []
    samples: dict[str, np.ndarray] = {}
    jpeg_maes: list[float] = []
    ssim_impl = None

    print(f"Replaying {sum(len(v) for v in people.values())} raw images...", flush=True)
    processed_count = 0
    for person, paths in people.items():
        label = label_map[person]
        train_p, val_p, test_p = _stratified_split(paths, train_r, val_r, seed + label)
        for split_name, split_paths in (("train", train_p), ("val", val_p), ("test", test_p)):
            for src in split_paths:
                apply_ll = ll_enabled and (rng.random() < ll_frac)
                rel = f"data/processed/images/{split_name}/{person}/{src.stem}_{'ll' if apply_ll else 'nl'}.jpg"
                if rel not in expected:
                    raise ReplayError(f"Replay produced an unexpected path: {rel}")
                exp = expected[rel]
                if bool(exp["low_light"]) != bool(apply_ll) or exp["identity"] != person or int(exp["label"]) != int(label):
                    raise ReplayError(
                        f"Replay diverged at {rel}: file={exp['low_light']}/{exp['identity']}/{exp['label']} "
                        f"replay={apply_ll}/{person}/{label}"
                    )
                seen.add(rel)
                needed = rel in test_index or rel in gallery_index or apply_ll or rel in sample_paths
                raw = read_bgr(src) if needed or apply_ll else None
                dark = None
                if apply_ll:
                    if raw is None:
                        raw = read_bgr(src)
                    dark = simulate_low_light(
                        raw,
                        brightness_range=ll_cfg["brightness_range"],
                        noise_std=float(ll_cfg["noise_std"]),
                        gamma_dark=ll_cfg["gamma_dark"],
                        rng=rng,
                    )
                    box = face_box(raw, cfg)
                    nl_crop, found = crop_resize(raw, box, size)
                    ll_crop, _ = crop_resize(dark, box, size)
                    enh_crop = enhance_image(ll_crop, cfg)
                    ssim_before, ssim_impl = ssim_gray(nl_crop, ll_crop)
                    ssim_after, ssim_impl = ssim_gray(nl_crop, enh_crop)
                    quality_rows.append(
                        {
                            "path": rel,
                            "split": split_name,
                            "identity": person,
                            "psnr_before": psnr_bgr(nl_crop, ll_crop),
                            "psnr_after": psnr_bgr(nl_crop, enh_crop),
                            "ssim_before": ssim_before,
                            "ssim_after": ssim_after,
                            "nl_face_box_found": found,
                        }
                    )
                    if rel in sample_paths:
                        samples[rel] = np.stack([nl_crop, ll_crop, enh_crop], axis=0)

                if rel in test_index or rel in gallery_index:
                    probe = dark if apply_ll else raw
                    if probe is None:
                        probe = read_bgr(src)
                    aligned, detected = align_for_recognition(probe, cfg)
                    enhanced = enhance_image(aligned, cfg)
                    stored = ROOT / rel
                    if stored.exists():
                        stored_img = read_bgr(stored)
                        if stored_img.shape == enhanced.shape:
                            jpeg_maes.append(float(np.mean(np.abs(enhanced.astype(np.float32) - stored_img.astype(np.float32)))))
                    if rel in test_index:
                        idx = test_index[rel]
                        test_aligned[idx] = aligned
                        test_enhanced[idx] = enhanced
                        test_detected[idx] = detected
                        test_frames[idx] = probe.copy()
                    if rel in gallery_index:
                        gidx = gallery_index[rel]
                        gallery_enhanced[gidx] = enhanced
                        gallery_detected[gidx] = detected

                processed_count += 1
                if processed_count % 200 == 0:
                    print(f"  replayed {processed_count}", flush=True)

    missing = sorted(set(expected) - seen)
    extra = sorted(seen - set(expected))
    if missing or extra:
        raise ReplayError(f"Replay path set mismatch. missing={len(missing)} extra={len(extra)} example={missing[:3] or extra[:3]}")

    if any(frame is None for frame in test_frames):
        raise ReplayError("A test frame was not generated.")

    sample_stack = np.stack([samples[path] for path in sample_paths], axis=0)
    np.savez_compressed(CACHE / "samples_enhancement.npz", images=sample_stack, paths=np.array(sample_paths))
    np.savez_compressed(
        CACHE / "arrays.npz",
        test_aligned=test_aligned,
        test_enhanced=test_enhanced,
        gallery_enhanced=gallery_enhanced,
    )

    def finite(rows: list[dict], key: str) -> list[float]:
        return [float(row[key]) for row in rows if row[key] is not None]

    before_psnr = finite(quality_rows, "psnr_before")
    after_psnr = finite(quality_rows, "psnr_after")
    before_ssim = finite(quality_rows, "ssim_before")
    after_ssim = finite(quality_rows, "ssim_after")
    psnr_delta = [
        float(row["psnr_after"] - row["psnr_before"])
        for row in quality_rows
        if row["psnr_before"] is not None and row["psnr_after"] is not None
    ]
    ssim_delta = [
        float(row["ssim_after"] - row["ssim_before"])
        for row in quality_rows
        if row["ssim_before"] is not None and row["ssim_after"] is not None
    ]
    quality_summary = {
        "n_ll_images": len(quality_rows),
        "n_face_box_found": int(sum(1 for row in quality_rows if row["nl_face_box_found"])),
        "psnr_definition": "10*log10(255^2/MSE) on registered 224 BGR crops; MSE over H*W*C",
        "ssim_definition": ssim_impl,
        "registration": "Face box detected on the original NL frame, then applied to both NL and LL before resize.",
        "psnr_before": mean_std(before_psnr),
        "psnr_after": mean_std(after_psnr),
        "ssim_before": mean_std(before_ssim),
        "ssim_after": mean_std(after_ssim),
        "psnr_improvement": mean_std(psnr_delta),
        "ssim_improvement": mean_std(ssim_delta),
        "jpeg_mae_replay_enhanced_vs_stored": mean_std(jpeg_maes),
        "n_jpeg_compared": len(jpeg_maes),
    }
    write_csv(RESULTS / "image_quality_per_image.csv", quality_rows)
    write_csv(RESULTS / "table_4_2_image_quality.csv", [
        {"stage": "before_enhancement", **{f"psnr_{k}": quality_summary["psnr_before"][k] for k in ("n", "mean", "std")}, **{f"ssim_{k}": quality_summary["ssim_before"][k] for k in ("mean", "std")}},
        {"stage": "after_clahe_gamma", **{f"psnr_{k}": quality_summary["psnr_after"][k] for k in ("n", "mean", "std")}, **{f"ssim_{k}": quality_summary["ssim_after"][k] for k in ("mean", "std")}},
        {"stage": "after_minus_before", **{f"psnr_{k}": quality_summary["psnr_improvement"][k] for k in ("n", "mean", "std")}, **{f"ssim_{k}": quality_summary["ssim_improvement"][k] for k in ("mean", "std")}},
    ])
    write_json(RESULTS / "image_quality_summary.json", quality_summary)

    test_meta = []
    for item in test_items:
        test_meta.append(
            {
                "path": item["path"],
                "identity": item["identity"],
                "label": int(item["label"]),
                "low_light": bool(item["low_light"]),
                "held_out": item["identity"] in held_out,
            }
        )
    manifest = {
        "seed": seed,
        "splits_file": str(splits_path.relative_to(ROOT)).replace("\\", "/"),
        "test_paths_sha256": hashlib.sha256("\n".join(test_paths).encode("utf-8")).hexdigest(),
        "n_test": n_test,
        "held_out": held_info,
        "gallery": gallery_meta,
        "gallery_per_identity": dict(Counter(item["identity"] for item in gallery_meta)),
        "gallery_identities_with_fewer_than_3": sorted(
            identity
            for identity, count in Counter(item["identity"] for item in gallery_meta).items()
            if count < 3
        ),
        "test_face_detected": int(test_detected.sum()),
        "gallery_face_detected": int(gallery_detected.sum()),
        "n_replayed": processed_count,
        "sample_paths": sample_paths,
    }
    write_json(CACHE / "manifest.json", manifest)
    write_json(RESULTS / "gallery_manifest.json", {"held_out": held_info, "gallery": gallery_meta})
    write_json(RESULTS / "test_set_manifest.json", {"sha256": manifest["test_paths_sha256"], "paths": test_paths})

    print(
        f"Replay OK. test={n_test} gallery={n_gallery} ll_quality={len(quality_rows)} "
        f"jpeg_mae_mean={quality_summary['jpeg_mae_replay_enhanced_vs_stored']['mean']}",
        flush=True,
    )
    return {
        "cfg": cfg,
        "test_meta": test_meta,
        "test_aligned": test_aligned,
        "test_enhanced": test_enhanced,
        "test_frames": test_frames,
        "gallery_meta": gallery_meta,
        "gallery_enhanced": gallery_enhanced,
        "quality_summary": quality_summary,
        "sample_paths": sample_paths,
        "sample_images": sample_stack,
        "replay": manifest,
        "held_out": held_info["held_out"],
    }
