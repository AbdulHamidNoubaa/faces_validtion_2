"""Build train/val/test splits with optional low-light simulation + preprocessing."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data_preparation.download_or_load_dataset import load_or_download
from src.preprocessing.face_pipeline import process_image_file
from src.utils.config_loader import ensure_dirs, load_config, project_root, resolve_path
from src.utils.logger import get_logger

logger = get_logger("data.splits")


def _list_person_images(raw_root: Path) -> dict[str, list[Path]]:
    mapping: dict[str, list[Path]] = {}
    for person_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        imgs = sorted(
            [
                p
                for p in person_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
        if imgs:
            mapping[person_dir.name] = imgs
    return mapping


def _stratified_split(
    paths: list[Path],
    train_r: float,
    val_r: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    rng = random.Random(seed)
    items = list(paths)
    rng.shuffle(items)
    n = len(items)
    n_train = max(1, int(round(n * train_r)))
    n_val = max(0, int(round(n * val_r)))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    n_test = n - n_train - n_val
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    if not test and items:
        # ensure at least one test if possible by moving from train
        if len(train) > 1:
            test = [train.pop()]
        elif len(val) > 0:
            test = [val.pop()]
    return train, val, test


def _process_one(
    src: Path,
    dest: Path,
    cfg: dict,
    apply_ll: bool,
    rng: np.random.Generator,
) -> bool:
    ok, _ = process_image_file(src, dest, cfg, apply_low_light=apply_ll, rng=rng)
    if not ok:
        logger.warning("Could not process %s", src)
    return ok


def build_splits(cfg: dict | None = None) -> Path:
    if cfg is None:
        cfg = load_config()
    ensure_dirs(cfg)
    seed = int(cfg["project"]["seed"])
    rng = np.random.default_rng(seed)
    random.seed(seed)

    raw_root = load_or_download(cfg)
    people = _list_person_images(raw_root)
    if not people:
        raise RuntimeError(f"No images found under {raw_root}")

    min_imgs = int(cfg["dataset"]["min_images_per_identity"])
    people = {k: v for k, v in people.items() if len(v) >= min_imgs}
    max_ids = cfg["dataset"].get("max_identities")
    if max_ids is not None:
        # keep identities with most images
        ordered = sorted(people.items(), key=lambda kv: len(kv[1]), reverse=True)
        people = dict(ordered[: int(max_ids)])

    label_map = {name: idx for idx, name in enumerate(sorted(people.keys()))}
    inv_label = {v: k for k, v in label_map.items()}

    processed_root = resolve_path(cfg["paths"]["processed_dir"]) / "images"
    if processed_root.exists():
        # clean previous processed images
        import shutil

        shutil.rmtree(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    train_r = float(cfg["dataset"]["train_ratio"])
    val_r = float(cfg["dataset"]["val_ratio"])
    ll_cfg = cfg["dataset"]["low_light"]
    ll_enabled = bool(ll_cfg.get("enabled", True))
    ll_frac = float(ll_cfg.get("fraction", 0.5))

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for person, paths in tqdm(people.items(), desc="Building splits"):
        label = label_map[person]
        train_p, val_p, test_p = _stratified_split(paths, train_r, val_r, seed + label)
        for split_name, split_paths in (
            ("train", train_p),
            ("val", val_p),
            ("test", test_p),
        ):
            for i, src in enumerate(split_paths):
                apply_ll = ll_enabled and (rng.random() < ll_frac)
                tag = "ll" if apply_ll else "nl"
                dest = (
                    processed_root
                    / split_name
                    / person
                    / f"{src.stem}_{tag}.jpg"
                )
                ok = _process_one(src, dest, cfg, apply_ll, rng)
                if not ok:
                    continue
                splits[split_name].append(
                    {
                        "path": str(dest.relative_to(project_root())).replace("\\", "/"),
                        "label": label,
                        "identity": person,
                        "low_light": apply_ll,
                    }
                )

    # Persist
    label_path = resolve_path(cfg["paths"]["label_map"])
    splits_path = resolve_path(cfg["paths"]["splits_file"])
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump({"label_to_name": inv_label, "name_to_label": label_map}, f, indent=2)
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_classes": len(label_map),
                "counts": {k: len(v) for k, v in splits.items()},
                "splits": splits,
                "source_raw": str(raw_root),
                "low_light_simulation": ll_enabled,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Done. Classes=%s | train=%s val=%s test=%s",
        len(label_map),
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    logger.info("Wrote %s and %s", label_path, splits_path)
    return splits_path


def main() -> None:
    build_splits()


if __name__ == "__main__":
    main()
