"""Download or load LFW / custom face datasets."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.config_loader import ensure_dirs, load_config, project_root, resolve_path
from src.utils.logger import get_logger

logger = get_logger("data.download")


def _save_rgb_as_bgr_jpg(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def fetch_lfw_to_raw(cfg: dict) -> Path:
    """
    Download LFW via sklearn and write person/image folder tree under data/raw/lfw.
    Filters by min_images_per_identity and optional max_identities.
    """
    from sklearn.datasets import fetch_lfw_people

    min_faces = int(cfg["dataset"]["min_images_per_identity"])
    max_ids = cfg["dataset"].get("max_identities")

    logger.info("Fetching LFW (min_faces_per_person=%s)...", min_faces)
    # color=True, resize=None keeps higher resolution for later face crop
    lfw = fetch_lfw_people(
        min_faces_per_person=min_faces,
        resize=1.0,
        color=True,
        funneled=True,
        download_if_missing=True,
    )

    # Map target index -> name
    target_names = list(lfw.target_names)
    # Optionally keep only top-N identities by image count
    counts: dict[int, int] = defaultdict(int)
    for t in lfw.target:
        counts[int(t)] += 1

    identity_ids = sorted(counts.keys(), key=lambda i: counts[i], reverse=True)
    if max_ids is not None:
        identity_ids = identity_ids[: int(max_ids)]
    keep = set(identity_ids)

    out_root = resolve_path(cfg["paths"]["raw_dir"]) / "lfw"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    per_person_idx: dict[int, int] = defaultdict(int)
    saved = 0
    for img, t in tqdm(zip(lfw.images, lfw.target), total=len(lfw.target), desc="Writing LFW"):
        tid = int(t)
        if tid not in keep:
            continue
        name = target_names[tid].replace(" ", "_")
        per_person_idx[tid] += 1
        # sklearn returns float images in [0,1]
        rgb = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        dest = out_root / name / f"{name}_{per_person_idx[tid]:04d}.jpg"
        _save_rgb_as_bgr_jpg(rgb, dest)
        saved += 1

    meta = {
        "source": "sklearn.datasets.fetch_lfw_people",
        "min_images_per_identity": min_faces,
        "max_identities": max_ids,
        "num_identities": len(keep),
        "num_images": saved,
        "note": "Academic use; low-light may be synthetically applied later.",
    }
    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("Saved %s images for %s identities -> %s", saved, len(keep), out_root)
    return out_root


def discover_custom(raw_custom: Path) -> Path:
    if not raw_custom.exists():
        raise FileNotFoundError(
            f"Custom dataset not found at {raw_custom}. "
            "Create folders data/raw/custom/<PersonName>/*.jpg"
        )
    return raw_custom


def load_or_download(cfg: dict | None = None) -> Path:
    """Main entry: return path to raw person-folder dataset."""
    if cfg is None:
        cfg = load_config()
    ensure_dirs(cfg)

    name = cfg["dataset"]["name"].lower()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])

    if name == "lfw":
        existing = raw_dir / "lfw"
        # Reuse if already populated
        if existing.exists() and any(existing.iterdir()):
            logger.info("Using existing LFW raw data at %s", existing)
            return existing
        return fetch_lfw_to_raw(cfg)

    if name == "custom":
        return discover_custom(raw_dir / "custom")

    if name == "celeba":
        celeba = raw_dir / "celeba"
        if not celeba.exists():
            raise FileNotFoundError(
                "CelebA is large; place aligned faces under data/raw/celeba/<id>/*.jpg "
                "or switch dataset.name to 'lfw' in config.yaml."
            )
        return celeba

    raise ValueError(f"Unknown dataset.name: {name}")


def main() -> None:
    cfg = load_config()
    path = load_or_download(cfg)
    print(f"Raw dataset ready at: {path}")
    print(f"Project root: {project_root()}")


if __name__ == "__main__":
    main()
