"""Process and mirror bank employee enrollment photos."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from django.utils import timezone

from src.inference.embeddings import embedding_from_json, embedding_from_rgb_float, embedding_to_json
from src.preprocessing.face_pipeline import process_image_file
from src.utils.config_loader import load_config, project_root, resolve_path
from web.checkpoint.models import EmployeePhoto


def _enrollment_paths(cfg: dict) -> tuple[Path, Path, Path]:
    raw = resolve_path(cfg.get("enrollment", {}).get("raw_dir", "data/enrollment/raw"))
    processed = resolve_path(cfg.get("enrollment", {}).get("processed_dir", "data/processed/employees"))
    mirror = resolve_path(cfg.get("enrollment", {}).get("custom_mirror_dir", "data/raw/custom"))
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    mirror.mkdir(parents=True, exist_ok=True)
    return raw, processed, mirror


def process_employee_photo(photo: EmployeePhoto, architecture: str = "mobilenetv2") -> EmployeePhoto:
    """Run the same preprocessing as LFW splits; store embedding for gallery match."""
    cfg = load_config()
    _, processed_root, mirror_root = _enrollment_paths(cfg)

    src = Path(photo.image.path)
    if not src.exists():
        photo.status = EmployeePhoto.Status.FAILED
        photo.error_message = "ملف الصورة غير موجود على القرص."
        photo.save(update_fields=["status", "error_message"])
        return photo

    slug = photo.employee.folder_slug
    tag = "ll" if photo.apply_low_light else "nl"
    stem = f"{photo.pk}_{src.stem}_{tag}.jpg"
    dest = processed_root / slug / stem
    rel = dest.relative_to(project_root()).as_posix()

    rng = np.random.default_rng(int(photo.pk))
    ok, face_detected = process_image_file(
        src, dest, cfg, apply_low_light=photo.apply_low_light, rng=rng
    )
    if not ok:
        photo.status = EmployeePhoto.Status.FAILED
        photo.error_message = "فشلت قراءة/حفظ الصورة بعد المعالجة."
        photo.save(update_fields=["status", "error_message"])
        return photo

    # Mirror into data/raw/custom/<code>/ for optional dataset rebuild
    mirror_dest = mirror_root / slug / stem
    mirror_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dest, mirror_dest)

    embedding_json = ""
    try:
        bgr = cv2.imread(str(dest))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        vec = embedding_from_rgb_float(rgb, architecture=architecture, cfg=cfg)
        embedding_json = embedding_to_json(vec)
    except Exception as exc:
        photo.error_message = f"Embedding: {exc}"

    photo.status = EmployeePhoto.Status.PROCESSED
    photo.processed_relpath = rel
    photo.face_detected = face_detected
    photo.embedding_json = embedding_json
    photo.processed_at = timezone.now()
    photo.save(
        update_fields=[
            "status",
            "processed_relpath",
            "face_detected",
            "embedding_json",
            "processed_at",
            "error_message",
        ]
    )
    return photo


def load_gallery_embeddings(active_only: bool = True) -> list[tuple[int, str, str, np.ndarray]]:
    """Returns list of (photo_id, employee_code, full_name_ar, embedding)."""
    qs = EmployeePhoto.objects.filter(status=EmployeePhoto.Status.PROCESSED).exclude(
        embedding_json=""
    )
    if active_only:
        qs = qs.filter(employee__is_active=True)
    out = []
    for p in qs.select_related("employee"):
        try:
            vec = embedding_from_json(p.embedding_json)
            out.append((p.pk, p.employee.employee_code, p.employee.full_name_ar, vec))
        except Exception:
            continue
    return out
