"""Match live face against enrolled bank employee gallery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.inference.embeddings import cosine_similarity, embedding_from_rgb_float
from src.utils.config_loader import load_config
from web.checkpoint.services.employee_processing import load_gallery_embeddings


@dataclass
class EmployeeMatch:
    employee_code: str
    full_name_ar: str
    similarity: float
    photo_id: int


def find_best_employee_match(
    rgb_float: np.ndarray,
    architecture: str = "mobilenetv2",
) -> EmployeeMatch | None:
    """Return highest-similarity enrolled employee, or None if gallery is empty."""
    cfg = load_config()
    gallery = load_gallery_embeddings(active_only=True)
    if not gallery:
        return None

    query = embedding_from_rgb_float(rgb_float, architecture=architecture, cfg=cfg)
    best: EmployeeMatch | None = None
    for photo_id, code, name, vec in gallery:
        sim = cosine_similarity(query, vec)
        if best is None or sim > best.similarity:
            best = EmployeeMatch(
                employee_code=code,
                full_name_ar=name,
                similarity=sim,
                photo_id=photo_id,
            )
    return best


def match_enrolled_employee(
    rgb_float: np.ndarray,
    architecture: str = "mobilenetv2",
    threshold: float | None = None,
) -> EmployeeMatch | None:
    cfg = load_config()
    if threshold is None:
        threshold = float(cfg.get("enrollment", {}).get("match_threshold", 0.72))
    best = find_best_employee_match(rgb_float, architecture)
    if best is None or best.similarity < threshold:
        return None
    return best


def _set_result_from_match(result, match: EmployeeMatch) -> None:
    result.identity = f"{match.full_name_ar} ({match.employee_code})"
    result.confidence = match.similarity
    result.accepted = match.similarity >= result.threshold
    result.decision_ar = "مقبول ✓" if result.accepted else "مرفوض ✗"
    result.decision_en = "ACCEPTED" if result.accepted else "REJECTED"
    result.match_source = "employee_gallery"
    result.employee_code = match.employee_code
    result.matched_photo_id = match.photo_id


def apply_verification_mode(result, mode: str = "employees_first") -> None:
    """
    modes:
      - employees_only: only enrolled bank staff; reject unknown faces
      - employees_first: prefer employee gallery, else keep LFW classifier
      - lfw_only: keep classifier output only
    """
    if result.preprocessed_rgb is None:
        return

    mode = (mode or "employees_first").lower()
    cfg = load_config()
    min_sim = float(cfg.get("enrollment", {}).get("match_threshold", 0.72))
    best = find_best_employee_match(result.preprocessed_rgb, result.architecture)

    if mode == "lfw_only":
        result.match_source = "classifier"
        return

    if best is not None and best.similarity >= min_sim:
        _set_result_from_match(result, best)
        return

    if mode == "employees_only":
        result.identity = "غير مسجّل — لا يوجد في سجل الموظفين"
        result.confidence = float(best.similarity) if best else 0.0
        result.accepted = False
        result.decision_ar = "مرفوض ✗"
        result.decision_en = "REJECTED"
        result.match_source = "employee_gallery"
        result.employee_code = None
        result.matched_photo_id = None


def apply_employee_match_to_result(result) -> None:
    """Backward-compatible: employees_first behavior."""
    apply_verification_mode(result, "employees_first")
