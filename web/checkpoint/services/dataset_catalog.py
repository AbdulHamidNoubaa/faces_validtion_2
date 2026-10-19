"""Read-only catalog of prepared LFW / processed dataset metadata."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from django.core.cache import cache

from src.utils.config_loader import load_config, project_root, resolve_path


def _load_splits() -> dict:
    cfg = load_config()
    path = resolve_path(cfg["paths"]["splits_file"])
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_labels() -> dict:
    cfg = load_config()
    path = resolve_path(cfg["paths"]["label_map"])
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dataset_summary() -> dict:
    cached = cache.get("bfr_dataset_summary")
    if cached:
        return cached

    splits = _load_splits()
    labels = _load_labels()
    if not splits:
        summary = {
            "ready": False,
            "message": "لم يتم تجهيز البيانات بعد. شغّل: python -m src.data_preparation.build_dataset_splits",
        }
        return summary

    counts = splits.get("counts", {})
    by_identity: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "low_light": 0})
    for split_name, items in splits.get("splits", {}).items():
        for item in items:
            ident = item.get("identity", "?")
            by_identity[ident][split_name] = by_identity[ident].get(split_name, 0) + 1
            if item.get("low_light"):
                by_identity[ident]["low_light"] += 1

    summary = {
        "ready": True,
        "num_classes": splits.get("num_classes", len(labels.get("name_to_label", {}))),
        "counts": counts,
        "total_images": sum(counts.values()) if counts else 0,
        "low_light_simulation": splits.get("low_light_simulation", False),
        "source_raw": splits.get("source_raw", ""),
        "identities_count": len(by_identity),
        "methodology_note": (
            "البيانات: subset من LFW مع محاكاة اصطناعية للإضاءة المنخفضة لأغراض البحث. "
            "راجع data/README_DATA.md في التقرير الأكاديمي."
        ),
    }
    cache.set("bfr_dataset_summary", summary, 120)
    return summary


def list_identities(limit: int = 200) -> list[dict]:
    splits = _load_splits()
    if not splits:
        return []
    stats: dict[str, dict] = {}
    for split_name, items in splits.get("splits", {}).items():
        for item in items:
            ident = item["identity"]
            if ident not in stats:
                stats[ident] = {
                    "identity": ident,
                    "label": item.get("label"),
                    "train": 0,
                    "val": 0,
                    "test": 0,
                    "low_light": 0,
                    "total": 0,
                }
            stats[ident]["total"] += 1
            if item.get("low_light"):
                stats[ident]["low_light"] += 1
            sn = split_name
            stats[ident][sn] = stats[ident].get(sn, 0) + 1
    ordered = sorted(stats.values(), key=lambda x: x["total"], reverse=True)
    return ordered[:limit]


def samples_for_identity(identity: str, split: str = "train", limit: int = 16) -> list[dict]:
    splits = _load_splits()
    if not splits:
        return []
    items = splits.get("splits", {}).get(split, [])
    out = []
    for item in items:
        if item.get("identity") != identity:
            continue
        rel = item["path"].replace("\\", "/")
        abs_path = project_root() / rel
        out.append(
            {
                "path": rel,
                "exists": abs_path.exists(),
                "low_light": item.get("low_light", False),
                "label": item.get("label"),
            }
        )
        if len(out) >= limit:
            break
    return out
