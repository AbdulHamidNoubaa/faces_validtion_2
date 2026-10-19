"""Shared path and config helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT: Path | None = None
_CONFIG_CACHE: dict[str, dict[str, Any]] = {}


def project_root() -> Path:
    """Return absolute path to project root (folder containing config/)."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    return _PROJECT_ROOT


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config with process-level caching (read from disk once)."""
    env_cfg = os.environ.get("BFR_CONFIG")

    if config_path is None:
        config_path = project_root() / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = project_root() / config_path

    if env_cfg and Path(env_cfg).exists():
        config_path = Path(env_cfg)

    cache_key = str(config_path)
    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _CONFIG_CACHE[cache_key] = cfg
    return cfg


def resolve_path(relative: str | Path) -> Path:
    """Resolve a path relative to project root."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return project_root() / p


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create standard output / data directories from config."""
    for key in ("raw_dir", "processed_dir", "models_dir", "logs_dir", "reports_dir"):
        resolve_path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
