"""Shared helpers for thesis experiments. Does not modify the system."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
CACHE = RESULTS / "cache"
SEED = 42
THRESHOLD = 0.52
TIMING_N = 200

EXPERIMENTS = ("E1", "E2", "E3", "E3b")
MASKED = ("M_NL", "M_LL", "M_LL_ENH")


def ensure_dirs() -> None:
    for path in (RESULTS, FIGURES, CACHE, ROOT / "experiments" / "assets"):
        path.mkdir(parents=True, exist_ok=True)


def set_python_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class JsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        return super().default(o)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, cls=JsonEncoder),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def mean_std(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def load_project_config() -> dict:
    from src.utils.config_loader import load_config

    return load_config(ROOT / "config" / "config.yaml")


def powershell_json(command: str) -> Any:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    text = (completed.stdout or "").strip()
    if not text:
        return {"error": (completed.stderr or "").strip()[:500]}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text[:2000]}


def collect_hardware() -> dict[str, Any]:
    cpu = powershell_json(
        "Get-CimInstance Win32_Processor | Select-Object -First 1 "
        "Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json"
    )
    mem = powershell_json(
        "Get-CimInstance Win32_ComputerSystem | "
        "Select-Object TotalPhysicalMemory,Manufacturer,Model | ConvertTo-Json"
    )
    gpu = powershell_json(
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json"
    )
    total = None
    if isinstance(mem, dict) and mem.get("TotalPhysicalMemory"):
        total = int(mem["TotalPhysicalMemory"])
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "cpu_cim": cpu,
        "memory_cim": mem,
        "ram_bytes": total,
        "ram_gib": (total / (1024**3)) if total else None,
        "gpu_cim": gpu,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }


def package_versions(names: list[str]) -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    found: dict[str, str] = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "NOT_INSTALLED"
    return found


def find_source_line(path: Path, needle: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return {
                "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "line": index,
                "text": line.strip(),
            }
    return None


def scan_for_token(token: str) -> list[dict[str, Any]]:
    roots = [ROOT / "src", ROOT / "web", ROOT / "config", ROOT / "app", ROOT / "tests", ROOT / "README.md"]
    hits: list[dict[str, Any]] = []
    files: list[Path] = []
    for entry in roots:
        if entry.is_file():
            files.append(entry)
        elif entry.exists():
            files.extend(p for p in entry.rglob("*") if p.is_file())
    allowed = {".py", ".yaml", ".yml", ".md", ".html", ".js", ".txt"}
    for path in files:
        if path.suffix.lower() not in allowed and path.name != "README.md":
            continue
        if "__pycache__" in path.parts or "migrations" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        for index, line in enumerate(text.splitlines(), start=1):
            if token in line:
                hits.append(
                    {
                        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                        "line": index,
                        "text": line.strip()[:240],
                    }
                )
    return hits


def fmt_ratio(value: float | None) -> str:
    if value is None:
        return "غير معرّف"
    return f"{value:.6f} ({value * 100:.2f}%)"


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "غير معرّف"
    return f"{value:.{digits}f}"


def thresholds() -> list[float]:
    return [round(0.30 + i * 0.02, 2) for i in range(26)]
