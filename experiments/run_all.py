"""Run every thesis measurement and write results/."""

from __future__ import annotations

import traceback
from datetime import datetime

from experiments.common import (
    RESULTS,
    SEED,
    THRESHOLD,
    collect_hardware,
    ensure_dirs,
    package_versions,
    set_python_seed,
    write_json,
)
from experiments.evaluate import evaluate
from experiments.prepare import code_facts, dataset_stats, leakage_check, load_project_config, prepare, training_info
from experiments.report import write_report

PACKAGES = [
    "tensorflow",
    "opencv-python",
    "Django",
    "mediapipe",
    "scikit-learn",
    "scikit-image",
    "numpy",
    "Pillow",
    "pandas",
    "matplotlib",
    "PyYAML",
    "scipy",
]


def main() -> None:
    ensure_dirs()
    set_python_seed(SEED)
    payload: dict = {
        "seed": SEED,
        "threshold": THRESHOLD,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        print("Collecting environment and dataset facts...", flush=True)
        payload["environment"] = collect_hardware()
        payload["environment"]["packages"] = package_versions(PACKAGES)
        cfg = load_project_config()
        payload["dataset"] = dataset_stats(cfg)
        payload["leakage"] = leakage_check(cfg)
        payload["training"] = training_info(cfg)
        payload["code_facts"] = code_facts(cfg)
        write_json(RESULTS / "all_results.json", payload)

        print("Preparing paired images...", flush=True)
        prepared = prepare(cfg)
        payload["quality"] = prepared["quality_summary"]
        payload["replay"] = prepared["replay"]
        write_json(RESULTS / "all_results.json", payload)

        print("Running verification experiments...", flush=True)
        payload.update(evaluate(prepared))
        payload["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(RESULTS / "all_results.json", payload)
        write_report(payload)
        print(f"Wrote {RESULTS / 'summary.md'}", flush=True)
    except Exception:
        payload["error"] = traceback.format_exc()
        payload["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(RESULTS / "all_results.json", payload)
        try:
            write_report(payload)
        except Exception:
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
