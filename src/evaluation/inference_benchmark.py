"""Benchmark single-image inference latency (ms)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger("benchmark")


def benchmark_model(
    model_path: str | Path,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    warmup: int = 5,
    runs: int = 50,
) -> dict:
    from src.models.load_model import load_keras_model

    model = load_keras_model(model_path)
    dummy = np.random.randint(0, 255, size=(1, *input_shape)).astype(np.float32)

    for _ in range(warmup):
        _ = model.predict(dummy, verbose=0)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model.predict(dummy, verbose=0)
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times)
    result = {
        "model_path": str(model_path),
        "warmup": warmup,
        "runs": runs,
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }
    return result


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def count_params(model_path: Path) -> int:
    from src.models.load_model import load_keras_model

    model = load_keras_model(model_path)
    return int(model.count_params())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--arch", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    arch = (args.arch or cfg["model"]["architecture"]).lower()
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    path = models_dir / f"{arch}_best.keras"
    if not path.exists():
        path = models_dir / f"{arch}_final.keras"
    if not path.exists():
        raise FileNotFoundError(path)

    res = benchmark_model(
        path,
        input_shape=tuple(cfg["model"]["input_shape"]),
        warmup=int(cfg["evaluation"]["benchmark_warmup"]),
        runs=int(cfg["evaluation"]["benchmark_runs"]),
    )
    res["size_mb"] = model_size_mb(path)
    res["num_params"] = count_params(path)

    out = resolve_path(cfg["paths"]["reports_dir"]) / f"{arch}_inference_benchmark.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    logger.info("Benchmark: mean=%.2f ms | size=%.2f MB | params=%s", res["mean_ms"], res["size_mb"], res["num_params"])
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
