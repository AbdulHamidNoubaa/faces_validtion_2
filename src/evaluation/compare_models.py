"""Compare MobileNetV2 against baseline models under the same protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluation.evaluate import evaluate
from src.evaluation.inference_benchmark import benchmark_model, count_params, model_size_mb
from src.training.train import train
from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger("compare")


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table without requiring the tabulate package."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def compare_models(
    cfg: dict | None = None,
    architectures: list[str] | None = None,
    train_if_missing: bool = True,
) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()

    if architectures is None:
        # Primary + two baselines for the thesis comparison table
        architectures = ["mobilenetv2", "mobilenetv1", "resnet50"]

    rows = []
    for arch in architectures:
        models_dir = resolve_path(cfg["paths"]["models_dir"])
        best = models_dir / f"{arch}_best.keras"
        final = models_dir / f"{arch}_final.keras"
        model_path = best if best.exists() else final

        if not model_path.exists():
            if not train_if_missing:
                logger.warning("Skipping %s — model not found", arch)
                continue
            logger.info("Training missing model: %s", arch)
            train(cfg, architecture=arch)
            model_path = best if best.exists() else (models_dir / f"{arch}_final.keras")

        metrics = evaluate(cfg, architecture=arch)
        bench = benchmark_model(
            model_path,
            input_shape=tuple(cfg["model"]["input_shape"]),
            warmup=int(cfg["evaluation"]["benchmark_warmup"]),
            runs=int(cfg["evaluation"]["benchmark_runs"]),
        )
        rows.append(
            {
                "architecture": arch,
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "auc_micro": metrics.get("auc_micro"),
                "inference_mean_ms": bench["mean_ms"],
                "model_size_mb": model_size_mb(model_path),
                "num_params": count_params(model_path),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = resolve_path(cfg["paths"]["reports_dir"]) / "model_comparison.csv"
    df.to_csv(out_csv, index=False)

    # Append markdown table to evaluation report
    md_path = resolve_path(cfg["paths"]["reports_dir"]) / "evaluation_report.md"
    table_md = "\n## Model Comparison Table\n\n" + _df_to_markdown(df) + "\n"
    if md_path.exists():
        text = md_path.read_text(encoding="utf-8")
        if "## Model Comparison Table" in text:
            text = text.split("## Model Comparison Table")[0].rstrip() + "\n"
            md_path.write_text(text + table_md, encoding="utf-8")
        else:
            with open(md_path, "a", encoding="utf-8") as f:
                f.write(table_md)
    else:
        md_path.write_text("# Evaluation Report\n" + table_md, encoding="utf-8")

    logger.info("Wrote comparison -> %s", out_csv)
    print(df.to_string(index=False))
    return df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--archs",
        nargs="+",
        default=None,
        help="Architectures to compare (default: mobilenetv2 mobilenetv1 resnet50)",
    )
    parser.add_argument("--no-train", action="store_true", help="Do not train missing models")
    args = parser.parse_args()
    compare_models(
        load_config(args.config),
        architectures=args.archs,
        train_if_missing=not args.no_train,
    )


if __name__ == "__main__":
    main()
