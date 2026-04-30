from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
ROOT = REPO_ROOT
DEFAULT_INPUT_DIR = ROOT / "src" / "fisher_geometry" / "verification_results" / "actual_val_loss_gap_sampled"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize sampled actual loss-gap verification runs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*_actual_loss_gap_sampled.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    summary = build_summary(payloads)

    output_json = args.output_json or (args.input_dir / "summary_actual_loss_gap_sampled_all_datasets.json")
    output_md = args.output_md or output_json.with_suffix(".md")
    output = {"summary": summary, "datasets": payloads}
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(summary, payloads), encoding="utf-8")
    print(to_markdown(summary, payloads), flush=True)
    print(f"Wrote {output_json}", flush=True)
    print(f"Wrote {output_md}", flush=True)


def build_summary(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    records = [record for payload in payloads for record in payload.get("records", [])]
    actual = np.asarray([float(row["actual_loss_gap"]) for row in records], dtype=np.float64)
    pred = np.asarray(
        [float(row["pred_score"]) if row.get("pred_score") is not None else np.nan for row in records],
        dtype=np.float64,
    )
    mask = np.isfinite(pred)
    return {
        "num_datasets": int(len(payloads)),
        "num_targets": int(len(records)),
        "mean_actual_loss_gap": nanmean(actual),
        "actual_positive_fraction": float(np.mean(actual > 0.0)) if actual.size else float("nan"),
        "mean_loss_osfi": nanmean(pred[mask]),
        "loss_osfi_actual_sign_agreement": (
            float(np.mean((pred[mask] > 0.0) == (actual[mask] > 0.0))) if np.any(mask) else float("nan")
        ),
        "loss_osfi_actual_spearman": spearmanr(pred[mask], actual[mask]) if np.sum(mask) >= 2 else float("nan"),
        "loss_osfi_actual_pearson": pearsonr(pred[mask], actual[mask]) if np.sum(mask) >= 2 else float("nan"),
    }


def nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def spearmanr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    from kernel_surrogate.kernel_attribution import spearmanr as _spearmanr

    return float(_spearmanr(np.asarray(lhs, dtype=np.float64), np.asarray(rhs, dtype=np.float64)))


def pearsonr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    x = np.asarray(lhs, dtype=np.float64)
    y = np.asarray(rhs, dtype=np.float64)
    if x.size < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fmt(value: Any, precision: int = 6) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.{precision}g}"


def to_markdown(summary: dict[str, Any], payloads: list[dict[str, Any]]) -> str:
    lines = [
        "# Sampled Actual Validation Loss Gap Summary",
        "",
        "Actual gap is `self_only_target_validation_loss - full_checkpoint_target_validation_loss`.",
        "Positive values mean the full-data checkpoint has lower target-group validation loss than a model trained only on that target group.",
        "Validation samples are assigned to train groups by nearest normalized train-group centroid.",
        "",
        "## Overall",
        "",
        f"- datasets: `{summary['num_datasets']}`",
        f"- sampled target groups: `{summary['num_targets']}`",
        f"- mean actual loss gap: `{summary['mean_actual_loss_gap']:.6g}`",
        f"- actual positive fraction: `{summary['actual_positive_fraction']:.3f}`",
        f"- L-OSFI/actual sign agreement: `{summary['loss_osfi_actual_sign_agreement']:.3f}`",
        f"- L-OSFI/actual Spearman: `{summary['loss_osfi_actual_spearman']:.3f}`",
        f"- L-OSFI/actual Pearson: `{summary['loss_osfi_actual_pearson']:.3f}`",
        "",
        "## Dataset Summary",
        "",
        "| dataset | groups | min size | mean L-OSFI | mean actual gap | actual positive frac | sign agree | Spearman | Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for payload in sorted(payloads, key=lambda item: str(item.get("config", {}).get("dataset", ""))):
        cfg = payload["config"]
        dataset_summary = payload["summary"]
        groups = ",".join(str(row["target_group"]) for row in payload.get("records", []))
        lines.append(
            f"| {cfg['dataset']} | {groups} | "
            f"{cfg.get('effective_min_group_size', 'n/a')} | "
            f"{fmt(dataset_summary.get('mean_loss_osfi'))} | "
            f"{fmt(dataset_summary.get('mean_actual_loss_gap'))} | "
            f"{fmt(dataset_summary.get('actual_positive_fraction'), precision=3)} | "
            f"{fmt(dataset_summary.get('loss_osfi_actual_sign_agreement'), precision=3)} | "
            f"{fmt(dataset_summary.get('loss_osfi_actual_spearman'), precision=3)} | "
            f"{fmt(dataset_summary.get('loss_osfi_actual_pearson'), precision=3)} |"
        )
    lines.extend(
        [
            "",
            "## Target Records",
            "",
            "| dataset | group | train size | eval size | pred gap | L-OSFI | full val loss | self val loss | actual val gap | sign agree |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for payload in sorted(payloads, key=lambda item: str(item.get("config", {}).get("dataset", ""))):
        dataset = payload["config"]["dataset"]
        for row in payload.get("records", []):
            lines.append(
                f"| {dataset} | {row['target_group']} | {row['train_group_size']} | {row['eval_group_size']} | "
                f"{float(row['pred_loss_gap']):.6g} | {float(row['loss_osfi']):.6g} | "
                f"{row['full_loss']:.6g} | {row['self_loss']:.6g} | "
                f"{row['actual_loss_gap']:.6g} | {row['sign_agree_pred_score']} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
