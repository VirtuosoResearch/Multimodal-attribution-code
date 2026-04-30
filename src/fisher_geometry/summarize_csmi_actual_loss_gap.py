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
DEFAULT_CSMI_JSON = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "verification_results"
    / "cross_sample_modal_interaction"
    / "cross_sample_modal_interaction_all_datasets.json"
)
DEFAULT_ACTUAL_DIR = ROOT / "src" / "fisher_geometry" / "verification_results" / "actual_val_loss_gap_sampled"
DEFAULT_OUTPUT_JSON = (
    ROOT
    / "src"
    / "kernel_surrogate"
    / "verification_results"
    / "cross_sample_modal_interaction"
    / "csmi_actual_val_loss_gap.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Join CSMI scores to sampled actual validation loss gaps.")
    parser.add_argument("--csmi-json", type=Path, default=DEFAULT_CSMI_JSON)
    parser.add_argument("--actual-dir", type=Path, default=DEFAULT_ACTUAL_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    csmi_payload = json.loads(args.csmi_json.read_text(encoding="utf-8"))
    actual_lookup = load_actual_lookup(args.actual_dir)
    records = []
    for dataset_output in csmi_payload.get("datasets", []):
        for record in dataset_output.get("records", []):
            key = (str(record["dataset"]), int(record["target_group"]))
            actual = actual_lookup.get(key)
            if actual is None:
                continue
            records.append(
                {
                    **record,
                    "actual_val_gap": float(actual["actual_loss_gap"]),
                    "full_val_loss": float(actual["full_loss"]),
                    "self_val_loss": float(actual["self_loss"]),
                    "eval_group_size": int(actual.get("eval_group_size", actual.get("group_size", 0))),
                    "train_group_size": int(actual.get("train_group_size", actual.get("group_size", 0))),
                    "actual_positive": bool(float(actual["actual_loss_gap"]) > 0.0),
                    "csmi_sign_agree": bool((float(record["csmi"]) > 0.0) == (float(actual["actual_loss_gap"]) > 0.0)),
                    "gain_sign_agree": bool(
                        (float(record["interaction_specific_gain"]) > 0.0)
                        == (float(actual["actual_loss_gap"]) > 0.0)
                    ),
                }
            )

    output = {
        "config": {
            "csmi_json": str(args.csmi_json),
            "actual_dir": str(args.actual_dir),
            "actual_gap": "self_only_target_validation_loss - full_checkpoint_target_validation_loss",
            "note": "This validates sampled target groups only; it is not a retraining ablation of residual-only interaction.",
        },
        "overall": summarize_records(records),
        "datasets": summarize_by_dataset(records),
        "records": records,
    }
    output_md = args.output_md or args.output_json.with_suffix(".md")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(output), encoding="utf-8")
    print(to_markdown(output), flush=True)
    print(f"Wrote {args.output_json}", flush=True)
    print(f"Wrote {output_md}", flush=True)


def load_actual_lookup(actual_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    lookup = {}
    for path in sorted(actual_dir.glob("*_actual_loss_gap_sampled.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = str(payload.get("config", {}).get("dataset", path.name.split("_seed")[0]))
        for record in payload.get("records", []):
            lookup[(dataset, int(record["target_group"]))] = record
    return lookup


def summarize_by_dataset(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    datasets = sorted({str(record["dataset"]) for record in records})
    return {
        dataset: summarize_records([record for record in records if str(record["dataset"]) == dataset])
        for dataset in datasets
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    queries = sorted({str(record["query"]) for record in records})
    return {query: summarize_query([record for record in records if str(record["query"]) == query]) for query in queries}


def summarize_query(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "num_targets": 0,
            "mean_actual_val_gap": float("nan"),
            "actual_positive_fraction": float("nan"),
            "mean_csmi": float("nan"),
            "mean_interaction_specific_gain": float("nan"),
            "csmi_actual_sign_agreement": float("nan"),
            "gain_actual_sign_agreement": float("nan"),
            "csmi_actual_spearman": float("nan"),
            "gain_actual_spearman": float("nan"),
            "csmi_actual_pearson": float("nan"),
            "gain_actual_pearson": float("nan"),
        }
    actual = np.asarray([float(record["actual_val_gap"]) for record in records], dtype=np.float64)
    csmi = np.asarray([float(record["csmi"]) for record in records], dtype=np.float64)
    gain = np.asarray([float(record["interaction_specific_gain"]) for record in records], dtype=np.float64)
    return {
        "num_targets": int(len(records)),
        "mean_actual_val_gap": finite_mean(actual),
        "actual_positive_fraction": positive_fraction(actual),
        "mean_csmi": finite_mean(csmi),
        "mean_interaction_specific_gain": finite_mean(gain),
        "csmi_actual_sign_agreement": float(np.mean((csmi > 0.0) == (actual > 0.0))),
        "gain_actual_sign_agreement": float(np.mean((gain > 0.0) == (actual > 0.0))),
        "csmi_actual_spearman": spearmanr(csmi, actual),
        "gain_actual_spearman": spearmanr(gain, actual),
        "csmi_actual_pearson": pearsonr(csmi, actual),
        "gain_actual_pearson": pearsonr(gain, actual),
    }


def finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def positive_fraction(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values > 0.0)) if values.size else float("nan")


def spearmanr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    x = np.asarray(lhs, dtype=np.float64)
    y = np.asarray(rhs, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    return pearsonr(rankdata(x), rankdata(y))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for idx, count in enumerate(counts):
            if count > 1:
                ranks[inverse == idx] = float(np.mean(ranks[inverse == idx]))
    return ranks


def pearsonr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    x = np.asarray(lhs, dtype=np.float64)
    y = np.asarray(rhs, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
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


def to_markdown(output: dict[str, Any]) -> str:
    lines = [
        "# CSMI vs Actual Validation Loss Gap",
        "",
        "Actual gap is `self_only_target_validation_loss - full_checkpoint_target_validation_loss`.",
        "Positive actual gap means the full-data checkpoint has lower target validation loss than target-only retraining.",
        "This joins CSMI to sampled target groups only; it is not a residual-only retraining ablation.",
        "",
        "## Overall",
        "",
        "| query | targets | mean actual gap | actual+ | mean CSMI | mean gain | sign CSMI | sign gain | CSMI Spearman | Gain Spearman | CSMI Pearson | Gain Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for query, summary in sorted(output["overall"].items()):
        lines.append(summary_row(query, summary))
    lines.extend(
        [
            "",
            "## Dataset Summary",
            "",
            "| dataset | query | targets | mean actual gap | actual+ | mean CSMI | mean gain | sign gain | Gain Spearman | Gain Pearson |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, query_map in sorted(output["datasets"].items()):
        for query, summary in sorted(query_map.items()):
            lines.append(
                f"| {dataset} | {query} | {int(summary['num_targets'])} | "
                f"{fmt(summary['mean_actual_val_gap'])} | {fmt(summary['actual_positive_fraction'], 3)} | "
                f"{fmt(summary['mean_csmi'])} | {fmt(summary['mean_interaction_specific_gain'])} | "
                f"{fmt(summary['gain_actual_sign_agreement'], 3)} | "
                f"{fmt(summary['gain_actual_spearman'], 3)} | {fmt(summary['gain_actual_pearson'], 3)} |"
            )
    lines.extend(
        [
            "",
            "## Target Records",
            "",
            "| dataset | group | query | actual gap | CSMI | gain | full val | self val | sign gain |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in output["records"]:
        lines.append(
            f"| {record['dataset']} | {record['target_group']} | {record['query']} | "
            f"{fmt(record['actual_val_gap'])} | {fmt(record['csmi'])} | "
            f"{fmt(record['interaction_specific_gain'])} | {fmt(record['full_val_loss'])} | "
            f"{fmt(record['self_val_loss'])} | {record['gain_sign_agree']} |"
        )
    return "\n".join(lines) + "\n"


def summary_row(query: str, summary: dict[str, float]) -> str:
    return (
        f"| {query} | {int(summary['num_targets'])} | {fmt(summary['mean_actual_val_gap'])} | "
        f"{fmt(summary['actual_positive_fraction'], 3)} | {fmt(summary['mean_csmi'])} | "
        f"{fmt(summary['mean_interaction_specific_gain'])} | "
        f"{fmt(summary['csmi_actual_sign_agreement'], 3)} | {fmt(summary['gain_actual_sign_agreement'], 3)} | "
        f"{fmt(summary['csmi_actual_spearman'], 3)} | {fmt(summary['gain_actual_spearman'], 3)} | "
        f"{fmt(summary['csmi_actual_pearson'], 3)} | {fmt(summary['gain_actual_pearson'], 3)} |"
    )


if __name__ == "__main__":
    main()
