from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUT_DIR = ROOT / "src" / "fisher_geometry" / "verification_results" / "fisher_loss_osfi"
DEFAULT_OUTPUT_JSON = DEFAULT_INPUT_DIR / "summary_loss_osfi_all_datasets.json"
DEFAULT_OUTPUT_MD = DEFAULT_INPUT_DIR / "summary_loss_osfi_all_datasets.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Loss-OSFI experiments across datasets.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    paths = sorted(path for path in args.input_dir.glob("*_seed0_loss_osfi_cuda1.json") if "sanity" not in path.name)
    if not paths:
        raise FileNotFoundError(f"no Loss-OSFI result JSONs found under {args.input_dir}")

    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload["config"]
        t1 = payload["summary"]["table_1_loss_osfi"]
        t2 = payload["summary"]["table_2_osfi_vs_loss_osfi"]
        t4 = payload["summary"]["table_4_loss_specific_control"]
        label = payload["summary"]["label_random_control"]
        rows.append(
            {
                "dataset": str(cfg["dataset"]),
                "path": str(path),
                "device": str(cfg["device"]),
                "checkpoint": str(cfg["full_checkpoint"]),
                "mean_loss_osfi": float(t1["mean_loss_osfi"]),
                "median_loss_osfi": float(t1["median_loss_osfi"]),
                "loss_osfi_positive_fraction": float(t1["loss_osfi_positive_fraction"]),
                "mean_pred_loss_gap": float(t1["mean_pred_loss_gap"]),
                "p_value": float(t1["bootstrap_p_value_mean_leq_zero"]),
                "ci_low": float(t1["mean_osfi_ci_low"]),
                "ci_high": float(t1["mean_osfi_ci_high"]),
                "mean_osfi": float(t2["mean_osfi"]),
                "mean_osfi_h": float(t2["mean_osfi_h"]),
                "osfi_loss_spearman": float(t2["osfi_loss_osfi_spearman"]),
                "osfi_h_loss_spearman": float(t2["osfi_h_loss_osfi_spearman"]),
                "osfi_loss_kendall": float(t2["osfi_loss_osfi_kendall"]),
                "sign_agreement": float(t2["osfi_loss_osfi_sign_agreement"]),
                "random_h_mean_loss_osfi": float(t4["random_h_mean_loss_osfi"]),
                "loss_specific_gain": float(t4["loss_specific_gain"]),
                "loss_specific_gain_positive_fraction": float(t4["loss_specific_gain_positive_fraction"]),
                "label_random_mean_loss_osfi": float(label.get("label_random_mean_loss_osfi", float("nan"))),
                "loss_group_gain": float(label.get("loss_group_gain", float("nan"))),
            }
        )

    aggregate = {
        "num_datasets": int(len(rows)),
        "datasets": [row["dataset"] for row in rows],
        "mean_loss_osfi_across_datasets": safe_mean(row["mean_loss_osfi"] for row in rows),
        "mean_osfi_across_datasets": safe_mean(row["mean_osfi"] for row in rows),
        "mean_loss_specific_gain": safe_mean(row["loss_specific_gain"] for row in rows),
        "mean_loss_group_gain": safe_mean(row["loss_group_gain"] for row in rows),
        "mean_osfi_loss_spearman": safe_mean(row["osfi_loss_spearman"] for row in rows),
        "mean_osfi_h_loss_spearman": safe_mean(row["osfi_h_loss_spearman"] for row in rows),
        "datasets_with_significant_positive_loss_osfi": [
            row["dataset"] for row in rows if row["p_value"] < 0.05 and row["mean_loss_osfi"] > 0.0
        ],
        "datasets_with_positive_loss_specific_gain": [
            row["dataset"] for row in rows if row["loss_specific_gain"] > 0.0
        ],
        "datasets_with_positive_loss_group_gain": [
            row["dataset"] for row in rows if row["loss_group_gain"] > 0.0
        ],
    }
    output = {"aggregate": aggregate, "datasets": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    args.output_md.write_text(to_markdown(output), encoding="utf-8")
    print(to_markdown(output), flush=True)
    print(f"Wrote {args.output_json}", flush=True)
    print(f"Wrote {args.output_md}", flush=True)


def safe_mean(values: Iterable[float]) -> float:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def to_markdown(output: dict[str, Any]) -> str:
    aggregate = output["aggregate"]
    rows = output["datasets"]
    lines = [
        "# Loss-Connected OSFI Across Datasets",
        "",
        f"- datasets: {', '.join(aggregate['datasets'])}",
        f"- mean L-OSFI across datasets: {aggregate['mean_loss_osfi_across_datasets']:.6g}",
        f"- mean OSFI across datasets: {aggregate['mean_osfi_across_datasets']:.6g}",
        f"- mean LossSpecificGain: {aggregate['mean_loss_specific_gain']:.6g}",
        f"- mean L-GroupGain: {aggregate['mean_loss_group_gain']:.6g}",
        f"- mean OSFI/L-OSFI Spearman: {aggregate['mean_osfi_loss_spearman']:.3f}",
        f"- significant positive L-OSFI datasets: {', '.join(aggregate['datasets_with_significant_positive_loss_osfi'])}",
        f"- positive LossSpecificGain datasets: {', '.join(aggregate['datasets_with_positive_loss_specific_gain'])}",
        f"- positive L-GroupGain datasets: {', '.join(aggregate['datasets_with_positive_loss_group_gain']) or 'none'}",
        "",
        "## Table 1: Loss-OSFI Existence",
        "",
        "| Dataset | Mean L-OSFI | Median L-OSFI | L-OSFI+ | Mean Pred Gap | p-value | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['mean_loss_osfi']:.6g} | {row['median_loss_osfi']:.6g} | "
            f"{row['loss_osfi_positive_fraction']:.3f} | {row['mean_pred_loss_gap']:.6g} | "
            f"{row['p_value']:.4g} | [{row['ci_low']:.6g}, {row['ci_high']:.6g}] |"
        )
    lines.extend(
        [
            "",
            "## Table 2: OSFI vs Loss-OSFI",
            "",
            "| Dataset | Mean OSFI | Mean L-OSFI | OSFI-H | Spearman | Kendall | Sign Agreement |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['mean_osfi']:.6g} | {row['mean_loss_osfi']:.6g} | "
            f"{row['mean_osfi_h']:.6g} | {row['osfi_loss_spearman']:.3f} | "
            f"{row['osfi_loss_kendall']:.3f} | {row['sign_agreement']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Table 4: Loss-Specific Control",
            "",
            "| Dataset | Mean L-OSFI | Random-H L-OSFI | LossSpecificGain | Gain+ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['mean_loss_osfi']:.6g} | "
            f"{row['random_h_mean_loss_osfi']:.6g} | {row['loss_specific_gain']:.6g} | "
            f"{row['loss_specific_gain_positive_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Label-Random Control",
            "",
            "| Dataset | KMeans L-OSFI | Label-Random L-OSFI | L-GroupGain |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['mean_loss_osfi']:.6g} | "
            f"{row['label_random_mean_loss_osfi']:.6g} | {row['loss_group_gain']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Positive L-OSFI means other groups reduce target loss-curvature-weighted inverse-Fisher uncertainty.",
            "- Random-H control checks whether the effect is specific to target loss curvature rather than arbitrary PSD directions.",
            "- Label-random control is stricter and tests whether k-means grouping is more loss-relevant than label-preserving random grouping.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
