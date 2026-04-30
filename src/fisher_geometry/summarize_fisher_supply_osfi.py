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


DEFAULT_INPUT_DIR = ROOT / "src" / "fisher_geometry" / "verification_results" / "fisher_supply_osfi"
DEFAULT_OUTPUT_JSON = DEFAULT_INPUT_DIR / "summary_supply_osfi_all_datasets.json"
DEFAULT_OUTPUT_MD = DEFAULT_INPUT_DIR / "summary_supply_osfi_all_datasets.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize supplementary OSFI experiments.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    paths = sorted(path for path in args.input_dir.glob("*_seed0_supply_osfi_cuda1.json") if "sanity" not in path.name)
    if not paths:
        raise FileNotFoundError(f"no OSFI result JSONs found under {args.input_dir}")

    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload["config"]
        table_a = payload["summary"]["table_a"]
        table_b = payload["summary"]["table_b_controls"]
        table_c = payload["summary"]["table_c_source_level"]
        rows.append(
            {
                "dataset": str(cfg["dataset"]),
                "path": str(path),
                "device": str(cfg["device"]),
                "checkpoint": str(cfg["full_checkpoint"]),
                "mean_osfi": float(table_a["mean_osfi"]),
                "median_osfi": float(table_a["median_osfi"]),
                "osfi_positive_fraction": float(table_a["osfi_positive_fraction"]),
                "mean_target_specific_osfi": float(table_a["mean_target_specific_osfi"]),
                "target_specific_positive_fraction": float(table_a["target_specific_positive_fraction"]),
                "bootstrap_p_value": float(table_a["bootstrap_p_value_mean_leq_zero"]),
                "wilcoxon_or_sign_p_value": float(table_a["wilcoxon_or_sign_p_value"]),
                "mean_osfi_ci_low": float(table_a["mean_osfi_ci_low"]),
                "mean_osfi_ci_high": float(table_a["mean_osfi_ci_high"]),
                "random_mean_osfi": float(table_b.get("random_mean_osfi", float("nan"))),
                "label_random_mean_osfi": float(table_b.get("label_random_mean_osfi", float("nan"))),
                "group_structure_gain": float(table_b.get("group_structure_gain", float("nan"))),
                "semantic_group_gain": float(table_b.get("semantic_group_gain", float("nan"))),
                "source_impact_positive_fraction": float(table_c["source_impact_positive_fraction"]),
                "modality_impact_positive_fraction": float(table_c["modality_impact_positive_fraction"]),
                "mean_source_impact": float(table_c["mean_source_impact"]),
                "mean_modality_impact": float(table_c["mean_modality_impact"]),
            }
        )

    aggregate = {
        "num_datasets": int(len(rows)),
        "datasets": [row["dataset"] for row in rows],
        "mean_osfi_across_datasets": safe_mean(row["mean_osfi"] for row in rows),
        "mean_target_specific_osfi_across_datasets": safe_mean(row["mean_target_specific_osfi"] for row in rows),
        "mean_osfi_positive_fraction": safe_mean(row["osfi_positive_fraction"] for row in rows),
        "mean_group_structure_gain": safe_mean(row["group_structure_gain"] for row in rows),
        "mean_semantic_group_gain": safe_mean(row["semantic_group_gain"] for row in rows),
        "datasets_with_positive_group_structure_gain": [
            row["dataset"] for row in rows if row["group_structure_gain"] > 0.0
        ],
        "datasets_with_positive_semantic_group_gain": [
            row["dataset"] for row in rows if row["semantic_group_gain"] > 0.0
        ],
        "datasets_with_significant_positive_osfi": [
            row["dataset"] for row in rows if row["bootstrap_p_value"] < 0.05 and row["mean_osfi"] > 0.0
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
        "# Supplementary OSFI Across Datasets",
        "",
        f"- datasets: {', '.join(aggregate['datasets'])}",
        f"- mean OSFI across datasets: {aggregate['mean_osfi_across_datasets']:.6g}",
        f"- mean target-specific OSFI: {aggregate['mean_target_specific_osfi_across_datasets']:.6g}",
        f"- mean OSFI+ fraction: {aggregate['mean_osfi_positive_fraction']:.3f}",
        f"- mean group-structure gain: {aggregate['mean_group_structure_gain']:.6g}",
        f"- mean semantic-group gain: {aggregate['mean_semantic_group_gain']:.6g}",
        f"- significant positive OSFI datasets: {', '.join(aggregate['datasets_with_significant_positive_osfi'])}",
        f"- positive group-structure gain datasets: {', '.join(aggregate['datasets_with_positive_group_structure_gain']) or 'none'}",
        f"- positive semantic-group gain datasets: {', '.join(aggregate['datasets_with_positive_semantic_group_gain']) or 'none'}",
        "",
        "## Table A: Other-Sample Fisher Impact",
        "",
        "| Dataset | Mean OSFI | Median OSFI | OSFI+ | Mean TargetSpecificOSFI | p-value | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['mean_osfi']:.6g} | {row['median_osfi']:.6g} | "
            f"{row['osfi_positive_fraction']:.3f} | {row['mean_target_specific_osfi']:.6g} | "
            f"{row['bootstrap_p_value']:.4g} | "
            f"[{row['mean_osfi_ci_low']:.6g}, {row['mean_osfi_ci_high']:.6g}] |"
        )
    lines.extend(
        [
            "",
            "## Table B: Controls",
            "",
            "| Dataset | KMeans OSFI | Random OSFI | Label-Random OSFI | GroupStructureGain | SemanticGroupGain |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['mean_osfi']:.6g} | {row['random_mean_osfi']:.6g} | "
            f"{row['label_random_mean_osfi']:.6g} | {row['group_structure_gain']:.6g} | "
            f"{row['semantic_group_gain']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Table C: Source-Level Effect",
            "",
            "| Dataset | SourceImpact+ | ModalityImpact+ | Mean SourceImpact | Mean ModalityImpact |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['source_impact_positive_fraction']:.3f} | "
            f"{row['modality_impact_positive_fraction']:.3f} | "
            f"{row['mean_source_impact']:.6g} | {row['mean_modality_impact']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The existence claim is supported if Mean OSFI is positive, OSFI+ is high, and the bootstrap p-value is small.",
            "- TargetSpecificOSFI controls for generic PSD Fisher mass by subtracting random-subspace OSFI.",
            "- GroupStructureGain and SemanticGroupGain are stricter controls; negative values mean k-means grouping did not outperform random controls for that dataset.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
