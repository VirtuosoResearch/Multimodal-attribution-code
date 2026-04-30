from __future__ import annotations

import argparse
import json
import math
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


DEFAULT_INPUT_DIR = ROOT / "src" / "fisher_geometry" / "verification_results" / "fisher_gpt_all_datasets"
DEFAULT_OUTPUT_JSON = DEFAULT_INPUT_DIR / "summary_all_datasets.json"
DEFAULT_OUTPUT_MD = DEFAULT_INPUT_DIR / "summary_all_datasets.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fisher_test_gpt metrics across datasets.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*_seed0_fisher_gpt_metrics_cuda1.json"))
    if not paths:
        raise FileNotFoundError(f"no dataset result JSONs found under {args.input_dir}")

    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload["config"]
        summary = payload["summary"]
        metrics = summary["metric_summary"]
        corr = summary["lds_proxy_correlations"]
        rows.append(
            {
                "dataset": str(cfg["dataset"]),
                "path": str(path),
                "checkpoint": str(cfg["full_checkpoint"]),
                "device": str(cfg["device"]),
                "num_parameters": int(cfg["num_parameters"]),
                "num_groups": int(cfg["num_groups"]),
                "fvc_mean": float(metrics["fvc_mean"]),
                "fvc_positive_fraction": float(metrics["fvc_positive_fraction"]),
                "nih_mean": float(metrics["nih_mean"]),
                "nih_positive_fraction": float(metrics["nih_positive_fraction"]),
                "nalign_mean": float(metrics["nalign_mean"]),
                "f_csmi_mean": float(metrics["f_csmi_mean"]),
                "f_csmi_positive_fraction": float(metrics["f_csmi_positive_fraction"]),
                "mmex_mean": float(metrics["mmex_mean"]),
                "mmex_positive_fraction": float(metrics["mmex_positive_fraction"]),
                "lds_fit_spearman": float(corr["lds_fit"]["fit_spearman"]),
                "fvc_vs_lds_spearman": float(corr["fvc_vs_source_lds_spearman"]),
                "nih_vs_lds_spearman": float(corr["nih_vs_source_lds_spearman"]),
                "nalign_vs_lds_spearman": float(corr["nalign_vs_source_lds_spearman"]),
                "f_csmi_vs_lds_spearman": float(corr["f_csmi_vs_source_lds_spearman"]),
                "top_fvc": summary["top_fvc"][:5],
                "top_f_csmi": summary["top_f_csmi"][:5],
            }
        )

    aggregate = {
        "num_datasets": int(len(rows)),
        "datasets": [row["dataset"] for row in rows],
        "mean_fvc_positive_fraction": safe_mean(row["fvc_positive_fraction"] for row in rows),
        "mean_nih_positive_fraction": safe_mean(row["nih_positive_fraction"] for row in rows),
        "mean_f_csmi_positive_fraction": safe_mean(row["f_csmi_positive_fraction"] for row in rows),
        "mean_mmex_positive_fraction": safe_mean(row["mmex_positive_fraction"] for row in rows),
        "mean_fvc_vs_lds_spearman": safe_mean(row["fvc_vs_lds_spearman"] for row in rows),
        "mean_nih_vs_lds_spearman": safe_mean(row["nih_vs_lds_spearman"] for row in rows),
        "mean_f_csmi_vs_lds_spearman": safe_mean(row["f_csmi_vs_lds_spearman"] for row in rows),
        "best_fvc_vs_lds_dataset": best_by(rows, "fvc_vs_lds_spearman"),
        "best_nih_vs_lds_dataset": best_by(rows, "nih_vs_lds_spearman"),
        "best_f_csmi_vs_lds_dataset": best_by(rows, "f_csmi_vs_lds_spearman"),
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


def best_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    finite = [row for row in rows if math.isfinite(float(row[key]))]
    if not finite:
        return None
    row = max(finite, key=lambda item: float(item[key]))
    return {"dataset": row["dataset"], key: row[key]}


def to_markdown(output: dict[str, Any]) -> str:
    aggregate = output["aggregate"]
    rows = output["datasets"]
    lines = [
        "# Fisher GPT Metrics Across Datasets",
        "",
        f"- datasets: {', '.join(aggregate['datasets'])}",
        f"- mean FVC positive fraction: {aggregate['mean_fvc_positive_fraction']:.3f}",
        f"- mean NIH positive fraction: {aggregate['mean_nih_positive_fraction']:.3f}",
        f"- mean F-CSMI positive fraction: {aggregate['mean_f_csmi_positive_fraction']:.3f}",
        f"- mean MMEx positive fraction: {aggregate['mean_mmex_positive_fraction']:.3f}",
        f"- mean FVC vs LDS Spearman: {aggregate['mean_fvc_vs_lds_spearman']:.3f}",
        f"- mean NIH vs LDS Spearman: {aggregate['mean_nih_vs_lds_spearman']:.3f}",
        f"- mean F-CSMI vs LDS Spearman: {aggregate['mean_f_csmi_vs_lds_spearman']:.3f}",
        "",
        "## Dataset Table",
        "",
        "| dataset | FVC+ | NIH+ | F-CSMI+ | MMEx+ | FVC~LDS rho | NIH~LDS rho | F-CSMI~LDS rho |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['fvc_positive_fraction']:.3f} | "
            f"{row['nih_positive_fraction']:.3f} | {row['f_csmi_positive_fraction']:.3f} | "
            f"{row['mmex_positive_fraction']:.3f} | {row['fvc_vs_lds_spearman']:.3f} | "
            f"{row['nih_vs_lds_spearman']:.3f} | {row['f_csmi_vs_lds_spearman']:.3f} |"
        )
    lines.extend(["", "## Per-Dataset Top FVC", ""])
    for row in rows:
        lines.append(f"### {row['dataset']}")
        lines.append("")
        lines.append("| target | source | modality | FVC | NIH | nAlign | LDS proxy |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for item in row["top_fvc"]:
            lines.append(
                f"| {item['target_group']} | {item['source_group']} | {item['source_modality']} | "
                f"{item['fvc']:.6g} | {item['nih']:.6g} | {item['nalign']:.6g} | "
                f"{item['source_lds_modality_loss_effect']:.6g} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
