"""Filter source-pair examples by relative PID and FIC gaps."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fisher_geometry.pid_matched_source_pair_loss_eval import (
    add_pid_close_percentiles,
    build_normalizers,
    build_source_pair_rows,
    collect_source_info,
    gap_over_global_range,
)


DEFAULT_RESULT_DIR = (
    REPO_ROOT
    / "src"
    / "fisher_geometry"
    / "verification_results"
    / "real_motivating_example"
    / "pid_matched_group_finetune_base1"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter all source-source pairs by PID_gap/range and FIC_gap/range."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pid-threshold", type=float, default=0.05)
    parser.add_argument("--fic-threshold", type=float, default=0.1)
    parser.add_argument("--pid-profile-space", choices=["raw_zscore", "raw_minmax", "clipped_norm"], default="raw_zscore")
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    result_paths = sorted(args.result_dir.glob(f"*_seed{int(args.seed)}_pid_matched_group_finetune.json"))
    if not result_paths:
        raise FileNotFoundError(f"no source result files found in {args.result_dir}")

    dataset_outputs = []
    all_rows = []
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = payload["config"]["dataset"]
        source_info = collect_source_info(payload["records"])
        pairs = build_source_pair_rows(source_info, pid_profile_space=str(args.pid_profile_space))
        add_pid_close_percentiles(pairs)
        normalizers = build_normalizers(pairs, ["source_pair_pid_distance", "FIC_gap", "test_loss_gap"])
        filtered = []
        for row in pairs:
            test_loss_gap = row.get("test_loss_gap")
            if test_loss_gap is None:
                continue
            pid_gap_over_range = gap_over_global_range(row["source_pair_pid_distance"], normalizers, "source_pair_pid_distance")
            fic_gap_over_range = gap_over_global_range(row["FIC_gap"], normalizers, "FIC_gap")
            test_loss_gap_over_range = gap_over_global_range(test_loss_gap, normalizers, "test_loss_gap")
            if pid_gap_over_range < float(args.pid_threshold) and fic_gap_over_range > float(args.fic_threshold):
                record = {
                    "dataset": dataset,
                    "source_pair": f"{int(row['source_group_a'])}-{int(row['source_group_b'])}",
                    "source_group_a": int(row["source_group_a"]),
                    "source_group_b": int(row["source_group_b"]),
                    "PID_close_percentile": float(row["PID_close_percentile"]),
                    "PID_gap": float(row["source_pair_pid_distance"]),
                    "PID_gap_over_range": float(pid_gap_over_range),
                    "FIC_gap": float(row["FIC_gap"]),
                    "FIC_gap_over_range": float(fic_gap_over_range),
                    "test_loss_gap": float(test_loss_gap),
                    "test_loss_gap_over_range": float(test_loss_gap_over_range),
                }
                filtered.append(record)
                all_rows.append(record)
        filtered.sort(
            key=lambda row: (
                float(row["PID_gap_over_range"]),
                -float(row["FIC_gap_over_range"]),
                -float(row["test_loss_gap_over_range"]),
            )
        )
        dataset_outputs.append(
            {
                "dataset": dataset,
                "num_all_pairs": len(pairs),
                "num_filtered_pairs": len(filtered),
                "filtered_pairs": filtered,
                "summary": summarize(filtered),
            }
        )

    output = {
        "config": {
            "result_dir": str(args.result_dir),
            "seed": int(args.seed),
            "pid_profile_space": str(args.pid_profile_space),
            "filter": {
                "PID_gap_over_range_lt": float(args.pid_threshold),
                "FIC_gap_over_range_gt": float(args.fic_threshold),
            },
        },
        "overall_summary": summarize(all_rows),
        "datasets": dataset_outputs,
    }
    prefix = args.output_prefix or f"source_pair_filter_pidlt{args.pid_threshold:g}_ficgt{args.fic_threshold:g}"
    output_json = args.result_dir / f"{prefix}.json"
    output_md = args.result_dir / f"{prefix}.md"
    output_json.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    output_md.write_text(to_markdown(output), encoding="utf-8")
    print(to_markdown(output))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "PID_gap_over_range_mean": None,
            "FIC_gap_over_range_mean": None,
            "test_loss_gap_over_range_mean": None,
            "test_loss_gap_mean": None,
        }
    return {
        "count": len(rows),
        "PID_close_percentile_mean": mean(rows, "PID_close_percentile"),
        "PID_close_percentile_max": max_value(rows, "PID_close_percentile"),
        "PID_gap_over_range_mean": mean(rows, "PID_gap_over_range"),
        "PID_gap_over_range_max": max_value(rows, "PID_gap_over_range"),
        "FIC_gap_over_range_mean": mean(rows, "FIC_gap_over_range"),
        "FIC_gap_over_range_max": max_value(rows, "FIC_gap_over_range"),
        "test_loss_gap_over_range_mean": mean(rows, "test_loss_gap_over_range"),
        "test_loss_gap_over_range_max": max_value(rows, "test_loss_gap_over_range"),
        "test_loss_gap_mean": mean(rows, "test_loss_gap"),
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def max_value(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.max([float(row[key]) for row in rows]))


def to_markdown(output: dict[str, Any]) -> str:
    cfg = output["config"]
    lines = [
        "# Filtered PID-Close / High-FIC Source Pairs",
        "",
        f"- PID profile space: `{cfg['pid_profile_space']}`",
        f"- filter: `PID_gap/range < {cfg['filter']['PID_gap_over_range_lt']}` and `FIC_gap/range > {cfg['filter']['FIC_gap_over_range_gt']}`",
        "",
        "## Dataset Summary",
        "",
        "| dataset | all pairs | kept pairs | PID pct max | PID_gap/range mean | FIC_gap/range mean | test_loss_gap/range mean | raw test_loss_gap mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in output["datasets"]:
        summary = item["summary"]
        lines.append(
            "| {dataset} | {all_pairs} | {kept} | {pid_pct} | {pid_gap} | {fic_gap} | {loss_gap} | {raw_loss} |".format(
                dataset=item["dataset"],
                all_pairs=int(item["num_all_pairs"]),
                kept=int(item["num_filtered_pairs"]),
                pid_pct=fmt(summary.get("PID_close_percentile_max")),
                pid_gap=fmt(summary.get("PID_gap_over_range_mean")),
                fic_gap=fmt(summary.get("FIC_gap_over_range_mean")),
                loss_gap=fmt(summary.get("test_loss_gap_over_range_mean")),
                raw_loss=fmt(summary.get("test_loss_gap_mean")),
            )
        )

    lines.extend(
        [
            "",
            "## Filtered Pairs",
            "",
            "| dataset | pair | PID pct | PID_gap/range | FIC_gap/range | test_loss_gap/range | raw test_loss_gap |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in output["datasets"]:
        for row in item["filtered_pairs"]:
            lines.append(
                "| {dataset} | {pair} | {pid_pct} | {pid_gap} | {fic_gap} | {loss_gap} | {raw_loss} |".format(
                    dataset=row["dataset"],
                    pair=row["source_pair"],
                    pid_pct=fmt(row["PID_close_percentile"]),
                    pid_gap=fmt(row["PID_gap_over_range"]),
                    fic_gap=fmt(row["FIC_gap_over_range"]),
                    loss_gap=fmt(row["test_loss_gap_over_range"]),
                    raw_loss=fmt(row["test_loss_gap"]),
                )
            )
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "nan"
    return f"{numeric:.4g}"


if __name__ == "__main__":
    main()
