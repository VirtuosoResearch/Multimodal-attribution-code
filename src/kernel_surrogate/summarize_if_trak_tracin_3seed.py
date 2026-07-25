from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
ROOT = REPO_ROOT


DEFAULT_IF_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "influence_function_lds.json"
DEFAULT_TRAK_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "trak_group_lds.json"
DEFAULT_TRACIN_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "tracin_group_lds.json"
DEFAULT_OUTPUT_JSON = ROOT / "src" / "kernel_surrogate" / "results" / "if_trak_tracin_3seed_summary.json"
DEFAULT_OUTPUT_MD = ROOT / "src" / "kernel_surrogate" / "results" / "if_trak_tracin_3seed_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize IF, TRAK, and TracIn held-out test LDS over selected split seeds."
    )
    parser.add_argument("--if-json", type=Path, default=DEFAULT_IF_JSON)
    parser.add_argument("--trak-json", type=Path, default=DEFAULT_TRAK_JSON)
    parser.add_argument("--tracin-json", type=Path, default=DEFAULT_TRACIN_JSON)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def load_records(path: Path, *, family: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for row in payload.get("records", []):
        item = dict(row)
        item["family"] = family
        records.append(item)
    return records


def summarize(records: Sequence[dict[str, Any]], *, split_seeds: set[int]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in records:
        if int(row["split_seed"]) not in split_seeds:
            continue
        key = (str(row["family"]), str(row["dataset"]), str(row["setting"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (family, dataset, setting, method), rows in sorted(grouped.items()):
        lds = np.asarray([float(row["test_lds"]) for row in rows], dtype=np.float64)
        pearson = np.asarray([float(row["test_pearson"]) for row in rows], dtype=np.float64)
        rmses = np.asarray([float(row["test_rmse"]) for row in rows], dtype=np.float64)
        seeds = sorted(int(row["split_seed"]) for row in rows)
        summaries.append(
            {
                "family": family,
                "dataset": dataset,
                "setting": setting,
                "method": method,
                "split_seeds": seeds,
                "num_seeds": int(len(rows)),
                "mean_test_lds": float(np.nanmean(lds)),
                "std_test_lds": float(np.nanstd(lds, ddof=0)),
                "mean_pearson": float(np.nanmean(pearson)),
                "std_pearson": float(np.nanstd(pearson, ddof=0)),
                "mean_rmse": float(np.nanmean(rmses)),
                "std_rmse": float(np.nanstd(rmses, ddof=0)),
            }
        )
    return summaries


def to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# IF / TRAK / TracIn 3-Seed Held-out Test LDS Summary",
        "",
        f"- split seeds: `{payload['config']['split_seeds']}`",
        f"- seed type: `{payload['config']['seed_type']}`",
        f"- metric source: `{payload['config']['metric_source']}`",
        "",
        "| family | dataset | setting | method | mean test LDS | std test LDS | mean Pearson | std Pearson | mean RMSE | std RMSE |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['family']} | {row['dataset']} | {row['setting']} | {row['method']} | "
            f"{row['mean_test_lds']:.6f} | {row['std_test_lds']:.6f} | "
            f"{row['mean_pearson']:.6f} | {row['std_pearson']:.6f} | "
            f"{row['mean_rmse']:.6f} | {row['std_rmse']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    split_seeds = {int(seed) for seed in args.split_seeds}
    records = []
    records.extend(load_records(args.if_json, family="influence_function"))
    records.extend(load_records(args.trak_json, family="trak"))
    records.extend(load_records(args.tracin_json, family="tracin"))
    payload = {
        "config": {
            "split_seeds": sorted(split_seeds),
            "seed_type": "comparison split_seed; kmeans LDS artifacts in this workspace are available only for dataset seed 0",
            "metric_source": "record.test_lds / record.test_pearson / record.test_rmse, computed on held-out test_indices from split_indices(...)",
            "inputs": {
                "influence_function": str(args.if_json),
                "trak": str(args.trak_json),
                "tracin": str(args.tracin_json),
            },
        },
        "summary": summarize(records, split_seeds=split_seeds),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    args.output_md.write_text(to_markdown(payload), encoding="utf-8")
    print(to_markdown(payload), flush=True)
    print(f"Saved JSON to {args.output_json}", flush=True)
    print(f"Saved Markdown to {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
