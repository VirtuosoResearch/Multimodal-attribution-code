from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opt_motivation import default_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multiseed Fisher/Hessian results.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["imdb", "humor", "sarcasm"],
        choices=["imdb", "humor", "sarcasm"],
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=default_output_dir() / "fisher_hessian_multiseed_runs",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=default_output_dir() / "fisher_hessian_multiseed_summary.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=default_output_dir() / "fisher_hessian_multiseed_summary.md",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if number != number:
        return "NaN"
    return f"{number:.{digits}f}"


def _aggregate_dataset(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = [int(item["seed"]) for item in payloads]
    devices = [item["config"]["device"] for item in payloads]
    pair_names = list(payloads[0]["metrics"]["directional_pairs"].keys())
    target_names = list(payloads[0]["metrics"]["targets"].keys())

    pair_summary: dict[str, Any] = {}
    for pair_name in pair_names:
        pairs = [item["metrics"]["directional_pairs"][pair_name] for item in payloads]
        pair_summary[pair_name] = {
            "positive_transfer_count": int(sum(pair["local_transfer"]["decision"] == "positive_transfer" for pair in pairs)),
            "negative_transfer_count": int(sum(pair["local_transfer"]["decision"] == "negative_transfer" for pair in pairs)),
            "stable_positive_transfer": int(sum(pair["local_transfer"]["decision"] == "positive_transfer" for pair in pairs)) >= (len(pairs) // 2 + 1),
            "stable_negative_transfer": int(sum(pair["local_transfer"]["decision"] == "negative_transfer" for pair in pairs)) >= (len(pairs) // 2 + 1),
            "normalized_complementarity": _mean_std([pair["fisher"]["normalized_complementarity"] for pair in pairs]),
            "normalized_redundancy": _mean_std([pair["fisher"]["normalized_redundancy"] for pair in pairs]),
            "grad_cosine": _mean_std([pair["gradient"]["mean_cosine"] for pair in pairs]),
            "conflict_rate": _mean_std([pair["gradient"]["batch_conflict_rate"] for pair in pairs]),
            "q_yy": _mean_std([pair["hessian"]["q_yy_given_target"] for pair in pairs]),
            "q_yx": _mean_std([pair["hessian"]["q_yx_given_target"] for pair in pairs]),
            "b_opt": _mean_std([pair["local_transfer"]["benefit"] for pair in pairs]),
        }

    target_summary: dict[str, Any] = {}
    for target_name in target_names:
        targets = [item["metrics"]["targets"][target_name] for item in payloads]
        target_summary[target_name] = {
            "fisher_trace": _mean_std([target["fisher_target"]["trace"] for target in targets]),
            "fisher_lambda_max": _mean_std([target["fisher_target"]["lambda_max"] for target in targets]),
            "hessian_lambda_max": _mean_std([target["hessian_target"]["lambda_max"] for target in targets]),
            "hessian_trace": _mean_std([target["hessian_target"]["trace_hutchinson"]["mean"] for target in targets]),
            "fisher_hessian_overlap": _mean_std([target["hessian_target"]["fisher_hessian_subspace_overlap"] for target in targets]),
        }

    stable_positive_pairs = [name for name, item in pair_summary.items() if item["stable_positive_transfer"]]
    stable_negative_pairs = [name for name, item in pair_summary.items() if item["stable_negative_transfer"]]
    complementarity_dominant_pairs = [
        name
        for name, item in pair_summary.items()
        if item["normalized_complementarity"]["mean"] > item["normalized_redundancy"]["mean"]
    ]
    return {
        "dataset": payloads[0]["dataset"],
        "seeds": seeds,
        "devices": devices,
        "config": payloads[0]["config"],
        "training": payloads[0]["training"],
        "targets": target_summary,
        "directional_pairs": pair_summary,
        "aggregate": {
            "num_seeds": len(payloads),
            "stable_positive_pairs": stable_positive_pairs,
            "stable_negative_pairs": stable_negative_pairs,
            "complementarity_dominant_pairs": complementarity_dominant_pairs,
            "hypothesis": {
                "new_directions_exist": len(complementarity_dominant_pairs) > 0,
                "complementarity_often_exceeds_redundancy": len(complementarity_dominant_pairs) >= max(1, len(pair_summary) // 2),
                "both_positive_and_negative_transfer_exist": bool(stable_positive_pairs) and bool(stable_negative_pairs),
            },
        },
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = ["# Fisher Hessian Multiseed Summary", ""]
    for dataset in summary["datasets"]:
        item = summary["results"][dataset]
        agg = item["aggregate"]
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Seeds: {', '.join(str(seed) for seed in item['seeds'])}",
                f"- Devices: {', '.join(item['devices'])}",
                f"- Stable positive pairs: {', '.join(agg['stable_positive_pairs']) if agg['stable_positive_pairs'] else 'none'}",
                f"- Stable negative pairs: {', '.join(agg['stable_negative_pairs']) if agg['stable_negative_pairs'] else 'none'}",
                f"- Complementarity-dominant pairs: {', '.join(agg['complementarity_dominant_pairs']) if agg['complementarity_dominant_pairs'] else 'none'}",
                f"- Hypothesis verdicts: new_directions_exist={agg['hypothesis']['new_directions_exist']}, "
                f"complementarity_often_exceeds_redundancy={agg['hypothesis']['complementarity_often_exceeds_redundancy']}, "
                f"both_positive_and_negative_transfer_exist={agg['hypothesis']['both_positive_and_negative_transfer_exist']}",
                "",
                "### Directional Pairs",
            ]
        )
        for pair_name, pair in item["directional_pairs"].items():
            lines.append(
                f"- {pair_name}: pos={pair['positive_transfer_count']}/{agg['num_seeds']}, "
                f"neg={pair['negative_transfer_count']}/{agg['num_seeds']}, "
                f"C_norm={_fmt(pair['normalized_complementarity']['mean'])}±{_fmt(pair['normalized_complementarity']['std'])}, "
                f"R_norm={_fmt(pair['normalized_redundancy']['mean'])}±{_fmt(pair['normalized_redundancy']['std'])}, "
                f"grad_cos={_fmt(pair['grad_cosine']['mean'])}±{_fmt(pair['grad_cosine']['std'])}, "
                f"B_opt={_fmt(pair['b_opt']['mean'], 8)}±{_fmt(pair['b_opt']['std'], 8)}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results: dict[str, Any] = {}
    for dataset in args.datasets:
        paths = sorted(args.runs_dir.glob(f"{dataset}_fisher_hessian_seed*.json"))
        if not paths:
            raise FileNotFoundError(f"No multiseed runs found for {dataset} in {args.runs_dir}")
        payloads = [_load(path) for path in paths]
        results[dataset] = _aggregate_dataset(payloads)
    summary = {"datasets": args.datasets, "results": results}
    args.output_json.write_text(json.dumps(summary, indent=2))
    args.output_md.write_text(_markdown(summary))
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
