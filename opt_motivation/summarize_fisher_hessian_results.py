from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize fresh Fisher/Hessian experiment outputs.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["imdb", "humor", "sarcasm"],
        choices=["imdb", "humor", "sarcasm"],
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-json", type=Path, default=RESULTS_DIR / "fisher_hessian_summary.json")
    parser.add_argument("--output-md", type=Path, default=RESULTS_DIR / "fisher_hessian_summary.md")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if number != number:
        return "NaN"
    return f"{number:.{digits}f}"


def _dataset_summary(payload: dict[str, Any]) -> dict[str, Any]:
    pair_items = payload["metrics"]["directional_pairs"]
    target_items = payload["metrics"]["targets"]
    positive = [name for name, pair in pair_items.items() if pair["local_transfer"]["decision"] == "positive_transfer"]
    negative = [name for name, pair in pair_items.items() if pair["local_transfer"]["decision"] == "negative_transfer"]
    return {
        "dataset": payload["dataset"],
        "config": payload["config"],
        "training": payload["training"],
        "targets": target_items,
        "directional_pairs": pair_items,
        "aggregate": {
            "num_pairs": len(pair_items),
            "num_positive_transfer": len(positive),
            "num_negative_transfer": len(negative),
            "positive_pairs": positive,
            "negative_pairs": negative,
        },
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = ["# Fisher Hessian Summary", ""]
    for dataset in summary["datasets"]:
        item = summary["results"][dataset]
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Device: {item['config']['device']}",
                f"- Epochs: {item['config']['epochs']}",
                f"- Positive transfer pairs: {item['aggregate']['num_positive_transfer']} / {item['aggregate']['num_pairs']}",
                f"- Negative transfer pairs: {item['aggregate']['num_negative_transfer']} / {item['aggregate']['num_pairs']}",
                "",
                "### Targets",
            ]
        )
        for target_name, target in item["targets"].items():
            fisher_target = target["fisher_target"]
            hessian_target = target["hessian_target"]
            lines.append(
                f"- {target_name}: F-trace={_fmt(fisher_target['trace'])}, "
                f"F-lambda_max={_fmt(fisher_target['lambda_max'])}, "
                f"H-lambda_max={_fmt(hessian_target['lambda_max'])}, "
                f"H-trace={_fmt(hessian_target['trace_hutchinson']['mean'])}, "
                f"F/H overlap={_fmt(hessian_target['fisher_hessian_subspace_overlap'])}"
            )
        lines.extend(["", "### Directional Pairs"])
        for pair_name, pair in item["directional_pairs"].items():
            lines.append(
                f"- {pair_name}: C_norm={_fmt(pair['fisher']['normalized_complementarity'])}, "
                f"R_norm={_fmt(pair['fisher']['normalized_redundancy'])}, "
                f"grad_cos={_fmt(pair['gradient']['mean_cosine'])}, "
                f"conflict={_fmt(pair['gradient']['batch_conflict_rate'])}, "
                f"q_yy={_fmt(pair['hessian']['q_yy_given_target'])}, "
                f"q_yx={_fmt(pair['hessian']['q_yx_given_target'])}, "
                f"B_opt={_fmt(pair['local_transfer']['benefit'], 8)}, "
                f"decision={pair['local_transfer']['decision']}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results = {}
    for dataset in args.datasets:
        path = args.results_dir / f"{dataset}_fisher_hessian_hypothesis.json"
        results[dataset] = _dataset_summary(_load(path))
    summary = {"datasets": args.datasets, "results": results}
    args.output_json.write_text(json.dumps(summary, indent=2))
    args.output_md.write_text(_markdown(summary))
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
