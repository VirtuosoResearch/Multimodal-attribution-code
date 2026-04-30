"""Summarize real-data CMIS outputs across datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernel_surrogate.real_data_cmis import to_markdown


DEFAULT_DATASETS = ("avmnist", "humor", "imdb", "mosei", "mosi", "sarcasm")

METHOD_LABELS = {
    "metric_rbf": "metric_rbf_last_layer_ours",
    "structured_rbf": "structured_rbf_ours",
    "raw_flat_rbf": "raw_flat_rbf_baseline",
}

UPDATED_LIMITATIONS = [
    "This implementation is group-level because the available LDS artifacts are group-subset masks.",
    "Protocol A uses existing subset checkpoints selected by checkpoint group_ids; it is a group-level checkpoint intervention, not a per-sample attention-matrix edit.",
    "The humor cached features do not include raw text/image payloads in this loader, so Protocol C visualization is not emitted by this script.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("src/kernel_surrogate/results"))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.result_dir
    rows: list[dict] = []
    method_stats = {method: {"effects": [], "high": [], "random": [], "wins": 0} for method in METHOD_LABELS}
    lds_stats = {method: {"lds": [], "rmse": [], "train_lds": [], "train_rmse": []} for method in METHOD_LABELS}

    for dataset in args.datasets:
        path = root / f"real_data_cmis_{dataset}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["limitations"] = UPDATED_LIMITATIONS
        path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")
        (root / f"real_data_cmis_{dataset}.md").write_text(to_markdown(data), encoding="utf-8")

        best_method = None
        best_effect = float("-inf")
        by_method = {}
        for result in data.get("protocol_a_kernel_results", []):
            method = str(result.get("method"))
            if result.get("skipped") or method not in METHOD_LABELS:
                continue
            effect = float(result["effect_size_vs_random_loss_increases"])
            high = float(result["high_loss_increase_vs_full"])
            random = float(result["random_loss_increase_mean"])
            method_stats[method]["effects"].append(effect)
            method_stats[method]["high"].append(high)
            method_stats[method]["random"].append(random)
            by_method[method] = {
                "effect_size": effect,
                "high_loss_increase": high,
                "random_loss_increase": random,
            }
            if effect > best_effect:
                best_effect = effect
                best_method = method

        for result in data.get("kernel_comparison", []):
            method = str(result.get("method"))
            if method not in METHOD_LABELS:
                continue
            lds_stats[method]["lds"].append(float(result["test_spearman"]))
            lds_stats[method]["rmse"].append(float(result["test_rmse"]))
            lds_stats[method]["train_lds"].append(float(result["train_spearman"]))
            lds_stats[method]["train_rmse"].append(float(result["train_rmse"]))

        if best_method is not None:
            method_stats[best_method]["wins"] += 1

        rows.append(
            {
                "dataset": dataset,
                "num_groups": data["config"]["num_groups"],
                "full_loss": data["full_checkpoint_metrics"]["loss"],
                "full_acc": data["full_checkpoint_metrics"].get("acc"),
                "best_method": METHOD_LABELS.get(best_method, best_method),
                "best_effect_size": best_effect,
                "methods": by_method,
                "lds": {
                    METHOD_LABELS[str(result["method"])]: {
                        "heldout_lds": float(result["test_spearman"]),
                        "heldout_rmse": float(result["test_rmse"]),
                        "train_lds": float(result["train_spearman"]),
                        "train_rmse": float(result["train_rmse"]),
                    }
                    for result in data.get("kernel_comparison", [])
                    if str(result.get("method")) in METHOD_LABELS
                },
                "gradient_params": data["config"].get("gradient_params"),
                "metric_normalization": data["config"].get("metric_normalization"),
                "protocol_a_mode": data["config"].get("protocol_a_mode"),
            }
        )

    summary = {
        "config": {
            "datasets": list(args.datasets),
            "metric_rbf_gradient_params": "last_layer",
            "metric_normalization": "cosine",
            "protocol_a_mode": "checkpoint_group_id",
            "ours": ["metric_rbf_last_layer_ours", "structured_rbf_ours"],
            "baseline": ["raw_flat_rbf_baseline"],
        },
        "rows": rows,
        "method_summary": {},
        "lds_summary": {},
    }

    for method, stat in method_stats.items():
        n = len(stat["effects"])
        label = METHOD_LABELS[method]
        summary["method_summary"][label] = {
            "num_datasets": n,
            "wins": stat["wins"],
            "mean_effect_size": sum(stat["effects"]) / n if n else None,
            "mean_high_loss_increase": sum(stat["high"]) / n if n else None,
            "mean_random_loss_increase": sum(stat["random"]) / n if n else None,
        }
        lds = lds_stats[method]
        summary["lds_summary"][label] = {
            "num_datasets": len(lds["lds"]),
            "mean_heldout_lds": sum(lds["lds"]) / len(lds["lds"]) if lds["lds"] else None,
            "mean_heldout_rmse": sum(lds["rmse"]) / len(lds["rmse"]) if lds["rmse"] else None,
            "mean_train_lds": sum(lds["train_lds"]) / len(lds["train_lds"]) if lds["train_lds"] else None,
            "mean_train_rmse": sum(lds["train_rmse"]) / len(lds["train_rmse"]) if lds["train_rmse"] else None,
        }

    summary_json = root / "real_data_cmis_all_datasets_summary.json"
    summary_md = root / "real_data_cmis_all_datasets_summary.md"
    summary_json.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    summary_md.write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))


def render_markdown(summary: dict) -> str:
    lines = [
        "# Real-Data CMIS All-Dataset Summary",
        "",
        "- Protocol A: existing subset checkpoints selected by checkpoint group_ids",
        "- Metric RBF: last-layer gradients only, cosine-normalized metric",
        "- Ours: metric_rbf_last_layer, structured_rbf",
        "- Baseline: raw_flat_rbf",
        "",
        "## Method Summary",
        "",
        "| method | role | mean effect size | wins | mean high increase | mean random increase |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for label, stat in summary["method_summary"].items():
        role = "baseline" if label == "raw_flat_rbf_baseline" else "ours"
        lines.append(
            f"| {label} | {role} | {stat['mean_effect_size']:.6g} | "
            f"{stat['wins']}/{stat['num_datasets']} | {stat['mean_high_loss_increase']:.6g} | "
            f"{stat['mean_random_loss_increase']:.6g} |"
        )

    lines.extend(
        [
            "",
            "## LDS Score Summary",
            "",
            "| method | role | mean held-out LDS | mean held-out RMSE | mean train LDS | mean train RMSE |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, stat in summary["lds_summary"].items():
        role = "baseline" if label == "raw_flat_rbf_baseline" else "ours"
        lines.append(
            f"| {label} | {role} | {stat['mean_heldout_lds']:.6g} | "
            f"{stat['mean_heldout_rmse']:.6g} | {stat['mean_train_lds']:.6g} | "
            f"{stat['mean_train_rmse']:.6g} |"
        )

    lines.extend(
        [
            "",
            "## Dataset Results",
            "",
            "| dataset | best method | metric_rbf_last_layer effect | structured_rbf effect | raw_flat_rbf effect |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary["rows"]:
        methods = row["methods"]
        lines.append(
            f"| {row['dataset']} | {row['best_method']} | {methods['metric_rbf']['effect_size']:.6g} | "
            f"{methods['structured_rbf']['effect_size']:.6g} | {methods['raw_flat_rbf']['effect_size']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Dataset LDS Scores",
            "",
            "| dataset | metric_rbf_last_layer LDS | structured_rbf LDS | raw_flat_rbf LDS |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary["rows"]:
        lds = row["lds"]
        lines.append(
            f"| {row['dataset']} | {lds['metric_rbf_last_layer_ours']['heldout_lds']:.6g} | "
            f"{lds['structured_rbf_ours']['heldout_lds']:.6g} | "
            f"{lds['raw_flat_rbf_baseline']['heldout_lds']:.6g} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
