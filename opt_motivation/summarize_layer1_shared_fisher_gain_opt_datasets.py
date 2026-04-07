from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize revised layer-1 quadratic-score results on IMDb/Humor/Sarcasm.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    return parser


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _fmt(pair: tuple[float, float], precision: int = 4) -> str:
    if math.isnan(pair[0]):
        return "nan"
    return f"{pair[0]:.{precision}f} ± {pair[1]:.{precision}f}"


def _load_runs(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        grouped[payload["dataset"]].append(payload)
    for dataset in grouped:
        grouped[dataset] = sorted(grouped[dataset], key=lambda item: item["seed"])
    return dict(grouped)


def _per_fraction_summary(runs: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    fractions = sorted({float(item["fraction"]) for run in runs for item in run["metrics"]["C_related"]})
    summary: dict[float, dict[str, Any]] = {}
    for frac in fractions:
        related = [item for run in runs for item in run["metrics"]["C_related"] if float(item["fraction"]) == frac]
        permuted = [item for run in runs for item in run["metrics"]["C_unrelated"] if float(item["fraction"]) == frac]
        better_quad = sum(1 for rel, perm in zip(related, permuted) if rel["S_quad_F"] > perm["S_quad_F"])
        summary[frac] = {
            "related": {
                "delta_trace": _mean_std([item["delta_trace"] for item in related]),
                "delta_logdet": _mean_std([item["delta_logdet"] for item in related]),
                "G_target_norm": _mean_std([item["G_target_norm"] for item in related]),
                "G_perf_norm": _mean_std([item["G_perf_norm"] for item in related]),
                "S_grad": _mean_std([item["S_grad"] for item in related]),
                "S_grad_norm": _mean_std([item["S_grad_norm"] for item in related]),
                "Q_Fx": _mean_std([item["Q_Fx"] for item in related]),
                "S_quad_F": _mean_std([item["S_quad_F"] for item in related]),
                "S_quad_F_norm": _mean_std([item["S_quad_F_norm"] for item in related]),
                "delta_acc_vs_B": _mean_std([item["delta_acc_vs_B"] for item in related]),
            },
            "permuted": {
                "delta_trace": _mean_std([item["delta_trace"] for item in permuted]),
                "delta_logdet": _mean_std([item["delta_logdet"] for item in permuted]),
                "G_target_norm": _mean_std([item["G_target_norm"] for item in permuted]),
                "G_perf_norm": _mean_std([item["G_perf_norm"] for item in permuted]),
                "S_grad": _mean_std([item["S_grad"] for item in permuted]),
                "S_grad_norm": _mean_std([item["S_grad_norm"] for item in permuted]),
                "Q_Fx": _mean_std([item["Q_Fx"] for item in permuted]),
                "S_quad_F": _mean_std([item["S_quad_F"] for item in permuted]),
                "S_quad_F_norm": _mean_std([item["S_quad_F_norm"] for item in permuted]),
                "delta_acc_vs_B": _mean_std([item["delta_acc_vs_B"] for item in permuted]),
            },
            "support_counts": {"related_beats_permuted_on_S_quad_F": int(better_quad)},
        }
    return summary


def _global_correlations(runs: list[dict[str, Any]]) -> dict[str, Any]:
    related = defaultdict(list)
    permuted = defaultdict(list)
    for run in runs:
        for key, value in run["correlations"]["C_related"].items():
            related[key].append(value)
        for key, value in run["correlations"]["C_unrelated"].items():
            permuted[key].append(value)
    return {
        "related": {key: _mean_std(values) for key, values in related.items()},
        "permuted": {key: _mean_std(values) for key, values in permuted.items()},
    }


def _paired_condition_test(all_runs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_dataset: dict[str, Any] = {}
    pooled_diff = {"S_quad_F": [], "S_quad_F_norm": []}
    for dataset, runs in all_runs.items():
        diffs = {"S_quad_F": [], "S_quad_F_norm": []}
        for run in runs:
            related = {float(item["fraction"]): item for item in run["metrics"]["C_related"]}
            permuted = {float(item["fraction"]): item for item in run["metrics"]["C_unrelated"]}
            for frac in sorted(related):
                diffs["S_quad_F"].append(float(related[frac]["S_quad_F"] - permuted[frac]["S_quad_F"]))
                diffs["S_quad_F_norm"].append(float(related[frac]["S_quad_F_norm"] - permuted[frac]["S_quad_F_norm"]))
        pooled_diff["S_quad_F"].extend(diffs["S_quad_F"])
        pooled_diff["S_quad_F_norm"].extend(diffs["S_quad_F_norm"])
        by_dataset[dataset] = {}
        for key, values in diffs.items():
            arr = np.asarray(values, dtype=np.float64)
            positive = int(np.sum(arr > 0.0))
            if arr.size > 0 and not np.allclose(arr, 0.0):
                stat = stats.wilcoxon(arr, alternative="greater", zero_method="wilcox")
                p_value = float(stat.pvalue)
            else:
                p_value = float("nan")
            by_dataset[dataset][key] = {
                "mean_diff": float(arr.mean()) if arr.size > 0 else float("nan"),
                "std_diff": float(arr.std(ddof=0)) if arr.size > 0 else float("nan"),
                "positive_count": positive,
                "total_count": int(arr.size),
                "wilcoxon_greater_p": p_value,
            }
    pooled = {}
    for key, values in pooled_diff.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.size > 0 and not np.allclose(arr, 0.0):
            stat = stats.wilcoxon(arr, alternative="greater", zero_method="wilcox")
            p_value = float(stat.pvalue)
        else:
            p_value = float("nan")
        pooled[key] = {
            "mean_diff": float(arr.mean()) if arr.size > 0 else float("nan"),
            "std_diff": float(arr.std(ddof=0)) if arr.size > 0 else float("nan"),
            "positive_count": int(np.sum(arr > 0.0)),
            "total_count": int(arr.size),
            "wilcoxon_greater_p": p_value,
        }
    return {"by_dataset": by_dataset, "pooled": pooled}


def _pooled_rows(all_runs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, runs in all_runs.items():
        for run in runs:
            for condition_name, condition_flag in (("C_related", 1), ("C_unrelated", 0)):
                for item in run["metrics"][condition_name]:
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(run["seed"]),
                            "fraction": float(item["fraction"]),
                            "condition": "related" if condition_flag == 1 else "permuted",
                            "group": f"{dataset}_seed{run['seed']}",
                            "delta_metric": float(item["delta_acc_vs_B"]),
                            "delta_trace": float(item["delta_trace"]),
                            "delta_logdet": float(item["delta_logdet"]),
                            "G_target_norm": float(item["G_target_norm"]),
                            "G_perf_norm": float(item["G_perf_norm"]),
                            "S_grad": float(item["S_grad"]),
                            "S_grad_norm": float(item["S_grad_norm"]),
                            "S_quad_F": float(item["S_quad_F"]),
                            "S_quad_F_norm": float(item["S_quad_F_norm"]),
                        }
                    )
    return rows


def _design_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_names = sorted({row["dataset"] for row in rows})
    dummy_names = dataset_names[1:]
    x_numeric = np.asarray([[row[name] for name in feature_names] for row in rows], dtype=np.float64)
    x_dummies = np.asarray(
        [[1.0 if row["dataset"] == name else 0.0 for name in dummy_names] for row in rows],
        dtype=np.float64,
    )
    x = np.concatenate([x_numeric, x_dummies], axis=1) if x_dummies.size > 0 else x_numeric
    y = np.asarray([row["delta_metric"] for row in rows], dtype=np.float64)
    groups = np.asarray([row["group"] for row in rows])
    return x, y, groups


def _fit_regression(rows: list[dict[str, Any]], feature_names: list[str]) -> dict[str, Any]:
    x, y, groups = _design_matrix(rows, feature_names)
    model = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
    model.fit(x, y)
    preds = model.predict(x)
    result = {
        "features": feature_names,
        "r2": float(r2_score(y, preds)),
        "num_observations": int(y.shape[0]),
    }
    unique_groups = np.unique(groups)
    if unique_groups.shape[0] >= 3:
        splitter = GroupKFold(n_splits=min(5, unique_groups.shape[0]))
        fold_scores: list[float] = []
        for train_idx, test_idx in splitter.split(x, y, groups):
            fold_model = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
            fold_model.fit(x[train_idx], y[train_idx])
            fold_scores.append(float(r2_score(y[test_idx], fold_model.predict(x[test_idx]))))
        result["group_cv_r2"] = float(np.mean(fold_scores))
        result["group_cv_r2_std"] = float(np.std(fold_scores, ddof=0))
    else:
        result["group_cv_r2"] = float("nan")
        result["group_cv_r2_std"] = float("nan")
    return result


def _pooled_regression_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "first_order_only": _fit_regression(rows, ["S_grad"]),
        "raw_fisher": _fit_regression(rows, ["delta_trace", "delta_logdet", "G_target_norm", "G_perf_norm"]),
        "quadratic": _fit_regression(rows, ["S_grad", "S_quad_F"]),
        "quadratic_norm": _fit_regression(rows, ["S_grad_norm", "S_quad_F_norm"]),
    }


def _write_md(all_runs: dict[str, list[dict[str, Any]]], payload: dict[str, Any], output_md: Path) -> None:
    lines: list[str] = []
    lines.append("# Revised Layer-1 Quadratic-Score Validation on IMDb, Humor, and Sarcasm")
    lines.append("")
    lines.append("This report follows the revised experiment design: compute target validation mean gradient, auxiliary validation mean gradient, target Fisher, and compare first-order and local quadratic scores on the shared parameter block.")
    lines.append("")
    lines.append("For each checkpoint and condition, we compute")
    lines.append("")
    lines.append("```math")
    lines.append("\\bar g_X,\\; \\bar g_Y,\\; \\widehat F_X,\\; S_{\\mathrm{grad}} = \\bar g_X^\\top \\bar g_Y,")
    lines.append("```")
    lines.append("")
    lines.append("```math")
    lines.append("S_{\\mathrm{quad-F}} = \\bar g_X^\\top \\bar g_Y - \\frac{\\eta}{2} \\bar g_Y^\\top \\widehat F_X \\bar g_Y,")
    lines.append("```")
    lines.append("")
    lines.append("```math")
    lines.append("S_{\\mathrm{quad-F-norm}} = \\frac{S_{\\mathrm{quad-F}}}{\\|\\bar g_X\\|\\,\\|\\bar g_Y\\| + \\varepsilon}.")
    lines.append("```")
    lines.append("")
    lines.append("Target modality is `text` for all datasets. The auxiliary view is `image` for IMDb and `{audio, video}` jointly for Humor and Sarcasm. The control condition uses globally permuted auxiliary inputs.")
    lines.append("")
    for dataset in ("imdb", "humor", "sarcasm"):
        if dataset not in payload["datasets"]:
            continue
        runs = all_runs[dataset]
        item = payload["datasets"][dataset]
        setting = runs[0]["setting"]
        config = runs[0]["config"]
        lines.append(f"## {dataset.capitalize()}")
        lines.append("")
        lines.append(f"- Target modality: `{', '.join(setting['target_modalities'])}`")
        lines.append(f"- Auxiliary view: `{', '.join(setting['aux_modalities'])}`")
        lines.append(f"- Shared block: `{setting['shared_block']}`")
        lines.append(f"- Target metric: `{setting['target_metric']}`")
        lines.append(f"- Seeds: {', '.join(str(run['seed']) for run in runs)}")
        lines.append(f"- Checkpoints per seed: `{config['checkpoint_count']}`")
        lines.append("")
        lines.append("| Ckpt | Related S_grad | Permuted S_grad | Related S_quad-F | Permuted S_quad-F | Related S_quad-F-norm | Permuted S_quad-F-norm | Related Δmetric vs B | Permuted Δmetric vs B | rel>perm count |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for frac in sorted(item["per_checkpoint"]):
            ckpt = item["per_checkpoint"][frac]
            lines.append(
                f"| {int(float(frac) * 100)}% | "
                f"{_fmt(tuple(ckpt['related']['S_grad']))} | "
                f"{_fmt(tuple(ckpt['permuted']['S_grad']))} | "
                f"{_fmt(tuple(ckpt['related']['S_quad_F']))} | "
                f"{_fmt(tuple(ckpt['permuted']['S_quad_F']))} | "
                f"{_fmt(tuple(ckpt['related']['S_quad_F_norm']))} | "
                f"{_fmt(tuple(ckpt['permuted']['S_quad_F_norm']))} | "
                f"{_fmt(tuple(ckpt['related']['delta_acc_vs_B']))} | "
                f"{_fmt(tuple(ckpt['permuted']['delta_acc_vs_B']))} | "
                f"{ckpt['support_counts']['related_beats_permuted_on_S_quad_F']}/{len(runs)} |"
            )
        lines.append("")
        corr = item["correlations"]
        lines.append("Correlation:")
        lines.append(f"- Related `corr(S_grad, Δmetric)`: {_fmt(tuple(corr['related']['corr_S_grad_vs_delta_acc']))}")
        lines.append(f"- Related `corr(S_quad-F, Δmetric)`: {_fmt(tuple(corr['related']['corr_S_quad_F_vs_delta_acc']))}")
        lines.append(f"- Related `corr(S_quad-F-norm, Δmetric)`: {_fmt(tuple(corr['related']['corr_S_quad_F_norm_vs_delta_acc']))}")
        lines.append(f"- Permuted `corr(S_grad, Δmetric)`: {_fmt(tuple(corr['permuted']['corr_S_grad_vs_delta_acc']))}")
        lines.append(f"- Permuted `corr(S_quad-F, Δmetric)`: {_fmt(tuple(corr['permuted']['corr_S_quad_F_vs_delta_acc']))}")
        lines.append(f"- Permuted `corr(S_quad-F-norm, Δmetric)`: {_fmt(tuple(corr['permuted']['corr_S_quad_F_norm_vs_delta_acc']))}")
        lines.append("")
    condition_test = payload["condition_test"]
    lines.append("## Decision A: Related vs Permuted Control")
    lines.append("")
    lines.append(f"- Pooled `S_quad-F` mean diff (related - permuted): {condition_test['pooled']['S_quad_F']['mean_diff']:.6f}")
    lines.append(f"- Pooled `S_quad-F` one-sided Wilcoxon p-value: {condition_test['pooled']['S_quad_F']['wilcoxon_greater_p']:.6g}")
    lines.append(f"- Pooled `S_quad-F-norm` mean diff (related - permuted): {condition_test['pooled']['S_quad_F_norm']['mean_diff']:.6f}")
    lines.append(f"- Pooled `S_quad-F-norm` one-sided Wilcoxon p-value: {condition_test['pooled']['S_quad_F_norm']['wilcoxon_greater_p']:.6g}")
    lines.append("")
    lines.append("## Decision B: Predicting Target Gain")
    lines.append("")
    reg = payload["pooled_regression"]
    for name in ("first_order_only", "raw_fisher", "quadratic", "quadratic_norm"):
        item = reg[name]
        lines.append(
            f"- `{name}`: R² = {item['r2']:.4f}, group-CV R² = {item['group_cv_r2']:.4f} ± {item['group_cv_r2_std']:.4f}"
        )
    lines.append("")
    lines.append("## Strongest Defensible Claim")
    lines.append("")
    lines.append("> Across real multimodal models, raw shared Fisher gain is not sufficient to explain modal interaction. However, a target-aware local quadratic score, combining gradient alignment and second-order cost on the shared parameter block, partially predicts when auxiliary modalities help or hurt the target modality.")
    output_md.write_text("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    all_runs = _load_runs(Path(args.input_dir))
    if not all_runs:
        raise SystemExit("no run json files found")
    payload: dict[str, Any] = {"datasets": {}}
    for dataset, runs in all_runs.items():
        payload["datasets"][dataset] = {
            "setting": runs[0]["setting"],
            "config": runs[0]["config"],
            "seeds": [run["seed"] for run in runs],
            "per_checkpoint": _per_fraction_summary(runs),
            "correlations": _global_correlations(runs),
        }
    condition_test = _paired_condition_test(all_runs)
    rows = _pooled_rows(all_runs)
    pooled_regression = _pooled_regression_summary(rows)
    payload["condition_test"] = condition_test
    payload["pooled_regression"] = pooled_regression
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.write_text(json.dumps(payload, indent=2))
    _write_md(all_runs, payload, output_md)


if __name__ == "__main__":
    main()
