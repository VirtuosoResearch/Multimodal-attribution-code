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
    parser = argparse.ArgumentParser(description="Posthoc curvature sensitivity analysis for layer-1 quadratic scores.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    return parser


def _fmt(x: float, precision: int = 4) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan"
    return f"{x:.{precision}f}"


def _load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        dataset = payload["dataset"]
        seed = int(payload["seed"])
        lr = float(payload["config"]["lr"])
        for condition_name, condition_label in (("C_related", "related"), ("C_unrelated", "permuted")):
            for item in payload["metrics"][condition_name]:
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "fraction": float(item["fraction"]),
                        "condition": condition_label,
                        "group": f"{dataset}_seed{seed}",
                        "delta_metric": float(item["delta_acc_vs_B"]),
                        "delta_trace": float(item["delta_trace"]),
                        "delta_logdet": float(item["delta_logdet"]),
                        "G_target_norm": float(item["G_target_norm"]),
                        "G_perf_norm": float(item["G_perf_norm"]),
                        "S_grad": float(item["S_grad"]),
                        "S_grad_norm": float(item["S_grad_norm"]),
                        "Q_Fx": float(item["Q_Fx"]),
                        "gbar_x_norm": float(item["gbar_x_norm"]),
                        "gbar_y_norm": float(item["gbar_y_norm"]),
                        "base_lr": lr,
                    }
                )
    return rows


def _paired_rows(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    keyed: dict[tuple[str, int, float, str], dict[str, Any]] = {}
    for row in rows:
        keyed[(row["dataset"], row["seed"], row["fraction"], row["condition"])] = row
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for dataset, seed, fraction, _ in sorted(keyed):
        key_rel = (dataset, seed, fraction, "related")
        key_perm = (dataset, seed, fraction, "permuted")
        if key_rel in keyed and key_perm in keyed:
            pairs.append((keyed[key_rel], keyed[key_perm]))
    return pairs


def _base_curvature_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        c2 = 0.5 * row["base_lr"] * row["Q_Fx"]
        ratio = c2 / max(abs(row["S_grad"]), 1e-12)
        eta_crit = 2.0 * abs(row["S_grad"]) / max(row["Q_Fx"], 1e-12)
        prefix = f"{row['condition']}_"
        by_dataset[row["dataset"]][prefix + "C2"].append(c2)
        by_dataset[row["dataset"]][prefix + "ratio"].append(ratio)
        by_dataset[row["dataset"]][prefix + "eta_crit"].append(eta_crit)
    summary: dict[str, Any] = {}
    for dataset, stats_map in by_dataset.items():
        summary[dataset] = {}
        for key, values in stats_map.items():
            arr = np.asarray(values, dtype=np.float64)
            summary[dataset][key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "median": float(np.median(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
    return summary


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
    result = {"features": feature_names, "r2": float(r2_score(y, preds))}
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


def _eta_sweep(rows: list[dict[str, Any]], eta_grid: list[float]) -> dict[str, Any]:
    pairs = _paired_rows(rows)
    baseline_diffs = np.asarray([rel["S_grad"] - perm["S_grad"] for rel, perm in pairs], dtype=np.float64)
    results = []
    for eta in eta_grid:
        augmented_rows: list[dict[str, Any]] = []
        related_minus_permuted: list[float] = []
        related_minus_permuted_norm: list[float] = []
        flip_count = 0
        for rel, perm in pairs:
            rel_c2 = 0.5 * eta * rel["Q_Fx"]
            perm_c2 = 0.5 * eta * perm["Q_Fx"]
            rel_quad = rel["S_grad"] - rel_c2
            perm_quad = perm["S_grad"] - perm_c2
            rel_denom = max(rel["gbar_x_norm"] * rel["gbar_y_norm"], 1e-12)
            perm_denom = max(perm["gbar_x_norm"] * perm["gbar_y_norm"], 1e-12)
            rel_quad_norm = rel_quad / rel_denom
            perm_quad_norm = perm_quad / perm_denom
            diff = rel_quad - perm_quad
            diff_norm = rel_quad_norm - perm_quad_norm
            related_minus_permuted.append(diff)
            related_minus_permuted_norm.append(diff_norm)
            baseline = rel["S_grad"] - perm["S_grad"]
            if np.sign(diff) != np.sign(baseline):
                flip_count += 1
            rel_row = dict(rel)
            rel_row["C_2nd"] = rel_c2
            rel_row["S_quad_eta"] = rel_quad
            rel_row["S_quad_eta_norm"] = rel_quad_norm
            perm_row = dict(perm)
            perm_row["C_2nd"] = perm_c2
            perm_row["S_quad_eta"] = perm_quad
            perm_row["S_quad_eta_norm"] = perm_quad_norm
            augmented_rows.extend([rel_row, perm_row])
        diff_arr = np.asarray(related_minus_permuted, dtype=np.float64)
        diff_norm_arr = np.asarray(related_minus_permuted_norm, dtype=np.float64)
        stat = stats.wilcoxon(diff_arr, alternative="greater", zero_method="wilcox") if not np.allclose(diff_arr, 0.0) else None
        stat_norm = stats.wilcoxon(diff_norm_arr, alternative="greater", zero_method="wilcox") if not np.allclose(diff_norm_arr, 0.0) else None
        regression = {
            "first_order_only": _fit_regression(augmented_rows, ["S_grad"]),
            "raw_fisher": _fit_regression(augmented_rows, ["delta_trace", "delta_logdet", "G_target_norm", "G_perf_norm"]),
            "quadratic_cost": _fit_regression(augmented_rows, ["S_grad", "C_2nd"]),
            "quadratic_score": _fit_regression(augmented_rows, ["S_quad_eta"]),
            "quadratic_score_norm": _fit_regression(augmented_rows, ["S_quad_eta_norm"]),
        }
        results.append(
            {
                "eta": eta,
                "mean_diff_S_quad": float(diff_arr.mean()),
                "mean_diff_S_quad_norm": float(diff_norm_arr.mean()),
                "positive_count": int(np.sum(diff_arr > 0.0)),
                "total_count": int(diff_arr.size),
                "flip_rate_vs_S_grad": float(flip_count / max(diff_arr.size, 1)),
                "wilcoxon_greater_p": float(stat.pvalue) if stat is not None else float("nan"),
                "wilcoxon_norm_greater_p": float(stat_norm.pvalue) if stat_norm is not None else float("nan"),
                "regression": regression,
            }
        )
    return {"eta_grid": results}


def _write_md(base_summary: dict[str, Any], sweep: dict[str, Any], output_md: Path) -> None:
    lines: list[str] = []
    lines.append("# Curvature Sensitivity Analysis for Layer-1 Quadratic Scores")
    lines.append("")
    lines.append("This report isolates the explicit second-order cost")
    lines.append("")
    lines.append("```math")
    lines.append("C_{2\\mathrm{nd}} = \\frac{\\eta}{2}\\,\\bar g_Y^\\top \\widehat F_X \\bar g_Y")
    lines.append("```")
    lines.append("")
    lines.append("and studies its magnitude, ratio to the first-order term, and eta-sensitivity.")
    lines.append("")
    lines.append("## Base-LR Magnitude")
    lines.append("")
    lines.append("| Dataset | Cond | mean C2nd | median eta_crit | mean C2nd/|S_grad| |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for dataset in ("imdb", "humor", "sarcasm"):
        if dataset not in base_summary:
            continue
        for cond in ("related", "permuted"):
            item = base_summary[dataset]
            lines.append(
                f"| {dataset} | {cond} | "
                f"{_fmt(item[f'{cond}_C2']['mean'])} | "
                f"{_fmt(item[f'{cond}_eta_crit']['median'])} | "
                f"{_fmt(item[f'{cond}_ratio']['mean'])} |"
            )
    lines.append("")
    lines.append("`eta_crit` is the scale at which the second-order term matches the first-order term in magnitude for a checkpoint-condition pair.")
    lines.append("")
    lines.append("## Eta Sweep")
    lines.append("")
    lines.append("| eta | mean(rel-perm) S_quad | p-value | flip rate | first-order CV R2 | raw-Fisher CV R2 | quad-cost CV R2 | quad-score CV R2 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in sweep["eta_grid"]:
        lines.append(
            f"| {_fmt(item['eta'], 6)} | "
            f"{_fmt(item['mean_diff_S_quad'])} | "
            f"{_fmt(item['wilcoxon_greater_p'], 6)} | "
            f"{_fmt(item['flip_rate_vs_S_grad'])} | "
            f"{_fmt(item['regression']['first_order_only']['group_cv_r2'])} | "
            f"{_fmt(item['regression']['raw_fisher']['group_cv_r2'])} | "
            f"{_fmt(item['regression']['quadratic_cost']['group_cv_r2'])} | "
            f"{_fmt(item['regression']['quadratic_score']['group_cv_r2'])} |"
        )
    lines.append("")
    best = max(sweep["eta_grid"], key=lambda item: item["regression"]["quadratic_cost"]["group_cv_r2"])
    lines.append("## Best Eta for Explicit Curvature Cost")
    lines.append("")
    lines.append(f"- best `eta` by `quadratic_cost` group-CV R²: `{best['eta']}`")
    lines.append(f"- `quadratic_cost` group-CV R²: `{best['regression']['quadratic_cost']['group_cv_r2']:.4f}`")
    lines.append(f"- `first_order_only` group-CV R²: `{best['regression']['first_order_only']['group_cv_r2']:.4f}`")
    lines.append(f"- `raw_fisher` group-CV R²: `{best['regression']['raw_fisher']['group_cv_r2']:.4f}`")
    lines.append("")
    lines.append("## Safe Takeaway")
    lines.append("")
    lines.append("> Target-aware local geometry helps explain modal interaction; the explicit second-order cost is suggestive, and becomes more visible once the curvature term is amplified, but it is still not fully isolated at the original training step size.")
    output_md.write_text("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    rows = _load_rows(Path(args.input_dir))
    if not rows:
        raise SystemExit("no run json files found")
    base_summary = _base_curvature_summary(rows)
    eta_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
    sweep = _eta_sweep(rows, eta_grid)
    payload = {"base_lr_summary": base_summary, "eta_sweep": sweep}
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.write_text(json.dumps(payload, indent=2))
    _write_md(base_summary, sweep, output_md)


if __name__ == "__main__":
    main()
