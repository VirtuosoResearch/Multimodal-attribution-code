from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from motivation.core import write_json
from opt_motivation import default_output_dir


def parse_args() -> argparse.Namespace:
    output_dir = default_output_dir()
    parser = argparse.ArgumentParser(description="Sample-level Fisher-vs-PID analysis for IMDb multiseed runs.")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=str(output_dir / "multiseed_runs"),
        help="Directory containing per-seed IMDb JSON outputs.",
    )
    parser.add_argument(
        "--imdb-path",
        type=str,
        default="/data/zhenshuo/datasets/multimodal_imdb.hdf5",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(output_dir / "imdb_sample_level_analysis.json"),
    )
    parser.add_argument(
        "--scatter-path",
        type=str,
        default=str(output_dir / "figures" / "imdb_sample_level_scatter.png"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--neighborhood-k",
        type=int,
        default=25,
    )
    return parser.parse_args()


def _mean_std(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }


def _two_group_test(left: np.ndarray, right: np.ndarray) -> dict[str, float] | None:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.size < 3 or right.size < 3:
        return None
    test = stats.mannwhitneyu(left, right, alternative="two-sided")
    return {"u_statistic": float(test.statistic), "p_value": float(test.pvalue)}


def _zscore_columns(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (features - mean) / std


def _compute_pid_neighborhoods(pid_features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    pid_z = _zscore_columns(np.asarray(pid_features, dtype=np.float32))
    model = NearestNeighbors(n_neighbors=min(k + 1, pid_z.shape[0]), metric="euclidean")
    model.fit(pid_z)
    distances, indices = model.kneighbors(pid_z)
    return distances[:, 1:], indices[:, 1:]


def _quartile_contrast(values: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    q1 = np.quantile(values, 0.25)
    q3 = np.quantile(values, 0.75)
    low = targets[values <= q1]
    high = targets[values >= q3]
    result: dict[str, Any] = {
        "low_count": int(low.shape[0]),
        "high_count": int(high.shape[0]),
        "low_mean": float(low.mean()) if low.size > 0 else float("nan"),
        "high_mean": float(high.mean()) if high.size > 0 else float("nan"),
        "delta": float(high.mean() - low.mean()) if low.size > 0 and high.size > 0 else float("nan"),
    }
    test = _two_group_test(high, low)
    result["mannwhitney"] = test
    return result


def _signed_fraction_test(deltas: np.ndarray) -> dict[str, Any]:
    deltas = np.asarray(deltas, dtype=np.float32)
    nonzero = deltas[np.abs(deltas) > 1e-8]
    if nonzero.size == 0:
        return {
            "num_nonzero": 0,
            "positive_fraction": float("nan"),
            "binomtest_p_value": None,
        }
    positive = int((nonzero > 0).sum())
    total = int(nonzero.shape[0])
    test = stats.binomtest(positive, total, p=0.5, alternative="greater")
    return {
        "num_nonzero": total,
        "positive_fraction": float(positive / total),
        "binomtest_p_value": float(test.pvalue),
    }


def _safe_wilcoxon(values: np.ndarray) -> dict[str, float] | None:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.abs(values) > 1e-8]
    if values.size < 10:
        return None
    test = stats.wilcoxon(values, alternative="greater", zero_method="wilcox")
    return {"statistic": float(test.statistic), "p_value": float(test.pvalue)}


def _compute_neighborhood_residual_analysis(
    *,
    pid_features: np.ndarray,
    fisher: np.ndarray,
    correctness: np.ndarray,
    neg_adv: np.ndarray,
    neighborhood_k: int,
) -> dict[str, Any]:
    distances, indices = _compute_pid_neighborhoods(pid_features, neighborhood_k)
    local_correctness = correctness[indices].mean(axis=1)
    local_neg_adv = neg_adv[indices].mean(axis=1)
    correctness_residual = correctness - local_correctness
    neg_adv_residual = neg_adv - local_neg_adv
    fisher_rank = (fisher[:, None] > fisher[indices]).mean(axis=1).astype(np.float32)
    mean_pid_distance = distances.mean(axis=1).astype(np.float32)

    corr_correctness = stats.spearmanr(fisher_rank, correctness_residual)
    corr_neg_adv = stats.spearmanr(fisher_rank, neg_adv_residual)
    corr_raw_fisher_correctness = stats.spearmanr(fisher, correctness_residual)
    corr_raw_fisher_neg_adv = stats.spearmanr(fisher, neg_adv_residual)

    return {
        "neighborhood_k": neighborhood_k,
        "pid_distance_summary": _mean_std(mean_pid_distance),
        "fisher_rank_vs_residual": {
            "correctness": {"rho": float(corr_correctness.statistic), "p_value": float(corr_correctness.pvalue)},
            "neg_adv": {"rho": float(corr_neg_adv.statistic), "p_value": float(corr_neg_adv.pvalue)},
        },
        "raw_fisher_vs_residual": {
            "correctness": {"rho": float(corr_raw_fisher_correctness.statistic), "p_value": float(corr_raw_fisher_correctness.pvalue)},
            "neg_adv": {"rho": float(corr_raw_fisher_neg_adv.statistic), "p_value": float(corr_raw_fisher_neg_adv.pvalue)},
        },
        "quartile_contrast": {
            "correctness_residual_by_fisher_rank": _quartile_contrast(fisher_rank, correctness_residual),
            "neg_adv_residual_by_fisher_rank": _quartile_contrast(fisher_rank, neg_adv_residual),
        },
        "sample_features": {
            "fisher_rank": fisher_rank.astype(float).tolist(),
            "mean_pid_distance": mean_pid_distance.astype(float).tolist(),
            "correctness_residual": correctness_residual.astype(float).tolist(),
            "neg_adv_residual": neg_adv_residual.astype(float).tolist(),
        },
    }


def _compute_local_fisher_stratification_analysis(
    *,
    pid_features: np.ndarray,
    fisher: np.ndarray,
    correctness: np.ndarray,
    neg_adv: np.ndarray,
    imdb_ids: np.ndarray,
    neighborhood_k: int,
    top_k: int,
) -> dict[str, Any]:
    distances, indices = _compute_pid_neighborhoods(pid_features, neighborhood_k)
    correctness_deltas: list[float] = []
    neg_adv_deltas: list[float] = []
    fisher_spreads: list[float] = []
    anchor_rows: list[dict[str, Any]] = []

    for anchor in range(indices.shape[0]):
        neighbor_idx = indices[anchor]
        neighbor_fisher = fisher[neighbor_idx]
        q1 = np.quantile(neighbor_fisher, 0.25)
        q3 = np.quantile(neighbor_fisher, 0.75)
        low_mask = neighbor_fisher <= q1
        high_mask = neighbor_fisher >= q3
        if int(low_mask.sum()) < 2 or int(high_mask.sum()) < 2:
            continue

        high_correctness = float(correctness[neighbor_idx[high_mask]].mean())
        low_correctness = float(correctness[neighbor_idx[low_mask]].mean())
        high_neg_adv = float(neg_adv[neighbor_idx[high_mask]].mean())
        low_neg_adv = float(neg_adv[neighbor_idx[low_mask]].mean())
        fisher_spread = float(q3 - q1)
        correctness_delta = high_correctness - low_correctness
        neg_adv_delta = high_neg_adv - low_neg_adv

        correctness_deltas.append(correctness_delta)
        neg_adv_deltas.append(neg_adv_delta)
        fisher_spreads.append(fisher_spread)
        anchor_rows.append(
            {
                "anchor_idx": int(anchor),
                "anchor_imdb_id": int(imdb_ids[anchor]),
                "anchor_fisher": float(fisher[anchor]),
                "anchor_correctness": float(correctness[anchor]),
                "anchor_neg_adv": float(neg_adv[anchor]),
                "mean_pid_distance": float(distances[anchor].mean()),
                "fisher_spread": fisher_spread,
                "high_fisher_mean": float(neighbor_fisher[high_mask].mean()),
                "low_fisher_mean": float(neighbor_fisher[low_mask].mean()),
                "correctness_delta": float(correctness_delta),
                "neg_adv_delta": float(neg_adv_delta),
            }
        )

    correctness_deltas_arr = np.asarray(correctness_deltas, dtype=np.float32)
    neg_adv_deltas_arr = np.asarray(neg_adv_deltas, dtype=np.float32)
    fisher_spreads_arr = np.asarray(fisher_spreads, dtype=np.float32)
    spread_threshold = float(np.quantile(fisher_spreads_arr, 0.75)) if fisher_spreads_arr.size > 0 else float("nan")
    high_spread_mask = fisher_spreads_arr >= spread_threshold if fisher_spreads_arr.size > 0 else np.zeros(0, dtype=bool)

    def _summary(values: np.ndarray) -> dict[str, Any]:
        summary = _mean_std(values)
        summary["median"] = float(np.median(values)) if values.size > 0 else float("nan")
        summary["positive_count"] = int((values > 0).sum()) if values.size > 0 else 0
        summary["positive_fraction"] = float((values > 0).mean()) if values.size > 0 else float("nan")
        summary["signed_fraction_test"] = _signed_fraction_test(values)
        summary["wilcoxon_greater_than_zero"] = _safe_wilcoxon(values)
        return summary

    ranked_rows = sorted(
        anchor_rows,
        key=lambda row: (row["correctness_delta"], row["fisher_spread"], -row["mean_pid_distance"]),
        reverse=True,
    )
    positive_high_spread_rows = [
        row
        for row, use_row in zip(anchor_rows, high_spread_mask.tolist(), strict=False)
        if use_row and row["correctness_delta"] > 0.0
    ]
    positive_high_spread_rows.sort(
        key=lambda row: (row["correctness_delta"], row["fisher_spread"], -row["mean_pid_distance"]),
        reverse=True,
    )

    return {
        "neighborhood_k": neighborhood_k,
        "num_anchors": int(len(anchor_rows)),
        "fisher_spread_summary": _mean_std(fisher_spreads_arr),
        "correctness_delta": _summary(correctness_deltas_arr),
        "neg_adv_delta": _summary(neg_adv_deltas_arr),
        "high_spread_subset": {
            "threshold": spread_threshold,
            "count": int(high_spread_mask.sum()),
            "correctness_delta": _summary(correctness_deltas_arr[high_spread_mask]) if fisher_spreads_arr.size > 0 else None,
            "neg_adv_delta": _summary(neg_adv_deltas_arr[high_spread_mask]) if fisher_spreads_arr.size > 0 else None,
        },
        "top_positive_correctness_anchors": ranked_rows[:top_k],
        "top_positive_correctness_high_spread_anchors": positive_high_spread_rows[:top_k],
    }


def _compute_pair_concordance_analysis(
    *,
    pid_features: np.ndarray,
    fisher: np.ndarray,
    correctness: np.ndarray,
    neg_adv: np.ndarray,
    imdb_ids: np.ndarray,
    neighborhood_k: int,
    top_k: int,
) -> dict[str, Any]:
    distances, indices = _compute_pid_neighborhoods(pid_features, neighborhood_k)
    pid_z = _zscore_columns(pid_features)
    pair_rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for left in range(indices.shape[0]):
        for local_index, right in enumerate(indices[left]):
            pair = (left, int(right)) if left < int(right) else (int(right), left)
            if pair[0] == pair[1] or pair in seen:
                continue
            seen.add(pair)
            l_idx, r_idx = pair
            fisher_delta = float(fisher[l_idx] - fisher[r_idx])
            correctness_delta = float(correctness[l_idx] - correctness[r_idx])
            neg_adv_delta = float(neg_adv[l_idx] - neg_adv[r_idx])
            pair_rows.append(
                {
                    "left_idx": int(l_idx),
                    "right_idx": int(r_idx),
                    "left_imdb_id": int(imdb_ids[l_idx]),
                    "right_imdb_id": int(imdb_ids[r_idx]),
                    "pid_distance": float(np.linalg.norm(pid_z[l_idx] - pid_z[r_idx])),
                    "knn_distance": float(distances[left, local_index]),
                    "fisher_delta": fisher_delta,
                    "correctness_delta": correctness_delta,
                    "neg_adv_delta": neg_adv_delta,
                }
            )

    fisher_deltas = np.asarray([row["fisher_delta"] for row in pair_rows], dtype=np.float32)
    correctness_deltas = np.asarray([row["correctness_delta"] for row in pair_rows], dtype=np.float32)
    neg_adv_deltas = np.asarray([row["neg_adv_delta"] for row in pair_rows], dtype=np.float32)
    pid_distances = np.asarray([row["pid_distance"] for row in pair_rows], dtype=np.float32)
    correctness_mask = (np.abs(fisher_deltas) > 1e-6) & (np.abs(correctness_deltas) > 1e-6)
    neg_adv_threshold = float(np.median(np.abs(neg_adv_deltas))) if neg_adv_deltas.size > 0 else 0.0
    neg_adv_mask = (np.abs(fisher_deltas) > 1e-6) & (np.abs(neg_adv_deltas) > max(neg_adv_threshold, 1e-6))

    def _concordance(mask: np.ndarray, outcome_deltas: np.ndarray) -> dict[str, Any]:
        if int(mask.sum()) == 0:
            return {"num_pairs": 0, "concordance_rate": float("nan"), "binomtest_p_value": None}
        concordant = np.sign(fisher_deltas[mask]) == np.sign(outcome_deltas[mask])
        successes = int(concordant.sum())
        total = int(concordant.shape[0])
        test = stats.binomtest(successes, total, p=0.5, alternative="greater")
        return {
            "num_pairs": total,
            "concordance_rate": float(successes / total),
            "binomtest_p_value": float(test.pvalue),
        }

    concordance_correctness = _concordance(correctness_mask, correctness_deltas)
    concordance_neg_adv = _concordance(neg_adv_mask, neg_adv_deltas)
    q25 = float(np.quantile(pid_distances, 0.25)) if pid_distances.size > 0 else float("nan")
    q10 = float(np.quantile(pid_distances, 0.10)) if pid_distances.size > 0 else float("nan")
    meaningful_correctness_gap = np.abs(correctness_deltas) >= (1.0 / 3.0 - 1e-6)
    correctness_subsets = {
        "all_nonzero_pairs": correctness_mask,
        "tight_pid_nonzero_pairs": correctness_mask & (pid_distances <= q25),
        "very_tight_pid_nonzero_pairs": correctness_mask & (pid_distances <= q10),
        "tight_pid_meaningful_gap_pairs": correctness_mask & (pid_distances <= q25) & meaningful_correctness_gap,
        "very_tight_pid_meaningful_gap_pairs": correctness_mask & (pid_distances <= q10) & meaningful_correctness_gap,
    }
    filtered_correctness = {name: _concordance(mask, correctness_deltas) for name, mask in correctness_subsets.items()}

    concordant_score = np.abs(fisher_deltas) + np.abs(correctness_deltas) + np.abs(neg_adv_deltas) - np.asarray(
        [row["knn_distance"] for row in pair_rows], dtype=np.float32
    )
    concordant_mask = np.sign(fisher_deltas) == np.sign(correctness_deltas)
    discordant_mask = ~concordant_mask

    def _unique_top_indices(mask: np.ndarray) -> list[int]:
        ordered = np.argsort(-np.where(mask, concordant_score, -np.inf))
        chosen: list[int] = []
        used: set[int] = set()
        for index in ordered:
            if not np.isfinite(concordant_score[index]):
                continue
            left_idx = pair_rows[int(index)]["left_idx"]
            right_idx = pair_rows[int(index)]["right_idx"]
            if left_idx in used or right_idx in used:
                continue
            used.add(left_idx)
            used.add(right_idx)
            chosen.append(int(index))
            if len(chosen) >= top_k:
                break
        return chosen

    top_concordant_indices = _unique_top_indices(concordant_mask)
    top_discordant_indices = _unique_top_indices(discordant_mask)

    return {
        "neighborhood_k": neighborhood_k,
        "pid_distance_thresholds": {"q25": q25, "q10": q10},
        "correctness_concordance": concordance_correctness,
        "filtered_correctness_concordance": filtered_correctness,
        "neg_adv_concordance": concordance_neg_adv,
        "top_concordant_pairs": [pair_rows[index] for index in top_concordant_indices if np.isfinite(concordant_score[index])],
        "top_discordant_pairs": [pair_rows[index] for index in top_discordant_indices if np.isfinite(concordant_score[index])],
    }


def _plot_scatter(
    *,
    synergy: np.ndarray,
    fisher: np.ndarray,
    correctness: np.ndarray,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    scatter = ax.scatter(
        synergy,
        fisher,
        c=correctness,
        cmap="viridis",
        s=10,
        alpha=0.55,
        linewidths=0.0,
    )
    ax.set_xlabel("LSMI synergy")
    ax.set_ylabel("Fisher score")
    ax.set_title("Sample-level PID synergy vs Fisher score")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correctness rate across seeds")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_seed_payloads(runs_dir: Path) -> list[dict[str, Any]]:
    files = sorted(runs_dir.glob("imdb_lsmi_fisher_seed*.json"))
    if not files:
        raise FileNotFoundError(f"No IMDb multiseed runs found under {runs_dir}")
    return [json_load(path) for path in files]


def json_load(path: Path) -> dict[str, Any]:
    return __import__("json").loads(path.read_text())


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    payloads = _load_seed_payloads(runs_dir)

    per_seed: list[dict[str, np.ndarray]] = []
    reference_sample_ids: list[str] | None = None
    for payload in payloads:
        analysis = payload["analysis"]
        features = analysis["features"]
        valid_mask = np.asarray(analysis["valid_regression_mask"], dtype=bool)
        sample_ids = [str(item) for item in analysis.get("sample_ids", [])]
        if sample_ids:
            if reference_sample_ids is None:
                reference_sample_ids = sample_ids
            elif reference_sample_ids != sample_ids:
                raise ValueError("Seed payloads do not align on sample IDs.")
        pid_names = list(features["pid_feature_names"])
        pid_feature_values = features["pid_feature_values"]
        pid = np.stack([np.asarray(pid_feature_values[name], dtype=np.float32) for name in pid_names], axis=1)[valid_mask]
        fisher_pair = np.stack(
            [
                np.asarray(features["gradcos"], dtype=np.float32),
                np.asarray(features["curvagree"], dtype=np.float32),
            ],
            axis=1,
        )[valid_mask]
        correctness = np.asarray(analysis["correctness"], dtype=np.int64)[valid_mask]
        negative_loss_advantage = np.asarray(features["negative_loss_advantage"], dtype=np.float32)[valid_mask]

        per_seed.append(
            {
                "pid": pid,
                "fisher_pair": fisher_pair,
                "fisher_score": np.asarray(features["fisher_score"], dtype=np.float32)[valid_mask],
                "synergy": np.asarray(pid_feature_values["synergy"], dtype=np.float32)[valid_mask],
                "correctness": correctness.astype(np.float32),
                "negative_loss_advantage": negative_loss_advantage,
            }
        )

    n_samples = per_seed[0]["pid"].shape[0]
    if any(item["pid"].shape[0] != n_samples for item in per_seed):
        raise ValueError("Seed payloads do not align on sample count.")

    pid_feature_names = list(payloads[0]["analysis"]["features"]["pid_feature_names"])
    mean_pid = np.mean(np.stack([item["pid"] for item in per_seed], axis=0), axis=0)
    mean_fisher_pair = np.mean(np.stack([item["fisher_pair"] for item in per_seed], axis=0), axis=0)
    mean_fisher = np.mean(np.stack([item["fisher_score"] for item in per_seed], axis=0), axis=0)
    mean_synergy = np.mean(np.stack([item["synergy"] for item in per_seed], axis=0), axis=0)
    mean_correctness = np.mean(np.stack([item["correctness"] for item in per_seed], axis=0), axis=0)
    mean_neg_adv = np.mean(np.stack([item["negative_loss_advantage"] for item in per_seed], axis=0), axis=0)

    if reference_sample_ids is not None:
        imdb_ids = np.asarray([int(item) for item in reference_sample_ids], dtype=np.int64)
    else:
        with h5py.File(args.imdb_path, "r") as handle:
            imdb_ids = np.asarray(handle["imdb_ids"][18160:25959], dtype=np.int64)

    neighborhood_residual_analysis = _compute_neighborhood_residual_analysis(
        pid_features=mean_pid,
        fisher=mean_fisher,
        correctness=mean_correctness,
        neg_adv=mean_neg_adv,
        neighborhood_k=args.neighborhood_k,
    )
    local_fisher_stratification_analysis = _compute_local_fisher_stratification_analysis(
        pid_features=mean_pid,
        fisher=mean_fisher,
        correctness=mean_correctness,
        neg_adv=mean_neg_adv,
        imdb_ids=imdb_ids,
        neighborhood_k=args.neighborhood_k,
        top_k=args.top_k,
    )
    pid_matched_pair_analysis = _compute_pair_concordance_analysis(
        pid_features=mean_pid,
        fisher=mean_fisher,
        correctness=mean_correctness,
        neg_adv=mean_neg_adv,
        imdb_ids=imdb_ids,
        neighborhood_k=args.neighborhood_k,
        top_k=args.top_k,
    )

    _plot_scatter(
        synergy=mean_synergy,
        fisher=mean_fisher,
        correctness=mean_correctness,
        output_path=args.scatter_path,
    )

    results = {
        "pid_feature_names": pid_feature_names,
        "num_seeds": len(per_seed),
        "num_samples": int(n_samples),
        "sample_alignment": {
            "checked_with_sample_ids": reference_sample_ids is not None,
            "sample_ids": reference_sample_ids,
        },
        "aggregate_feature_summary": {
            "fisher_score": _mean_std(mean_fisher),
            "gradcos": _mean_std(mean_fisher_pair[:, 0]),
            "curvagree": _mean_std(mean_fisher_pair[:, 1]),
            "correctness_rate": _mean_std(mean_correctness),
            "negative_loss_advantage": _mean_std(mean_neg_adv),
        },
        "pid_neighborhood_residual_analysis": neighborhood_residual_analysis,
        "local_fisher_stratification_analysis": local_fisher_stratification_analysis,
        "pid_matched_pair_analysis": pid_matched_pair_analysis,
        "artifacts": {"scatter": str(args.scatter_path)},
    }
    write_json(results, args.output_json)


if __name__ == "__main__":
    main()
