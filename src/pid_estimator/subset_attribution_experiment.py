from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from pid_estimator.subset_attribution import (
    RBFSubsetSurrogateConfig,
    StructuredSubsetSurrogateConfig,
    fit_linear_subset_surrogate,
    fit_rbf_subset_surrogate,
    fit_structured_subset_surrogate,
    make_coupling_game,
    make_exact_logic_bridge_game,
    make_missing_modality_rescue_game,
    make_motif_mixture_game,
    sample_global_subset_queries,
    sample_pair_probes,
)


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    exact_num_concepts: int = 4
    mixture_num_concepts: int = 8
    coupling_num_concepts: int = 8
    exact_num_queries: int = 160
    budget_num_queries: int = 320
    holdout_num_queries: int = 512
    pair_probes_per_edge: int = 4
    ridge_lambda: float = 1e-4
    pair_probe_weight: float = 4.0
    feature_bernoulli_p: float = 0.5
    logic_case: str = "xor"
    mixture_probs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    synergy_sweep: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    coupling_sweep: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    ablation_num_concepts: int = 12
    ablation_num_queries: int = 32
    rescue_num_concepts: int = 12
    rescue_num_queries: int = 32
    rescue_missing_rate: float = 0.6
    rescue_partner_purity: float = 0.8
    rescue_missing_sweep: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    rescue_partner_sweep: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structured subset-attribution surrogate experiments on synthetic games.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--exact-num-concepts", type=int, default=4)
    parser.add_argument("--mixture-num-concepts", type=int, default=8)
    parser.add_argument("--coupling-num-concepts", type=int, default=8)
    parser.add_argument("--exact-num-queries", type=int, default=160)
    parser.add_argument("--budget-num-queries", type=int, default=320)
    parser.add_argument("--holdout-num-queries", type=int, default=512)
    parser.add_argument("--pair-probes-per-edge", type=int, default=4)
    parser.add_argument("--ridge-lambda", type=float, default=1e-4)
    parser.add_argument("--pair-probe-weight", type=float, default=4.0)
    parser.add_argument("--feature-bernoulli-p", type=float, default=0.5)
    parser.add_argument("--logic-case", type=str, default="xor")
    parser.add_argument("--ablation-num-concepts", type=int, default=12)
    parser.add_argument("--ablation-num-queries", type=int, default=32)
    parser.add_argument("--rescue-num-concepts", type=int, default=12)
    parser.add_argument("--rescue-num-queries", type=int, default=32)
    parser.add_argument("--rescue-missing-rate", type=float, default=0.6)
    parser.add_argument("--rescue-partner-purity", type=float, default=0.8)
    parser.add_argument("--output-json", type=str, default="src/pid_estimator/results/subset_attribution_experiment.json")
    parser.add_argument("--output-md", type=str, default="src/pid_estimator/results/subset_attribution_experiment.md")
    args = parser.parse_args()

    config = ExperimentConfig(
        seeds=tuple(args.seeds),
        exact_num_concepts=args.exact_num_concepts,
        mixture_num_concepts=args.mixture_num_concepts,
        coupling_num_concepts=args.coupling_num_concepts,
        exact_num_queries=args.exact_num_queries,
        budget_num_queries=args.budget_num_queries,
        holdout_num_queries=args.holdout_num_queries,
        pair_probes_per_edge=args.pair_probes_per_edge,
        ridge_lambda=args.ridge_lambda,
        pair_probe_weight=args.pair_probe_weight,
        feature_bernoulli_p=args.feature_bernoulli_p,
        logic_case=args.logic_case,
        ablation_num_concepts=args.ablation_num_concepts,
        ablation_num_queries=args.ablation_num_queries,
        rescue_num_concepts=args.rescue_num_concepts,
        rescue_num_queries=args.rescue_num_queries,
        rescue_missing_rate=args.rescue_missing_rate,
        rescue_partner_purity=args.rescue_partner_purity,
    )
    payload = run_all_experiments(config)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(payload), encoding="utf-8")
    print(to_markdown(payload))


def run_all_experiments(config: ExperimentConfig) -> dict[str, object]:
    exact_runs = [run_exact_logic(seed, config) for seed in config.seeds]
    motif_runs = [run_motif_mixture(seed, config) for seed in config.seeds]
    synergy_sweep_runs = [run_synergy_sweep(seed, config) for seed in config.seeds]
    coupling_sweep_runs = [run_coupling_sweep(seed, config) for seed in config.seeds]
    pair_probe_ablation_runs = [run_pair_probe_ablation(seed, config) for seed in config.seeds]
    baseline_comparison_runs = [run_baseline_comparison(seed, config) for seed in config.seeds]
    rescue_benchmark_runs = [run_rescue_benchmark(seed, config) for seed in config.seeds]
    return {
        "config": {
            **asdict(config),
            "surrogate_config": asdict(
                StructuredSubsetSurrogateConfig(
                    feature_bernoulli_p=config.feature_bernoulli_p,
                    ridge_lambda=config.ridge_lambda,
                    pair_probe_weight=config.pair_probe_weight,
                )
            ),
        },
        "exact_logic": {
            "runs": exact_runs,
            "summary": summarize_numeric_dicts(run["metrics"] for run in exact_runs),
        },
        "motif_mixture": {
            "runs": motif_runs,
            "summary": summarize_numeric_dicts(run["metrics"] for run in motif_runs),
        },
        "synergy_sweep": {
            "runs": synergy_sweep_runs,
            "summary": summarize_sweep_runs(synergy_sweep_runs, key_name="pi_synergy"),
        },
        "coupling_sweep": {
            "runs": coupling_sweep_runs,
            "summary": summarize_sweep_runs(coupling_sweep_runs, key_name="coupling_strength"),
        },
        "pair_probe_ablation": {
            "runs": pair_probe_ablation_runs,
            "summary": summarize_method_metric_runs(pair_probe_ablation_runs),
        },
        "baseline_comparison": {
            "runs": baseline_comparison_runs,
            "summary": summarize_method_metric_runs(baseline_comparison_runs),
        },
        "missing_modality_rescue": {
            "runs": rescue_benchmark_runs,
            "summary": summarize_method_metric_runs(rescue_benchmark_runs),
        },
    }


def run_exact_logic(seed: int, config: ExperimentConfig) -> dict[str, object]:
    game = make_exact_logic_bridge_game(num_concepts=config.exact_num_concepts, logic_case=config.logic_case)
    model = fit_game(
        game=game,
        query_seed=seed,
        num_queries=config.exact_num_queries,
        pair_probes_per_edge=config.pair_probes_per_edge,
        fit_config=config,
    )
    all_masks = enumerate_masks(game.num_atoms)
    all_targets = game.utility_batch(all_masks)
    all_predictions = model.predict_batch(all_masks)
    estimated = model.estimate_pair_interactions()
    exact = game.exact_pair_interactions(exhaustive=True)
    estimated_array = np.asarray([estimated[edge] for edge in game.candidate_edges], dtype=np.float64)
    exact_array = np.asarray([exact[edge] for edge in game.candidate_edges], dtype=np.float64)
    metrics = {
        "r2": r2_score(all_targets, all_predictions),
        "rmse": rmse(all_targets, all_predictions),
        "mae": mae(all_targets, all_predictions),
        "pair_rmse": rmse(exact_array, estimated_array),
        "pair_pearson": pearson_corr(exact_array, estimated_array),
        "pair_spearman": spearman_corr(exact_array, estimated_array),
        "pair_sign_acc": sign_accuracy(exact_array, estimated_array),
        "positive_mass_error": abs(game.positive_interaction_mass(exact) - model.attribution_mass(positive_only=True)),
    }
    return {
        "seed": seed,
        "num_atoms": game.num_atoms,
        "num_candidate_edges": len(game.candidate_edges),
        "metrics": metrics,
    }


def run_motif_mixture(seed: int, config: ExperimentConfig) -> dict[str, object]:
    game = make_motif_mixture_game(
        num_concepts=config.mixture_num_concepts,
        pi_redundancy=config.mixture_probs[0],
        pi_unique=config.mixture_probs[1],
        pi_synergy=config.mixture_probs[2],
        seed=seed,
    )
    model = fit_game(
        game=game,
        query_seed=seed,
        num_queries=config.budget_num_queries,
        pair_probes_per_edge=config.pair_probes_per_edge,
        fit_config=config,
    )
    holdout_masks, holdout_targets = sample_global_subset_queries(
        game,
        num_queries=config.holdout_num_queries,
        seed=10_000 + seed,
    )
    holdout_predictions = model.predict_batch(holdout_masks)
    estimated = model.estimate_pair_interactions()
    exact = game.exact_pair_interactions(exhaustive=False)
    scores = np.asarray([estimated[edge] for edge in game.candidate_edges], dtype=np.float64)
    exact_scores = np.asarray([exact[edge] for edge in game.candidate_edges], dtype=np.float64)
    labels = [game.edge_labels[edge] for edge in game.candidate_edges]
    tau = 0.05 * float(np.max(np.abs(scores))) if scores.size else 0.0
    predictions = classify_scores(scores, tau=tau)
    metrics = {
        "r2": r2_score(holdout_targets, holdout_predictions),
        "rmse": rmse(holdout_targets, holdout_predictions),
        "mae": mae(holdout_targets, holdout_predictions),
        "pair_rmse": rmse(exact_scores, scores),
        "pair_pearson": pearson_corr(exact_scores, scores),
        "pair_spearman": spearman_corr(exact_scores, scores),
        "auroc_synergy": binary_auroc([label == "S" for label in labels], scores),
        "auprc_synergy": binary_auprc([label == "S" for label in labels], scores),
        "macro_f1": macro_f1(labels, predictions, classes=("R", "U", "S")),
        "balanced_acc": balanced_accuracy(labels, predictions, classes=("R", "U", "S")),
        "faithfulness_at_2": faithfulness_at_k(game, model, k=2, seed=seed),
        "pair_gain_at_2": pair_gain_at_k(game, model, k=2),
        "positive_mass": model.attribution_mass(positive_only=True),
    }
    return {
        "seed": seed,
        "num_atoms": game.num_atoms,
        "num_candidate_edges": len(game.candidate_edges),
        "metrics": metrics,
    }


def run_synergy_sweep(seed: int, config: ExperimentConfig) -> dict[str, object]:
    rows: list[dict[str, float]] = []
    for index, pi_synergy in enumerate(config.synergy_sweep):
        remainder = 1.0 - pi_synergy
        pi_redundancy = remainder / 2.0
        pi_unique = remainder / 2.0
        game = make_motif_mixture_game(
            num_concepts=config.mixture_num_concepts,
            pi_redundancy=pi_redundancy,
            pi_unique=pi_unique,
            pi_synergy=pi_synergy,
            seed=100 * seed + index,
        )
        model = fit_game(
            game=game,
            query_seed=100 * seed + index,
            num_queries=config.budget_num_queries,
            pair_probes_per_edge=config.pair_probes_per_edge,
            fit_config=config,
        )
        rows.append(
            {
                "pi_synergy": float(pi_synergy),
                "positive_mass": model.attribution_mass(positive_only=True),
                "exact_positive_mass": game.positive_interaction_mass(),
            }
        )
    pi_values = np.asarray([row["pi_synergy"] for row in rows], dtype=np.float64)
    masses = np.asarray([row["positive_mass"] for row in rows], dtype=np.float64)
    exact_masses = np.asarray([row["exact_positive_mass"] for row in rows], dtype=np.float64)
    return {
        "seed": seed,
        "rows": rows,
        "metrics": {
            "corr_pi_synergy_vs_positive_mass": pearson_corr(pi_values, masses),
            "corr_exact_vs_estimated_mass": pearson_corr(exact_masses, masses),
            "monotonicity": monotonicity_score(masses),
        },
    }


def run_coupling_sweep(seed: int, config: ExperimentConfig) -> dict[str, object]:
    rows: list[dict[str, float]] = []
    for index, coupling_strength in enumerate(config.coupling_sweep):
        game = make_coupling_game(
            num_concepts=config.coupling_num_concepts,
            coupling_strength=coupling_strength,
        )
        model = fit_game(
            game=game,
            query_seed=1_000 + 100 * seed + index,
            num_queries=config.budget_num_queries,
            pair_probes_per_edge=config.pair_probes_per_edge,
            fit_config=config,
        )
        rows.append(
            {
                "coupling_strength": float(coupling_strength),
                "positive_mass": model.attribution_mass(positive_only=True),
                "exact_positive_mass": game.positive_interaction_mass(),
            }
        )
    strengths = np.asarray([row["coupling_strength"] for row in rows], dtype=np.float64)
    masses = np.asarray([row["positive_mass"] for row in rows], dtype=np.float64)
    exact_masses = np.asarray([row["exact_positive_mass"] for row in rows], dtype=np.float64)
    return {
        "seed": seed,
        "rows": rows,
        "metrics": {
            "corr_coupling_vs_positive_mass": pearson_corr(strengths, masses),
            "corr_exact_vs_estimated_mass": pearson_corr(exact_masses, masses),
            "monotonicity": monotonicity_score(masses),
        },
    }


def run_pair_probe_ablation(seed: int, config: ExperimentConfig) -> dict[str, object]:
    methods = ("structured_no_pair_probes", "structured_pair_probes")
    game = make_motif_mixture_game(
        num_concepts=config.ablation_num_concepts,
        pi_redundancy=config.mixture_probs[0],
        pi_unique=config.mixture_probs[1],
        pi_synergy=config.mixture_probs[2],
        seed=seed,
    )
    subset_masks, subset_values, pair_probes = sample_training_queries(game, seed=seed, config=config, num_queries=config.ablation_num_queries)
    holdout_masks, holdout_targets = sample_global_subset_queries(game, num_queries=config.holdout_num_queries, seed=10_000 + seed)
    exact_scores = game.exact_pair_interactions(exhaustive=False)
    exact_mass = game.positive_interaction_mass(exact_scores)
    method_metrics = {}
    for method in methods:
        model = fit_method_from_queries(method, game, subset_masks, subset_values, pair_probes, config)
        predictions = model.predict_batch(holdout_masks)
        estimated_scores = model.estimate_pair_interactions()
        score_array = np.asarray([estimated_scores[edge] for edge in game.candidate_edges], dtype=np.float64)
        exact_array = np.asarray([exact_scores[edge] for edge in game.candidate_edges], dtype=np.float64)
        labels = [game.edge_labels[edge] for edge in game.candidate_edges]
        tau = 0.05 * float(np.max(np.abs(score_array))) if score_array.size else 0.0
        method_metrics[method] = {
            "utility_r2": r2_score(holdout_targets, predictions),
            "pair_rmse": rmse(exact_array, score_array),
            "auroc_synergy": binary_auroc([label == "S" for label in labels], score_array),
            "auprc_synergy": binary_auprc([label == "S" for label in labels], score_array),
            "mass_abs_error": abs(model.attribution_mass(positive_only=True) - exact_mass),
        }

    sweep_scores = {method: [] for method in methods}
    sweep_exact_scores = []
    for index, pi_synergy in enumerate(config.synergy_sweep):
        remainder = 1.0 - pi_synergy
        game = make_motif_mixture_game(
            num_concepts=config.ablation_num_concepts,
            pi_redundancy=remainder / 2.0,
            pi_unique=remainder / 2.0,
            pi_synergy=pi_synergy,
            seed=100 * seed + index,
        )
        subset_masks, subset_values, pair_probes = sample_training_queries(
            game,
            seed=100 * seed + index,
            config=config,
            num_queries=config.ablation_num_queries,
        )
        sweep_exact_scores.append(game.positive_interaction_mass())
        for method in methods:
            model = fit_method_from_queries(method, game, subset_masks, subset_values, pair_probes, config)
            sweep_scores[method].append(model.attribution_mass(positive_only=True))
    synergy_values = np.asarray(config.synergy_sweep, dtype=np.float64)
    exact_masses = np.asarray(sweep_exact_scores, dtype=np.float64)
    for method in methods:
        masses = np.asarray(sweep_scores[method], dtype=np.float64)
        method_metrics[method].update(
            {
                "sweep_corr_pi_synergy_vs_positive_mass": pearson_corr(synergy_values, masses),
                "sweep_corr_exact_vs_estimated_mass": pearson_corr(exact_masses, masses),
                "sweep_monotonicity": monotonicity_score(masses),
            }
        )
    return {"seed": seed, "methods": method_metrics}


def run_baseline_comparison(seed: int, config: ExperimentConfig) -> dict[str, object]:
    methods = ("linear", "rbf", "structured_no_pair_probes", "structured_pair_probes")
    metrics_by_method = {method: {} for method in methods}

    exact_game = make_exact_logic_bridge_game(num_concepts=config.exact_num_concepts, logic_case=config.logic_case)
    exact_masks, exact_values, exact_pair_probes = sample_training_queries(
        exact_game,
        seed=seed,
        config=config,
        num_queries=config.exact_num_queries,
    )
    exact_truth = exact_game.exact_pair_interactions(exhaustive=True)
    for method in methods:
        model = fit_method_from_queries(method, exact_game, exact_masks, exact_values, exact_pair_probes, config)
        estimated = model.estimate_pair_interactions()
        estimated_array = np.asarray([estimated[edge] for edge in exact_game.candidate_edges], dtype=np.float64)
        exact_array = np.asarray([exact_truth[edge] for edge in exact_game.candidate_edges], dtype=np.float64)
        metrics_by_method[method]["exact_pair_rmse"] = rmse(exact_array, estimated_array)

    mixture_game = make_motif_mixture_game(
        num_concepts=config.mixture_num_concepts,
        pi_redundancy=config.mixture_probs[0],
        pi_unique=config.mixture_probs[1],
        pi_synergy=config.mixture_probs[2],
        seed=seed,
    )
    subset_masks, subset_values, pair_probes = sample_training_queries(
        mixture_game,
        seed=seed,
        config=config,
        num_queries=config.budget_num_queries,
    )
    holdout_masks, holdout_targets = sample_global_subset_queries(mixture_game, num_queries=config.holdout_num_queries, seed=20_000 + seed)
    exact_scores = mixture_game.exact_pair_interactions(exhaustive=False)
    exact_mass = mixture_game.positive_interaction_mass(exact_scores)
    labels = [mixture_game.edge_labels[edge] for edge in mixture_game.candidate_edges]
    exact_array = np.asarray([exact_scores[edge] for edge in mixture_game.candidate_edges], dtype=np.float64)
    for method in methods:
        model = fit_method_from_queries(method, mixture_game, subset_masks, subset_values, pair_probes, config)
        holdout_predictions = model.predict_batch(holdout_masks)
        estimated = model.estimate_pair_interactions()
        score_array = np.asarray([estimated[edge] for edge in mixture_game.candidate_edges], dtype=np.float64)
        tau = 0.05 * float(np.max(np.abs(score_array))) if score_array.size else 0.0
        predicted_labels = classify_scores(score_array, tau=tau)
        metrics_by_method[method].update(
            {
                "mixture_utility_r2": r2_score(holdout_targets, holdout_predictions),
                "mixture_pair_rmse": rmse(exact_array, score_array),
                "mixture_auroc_synergy": binary_auroc([label == "S" for label in labels], score_array),
                "mixture_auprc_synergy": binary_auprc([label == "S" for label in labels], score_array),
                "mixture_macro_f1": macro_f1(labels, predicted_labels, classes=("R", "U", "S")),
                "mixture_mass_abs_error": abs(model.attribution_mass(positive_only=True) - exact_mass),
            }
        )

    exact_masses = []
    sweep_masses = {method: [] for method in methods}
    for index, pi_synergy in enumerate(config.synergy_sweep):
        remainder = 1.0 - pi_synergy
        game = make_motif_mixture_game(
            num_concepts=config.mixture_num_concepts,
            pi_redundancy=remainder / 2.0,
            pi_unique=remainder / 2.0,
            pi_synergy=pi_synergy,
            seed=200 * seed + index,
        )
        subset_masks, subset_values, pair_probes = sample_training_queries(game, seed=200 * seed + index, config=config, num_queries=config.budget_num_queries)
        exact_masses.append(game.positive_interaction_mass())
        for method in methods:
            model = fit_method_from_queries(method, game, subset_masks, subset_values, pair_probes, config)
            sweep_masses[method].append(model.attribution_mass(positive_only=True))
    synergy_values = np.asarray(config.synergy_sweep, dtype=np.float64)
    exact_masses_array = np.asarray(exact_masses, dtype=np.float64)
    for method in methods:
        masses = np.asarray(sweep_masses[method], dtype=np.float64)
        metrics_by_method[method].update(
            {
                "sweep_corr_pi_synergy_vs_positive_mass": pearson_corr(synergy_values, masses),
                "sweep_corr_exact_vs_estimated_mass": pearson_corr(exact_masses_array, masses),
                "sweep_monotonicity": monotonicity_score(masses),
            }
        )
    return {"seed": seed, "methods": metrics_by_method}


def run_rescue_benchmark(seed: int, config: ExperimentConfig) -> dict[str, object]:
    methods = ("structured_no_pair_probes", "structured_pair_probes")
    metrics_by_method = {method: {} for method in methods}

    game = make_missing_modality_rescue_game(
        num_concepts=config.rescue_num_concepts,
        missing_rate=config.rescue_missing_rate,
        partner_purity=config.rescue_partner_purity,
        seed=seed,
    )
    subset_masks, subset_values, pair_probes = sample_training_queries(game, seed=seed, config=config, num_queries=config.rescue_num_queries)
    true_labels = [edge in game.rescue_edges for edge in game.candidate_edges]  # type: ignore[attr-defined]
    rescue_k = max(1, int(sum(true_labels)))
    for method in methods:
        model = fit_method_from_queries(method, game, subset_masks, subset_values, pair_probes, config)
        estimated = model.estimate_pair_interactions()
        score_array = np.asarray([estimated[edge] for edge in game.candidate_edges], dtype=np.float64)
        metrics_by_method[method].update(
            {
                "rescue_recall_at_k": binary_recall_at_k(true_labels, score_array, k=rescue_k),
                "rescue_auprc": binary_auprc(true_labels, score_array),
                "rescue_auroc": binary_auroc(true_labels, score_array),
                "rescue_positive_mass": model.attribution_mass(positive_only=True),
            }
        )

    missing_values = np.asarray(config.rescue_missing_sweep, dtype=np.float64)
    partner_values = np.asarray(config.rescue_partner_sweep, dtype=np.float64)
    missing_masses = {method: [] for method in methods}
    partner_masses = {method: [] for method in methods}

    for index, missing_rate in enumerate(config.rescue_missing_sweep):
        game = make_missing_modality_rescue_game(
            num_concepts=config.rescue_num_concepts,
            missing_rate=missing_rate,
            partner_purity=config.rescue_partner_purity,
            seed=300 * seed + index,
        )
        subset_masks, subset_values, pair_probes = sample_training_queries(game, seed=300 * seed + index, config=config, num_queries=config.rescue_num_queries)
        for method in methods:
            model = fit_method_from_queries(method, game, subset_masks, subset_values, pair_probes, config)
            missing_masses[method].append(model.attribution_mass(positive_only=True))

    for index, partner_purity in enumerate(config.rescue_partner_sweep):
        game = make_missing_modality_rescue_game(
            num_concepts=config.rescue_num_concepts,
            missing_rate=config.rescue_missing_rate,
            partner_purity=partner_purity,
            seed=400 * seed + index,
        )
        subset_masks, subset_values, pair_probes = sample_training_queries(game, seed=400 * seed + index, config=config, num_queries=config.rescue_num_queries)
        for method in methods:
            model = fit_method_from_queries(method, game, subset_masks, subset_values, pair_probes, config)
            partner_masses[method].append(model.attribution_mass(positive_only=True))

    for method in methods:
        missing_array = np.asarray(missing_masses[method], dtype=np.float64)
        partner_array = np.asarray(partner_masses[method], dtype=np.float64)
        metrics_by_method[method].update(
            {
                "missing_sweep_corr_missing_rate_vs_positive_mass": pearson_corr(missing_values, missing_array),
                "missing_sweep_monotonicity": monotonicity_score(missing_array),
                "partner_sweep_corr_partner_purity_vs_positive_mass": pearson_corr(partner_values, partner_array),
                "partner_sweep_monotonicity": monotonicity_score(partner_array),
            }
        )
    return {"seed": seed, "methods": metrics_by_method}


def sample_training_queries(game, *, seed: int, config: ExperimentConfig, num_queries: int) -> tuple[np.ndarray, np.ndarray, list]:
    subset_masks, subset_values = sample_global_subset_queries(game, num_queries=num_queries, seed=seed)
    pair_probes = sample_pair_probes(game, num_backgrounds_per_edge=config.pair_probes_per_edge, seed=50_000 + seed)
    return subset_masks, subset_values, pair_probes


def fit_method_from_queries(method: str, game, subset_masks: np.ndarray, subset_values: np.ndarray, pair_probes: list, config: ExperimentConfig):
    structured_config = StructuredSubsetSurrogateConfig(
        feature_bernoulli_p=config.feature_bernoulli_p,
        ridge_lambda=config.ridge_lambda,
        pair_probe_weight=config.pair_probe_weight,
    )
    if method == "structured_pair_probes":
        return fit_structured_subset_surrogate(
            subset_masks,
            subset_values,
            num_atoms=game.num_atoms,
            candidate_edges=game.candidate_edges,
            pair_probes=pair_probes,
            config=structured_config,
        )
    if method == "structured_no_pair_probes":
        return fit_structured_subset_surrogate(
            subset_masks,
            subset_values,
            num_atoms=game.num_atoms,
            candidate_edges=game.candidate_edges,
            pair_probes=None,
            config=structured_config,
        )
    if method == "linear":
        return fit_linear_subset_surrogate(
            subset_masks,
            subset_values,
            num_atoms=game.num_atoms,
            candidate_edges=game.candidate_edges,
            config=structured_config,
        )
    if method == "rbf":
        return fit_rbf_subset_surrogate(
            subset_masks,
            subset_values,
            num_atoms=game.num_atoms,
            candidate_edges=game.candidate_edges,
            config=RBFSubsetSurrogateConfig(ridge_lambda=config.ridge_lambda, gamma=None),
        )
    raise ValueError(f"unknown method: {method}")


def fit_game(game, *, query_seed: int, num_queries: int, pair_probes_per_edge: int, fit_config: ExperimentConfig):
    subset_masks, subset_values = sample_global_subset_queries(
        game,
        num_queries=num_queries,
        seed=query_seed,
    )
    pair_probes = sample_pair_probes(
        game,
        num_backgrounds_per_edge=pair_probes_per_edge,
        seed=50_000 + query_seed,
    )
    return fit_structured_subset_surrogate(
        subset_masks,
        subset_values,
        num_atoms=game.num_atoms,
        candidate_edges=game.candidate_edges,
        pair_probes=pair_probes,
        config=StructuredSubsetSurrogateConfig(
            feature_bernoulli_p=fit_config.feature_bernoulli_p,
            ridge_lambda=fit_config.ridge_lambda,
            pair_probe_weight=fit_config.pair_probe_weight,
        ),
    )


def enumerate_masks(num_atoms: int) -> np.ndarray:
    masks = np.zeros((2**num_atoms, num_atoms), dtype=bool)
    for index in range(2**num_atoms):
        for bit in range(num_atoms):
            masks[index, bit] = bool((index >> bit) & 1)
    return masks


def summarize_numeric_dicts(rows: Iterable[dict[str, float]]) -> dict[str, dict[str, float]]:
    collected: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            collected.setdefault(key, []).append(float(value))
    return {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for key, values in collected.items()
    }


def summarize_sweep_runs(runs: list[dict[str, object]], *, key_name: str) -> dict[str, object]:
    metric_summary = summarize_numeric_dicts(run["metrics"] for run in runs)
    row_means: dict[float, list[float]] = {}
    exact_means: dict[float, list[float]] = {}
    for run in runs:
        for row in run["rows"]:
            key = float(row[key_name])
            row_means.setdefault(key, []).append(float(row["positive_mass"]))
            exact_means.setdefault(key, []).append(float(row["exact_positive_mass"]))
    curve = [
        {
            key_name: key,
            "positive_mass_mean": float(np.mean(values)),
            "positive_mass_std": float(np.std(values)),
            "exact_positive_mass_mean": float(np.mean(exact_means[key])),
        }
        for key, values in sorted(row_means.items(), key=lambda item: item[0])
    ]
    return {
        "metrics": metric_summary,
        "curve": curve,
    }


def summarize_method_metric_runs(runs: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    method_metric_values: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        for method, metrics in run["methods"].items():
            method_metric_values.setdefault(method, {})
            for metric_name, value in metrics.items():
                method_metric_values[method].setdefault(metric_name, []).append(float(value))
    return {
        method: {
            metric_name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            for metric_name, values in metric_values.items()
        }
        for method, metric_values in method_metric_values.items()
    }


def r2_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    residual = float(np.sum((targets - predictions) ** 2))
    total = float(np.sum((targets - np.mean(targets)) ** 2))
    if total <= 1e-12:
        return 1.0 if residual <= 1e-12 else 0.0
    return float(1.0 - residual / total)


def rmse(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(targets) - np.asarray(predictions)) ** 2)))


def mae(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(targets) - np.asarray(predictions))))


def pearson_corr(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denominator)


def spearman_corr(first: np.ndarray, second: np.ndarray) -> float:
    return pearson_corr(rankdata(first), rankdata(second))


def rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty_like(array, dtype=np.float64)
    start = 0
    while start < array.shape[0]:
        end = start + 1
        while end < array.shape[0] and array[order[end]] == array[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def sign_accuracy(targets: np.ndarray, predictions: np.ndarray, tolerance: float = 1e-5) -> float:
    target_values = np.asarray(targets, dtype=np.float64)
    prediction_values = np.asarray(predictions, dtype=np.float64)
    target_signs = np.sign(np.where(np.abs(target_values) <= tolerance, 0.0, target_values))
    prediction_signs = np.sign(np.where(np.abs(prediction_values) <= tolerance, 0.0, prediction_values))
    return float(np.mean(target_signs == prediction_signs))


def classify_scores(scores: np.ndarray, *, tau: float) -> list[str]:
    labels = []
    for score in np.asarray(scores, dtype=np.float64):
        if score > tau:
            labels.append("S")
        elif score < -tau:
            labels.append("R")
        else:
            labels.append("U")
    return labels


def binary_auroc(binary_labels: Iterable[bool], scores: np.ndarray) -> float:
    labels = np.asarray(list(binary_labels), dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels))
    negatives = int(np.sum(~labels))
    if positives == 0 or negatives == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = rankdata(scores)
    positive_rank_sum = float(np.sum(ranks[labels]))
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def binary_auprc(binary_labels: Iterable[bool], scores: np.ndarray) -> float:
    labels = np.asarray(list(binary_labels), dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels))
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    for label in labels:
        if label:
            true_positives += 1
        else:
            false_positives += 1
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / positives
        area += precision * (recall - previous_recall)
        previous_recall = recall
    return float(area)


def binary_recall_at_k(binary_labels: Iterable[bool], scores: np.ndarray, *, k: int) -> float:
    labels = np.asarray(list(binary_labels), dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels))
    if positives == 0:
        return 0.0
    topk = np.argsort(-scores, kind="mergesort")[: max(k, 1)]
    return float(np.sum(labels[topk]) / positives)


def macro_f1(true_labels: Sequence[str], predicted_labels: Sequence[str], *, classes: tuple[str, ...]) -> float:
    scores = []
    for class_name in classes:
        true_positive = sum(t == class_name and p == class_name for t, p in zip(true_labels, predicted_labels))
        false_positive = sum(t != class_name and p == class_name for t, p in zip(true_labels, predicted_labels))
        false_negative = sum(t == class_name and p != class_name for t, p in zip(true_labels, predicted_labels))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        if precision + recall <= 1e-12:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def balanced_accuracy(true_labels: Sequence[str], predicted_labels: Sequence[str], *, classes: tuple[str, ...]) -> float:
    recalls = []
    for class_name in classes:
        true_positive = sum(t == class_name and p == class_name for t, p in zip(true_labels, predicted_labels))
        false_negative = sum(t == class_name and p != class_name for t, p in zip(true_labels, predicted_labels))
        recalls.append(true_positive / max(true_positive + false_negative, 1))
    return float(np.mean(recalls))


def monotonicity_score(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[0] <= 1:
        return 1.0
    return float(np.mean(array[1:] >= array[:-1]))


def faithfulness_at_k(game, model, *, k: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    predicted = model.estimate_pair_interactions()
    ranked_pairs = [edge for edge, value in sorted(predicted.items(), key=lambda item: item[1], reverse=True) if value > 0]
    top_pairs = ranked_pairs[:k]
    random_pairs = list(rng.choice(game.candidate_edges, size=min(k, len(game.candidate_edges)), replace=False)) if game.candidate_edges else []
    full_mask = game.full_mask()
    top_mask = remove_pair_endpoints(full_mask, top_pairs)
    rand_mask = remove_pair_endpoints(full_mask, random_pairs)
    drop_top = game.utility(full_mask) - game.utility(top_mask)
    drop_rand = game.utility(full_mask) - game.utility(rand_mask)
    return float(drop_top - drop_rand)


def pair_gain_at_k(game, model, *, k: int) -> float:
    predicted = model.estimate_pair_interactions()
    ranked_pairs = [edge for edge, value in sorted(predicted.items(), key=lambda item: item[1], reverse=True) if value > 0]
    top_pairs = ranked_pairs[:k]
    full_mask = game.full_mask()
    pair_mask = remove_pair_endpoints(full_mask, top_pairs)
    drop_pair = game.utility(full_mask) - game.utility(pair_mask)
    main_effect_order = np.argsort(-np.abs(model.main_effects))
    atoms_to_remove = set(int(index) for index in main_effect_order[: 2 * len(top_pairs)])
    main_mask = full_mask.copy()
    for atom in atoms_to_remove:
        main_mask[atom] = False
    drop_main = game.utility(full_mask) - game.utility(main_mask)
    return float(drop_pair - drop_main)


def remove_pair_endpoints(mask: np.ndarray, edges: Iterable[tuple[int, int]]) -> np.ndarray:
    updated = np.asarray(mask, dtype=bool).copy()
    for left, right in edges:
        updated[left] = False
        updated[right] = False
    return updated


def to_markdown(payload: dict[str, object]) -> str:
    exact_summary = payload["exact_logic"]["summary"]
    mixture_summary = payload["motif_mixture"]["summary"]
    synergy_summary = payload["synergy_sweep"]["summary"]
    coupling_summary = payload["coupling_sweep"]["summary"]
    ablation_summary = payload["pair_probe_ablation"]["summary"]
    baseline_summary = payload["baseline_comparison"]["summary"]
    rescue_summary = payload["missing_modality_rescue"]["summary"]
    lines = [
        "# Structured Subset Attribution Experiment",
        "",
        "## Exact Logic",
        _metric_line("Utility R2", exact_summary["r2"]),
        _metric_line("Utility RMSE", exact_summary["rmse"]),
        _metric_line("Pair RMSE", exact_summary["pair_rmse"]),
        _metric_line("Pair Pearson", exact_summary["pair_pearson"]),
        _metric_line("Pair Spearman", exact_summary["pair_spearman"]),
        _metric_line("Pair SignAcc", exact_summary["pair_sign_acc"]),
        "",
        "## Motif Mixture",
        _metric_line("Utility R2", mixture_summary["r2"]),
        _metric_line("Utility RMSE", mixture_summary["rmse"]),
        _metric_line("Synergy AUROC", mixture_summary["auroc_synergy"]),
        _metric_line("Synergy AUPRC", mixture_summary["auprc_synergy"]),
        _metric_line("Macro F1", mixture_summary["macro_f1"]),
        _metric_line("Balanced Acc", mixture_summary["balanced_acc"]),
        _metric_line("Faithfulness@2", mixture_summary["faithfulness_at_2"]),
        _metric_line("PairGain@2", mixture_summary["pair_gain_at_2"]),
        "",
        "## Synergy Sweep",
        _metric_line("Corr(pi_S, positive mass)", synergy_summary["metrics"]["corr_pi_synergy_vs_positive_mass"]),
        _metric_line("Corr(exact, estimated mass)", synergy_summary["metrics"]["corr_exact_vs_estimated_mass"]),
        _metric_line("Monotonicity", synergy_summary["metrics"]["monotonicity"]),
        "",
        "## Coupling Sweep",
        _metric_line("Corr(coupling, positive mass)", coupling_summary["metrics"]["corr_coupling_vs_positive_mass"]),
        _metric_line("Corr(exact, estimated mass)", coupling_summary["metrics"]["corr_exact_vs_estimated_mass"]),
        _metric_line("Monotonicity", coupling_summary["metrics"]["monotonicity"]),
        "",
        "## Pair Probe Ablation",
        "### Structured without Pair Probes",
        _metric_line("Pair RMSE", ablation_summary["structured_no_pair_probes"]["pair_rmse"]),
        _metric_line("Synergy AUROC", ablation_summary["structured_no_pair_probes"]["auroc_synergy"]),
        _metric_line("Mass Abs Error", ablation_summary["structured_no_pair_probes"]["mass_abs_error"]),
        _metric_line("Sweep Corr(exact, estimated mass)", ablation_summary["structured_no_pair_probes"]["sweep_corr_exact_vs_estimated_mass"]),
        "",
        "### Structured with Pair Probes",
        _metric_line("Pair RMSE", ablation_summary["structured_pair_probes"]["pair_rmse"]),
        _metric_line("Synergy AUROC", ablation_summary["structured_pair_probes"]["auroc_synergy"]),
        _metric_line("Mass Abs Error", ablation_summary["structured_pair_probes"]["mass_abs_error"]),
        _metric_line("Sweep Corr(exact, estimated mass)", ablation_summary["structured_pair_probes"]["sweep_corr_exact_vs_estimated_mass"]),
        "",
        "## Baseline Comparison",
        "### Linear",
        _metric_line("Exact Pair RMSE", baseline_summary["linear"]["exact_pair_rmse"]),
        _metric_line("Mixture Pair RMSE", baseline_summary["linear"]["mixture_pair_rmse"]),
        _metric_line("Mixture AUROC", baseline_summary["linear"]["mixture_auroc_synergy"]),
        _metric_line("Sweep Corr(exact, estimated mass)", baseline_summary["linear"]["sweep_corr_exact_vs_estimated_mass"]),
        "",
        "### RBF Kernel",
        _metric_line("Exact Pair RMSE", baseline_summary["rbf"]["exact_pair_rmse"]),
        _metric_line("Mixture Pair RMSE", baseline_summary["rbf"]["mixture_pair_rmse"]),
        _metric_line("Mixture AUROC", baseline_summary["rbf"]["mixture_auroc_synergy"]),
        _metric_line("Sweep Corr(exact, estimated mass)", baseline_summary["rbf"]["sweep_corr_exact_vs_estimated_mass"]),
        "",
        "### Structured Second-Order + Pair Probes",
        _metric_line("Exact Pair RMSE", baseline_summary["structured_pair_probes"]["exact_pair_rmse"]),
        _metric_line("Mixture Pair RMSE", baseline_summary["structured_pair_probes"]["mixture_pair_rmse"]),
        _metric_line("Mixture AUROC", baseline_summary["structured_pair_probes"]["mixture_auroc_synergy"]),
        _metric_line("Sweep Corr(exact, estimated mass)", baseline_summary["structured_pair_probes"]["sweep_corr_exact_vs_estimated_mass"]),
        "",
        "## Missing-Modality Rescue",
        "### Structured without Pair Probes",
        _metric_line("Recall@k", rescue_summary["structured_no_pair_probes"]["rescue_recall_at_k"]),
        _metric_line("AUPRC", rescue_summary["structured_no_pair_probes"]["rescue_auprc"]),
        _metric_line("Corr(missing rate, mass)", rescue_summary["structured_no_pair_probes"]["missing_sweep_corr_missing_rate_vs_positive_mass"]),
        _metric_line("Corr(partner purity, mass)", rescue_summary["structured_no_pair_probes"]["partner_sweep_corr_partner_purity_vs_positive_mass"]),
        "",
        "### Structured with Pair Probes",
        _metric_line("Recall@k", rescue_summary["structured_pair_probes"]["rescue_recall_at_k"]),
        _metric_line("AUPRC", rescue_summary["structured_pair_probes"]["rescue_auprc"]),
        _metric_line("Corr(missing rate, mass)", rescue_summary["structured_pair_probes"]["missing_sweep_corr_missing_rate_vs_positive_mass"]),
        _metric_line("Corr(partner purity, mass)", rescue_summary["structured_pair_probes"]["partner_sweep_corr_partner_purity_vs_positive_mass"]),
        "",
    ]
    return "\n".join(lines)


def _metric_line(name: str, summary: dict[str, float]) -> str:
    return f"- {name}: {summary['mean']:.4f} ± {summary['std']:.4f}"


if __name__ == "__main__":
    main()
