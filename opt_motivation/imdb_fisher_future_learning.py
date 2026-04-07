from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from motivation.core import write_json
from opt_motivation import default_output_dir
from opt_motivation.core import (
    _correctness_from_probs,
    auto_device,
    build_joint_model,
    collect_gradient_signals,
    compute_optimization_alignment,
    evaluate_joint_model,
    load_imdb,
    train_joint_model_with_snapshots,
)


def parse_args() -> argparse.Namespace:
    output_dir = default_output_dir()
    parser = argparse.ArgumentParser(description="IMDb early-Fisher future-learning analysis.")
    parser.add_argument("--imdb-path", type=str, default="/data/zhenshuo/datasets/multimodal_imdb.hdf5")
    parser.add_argument("--analysis-split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--valid-max-samples", type=int, default=None)
    parser.add_argument("--test-max-samples", type=int, default=None)
    parser.add_argument("--imdb-text-dropout-prob", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=auto_device())
    parser.add_argument("--encoder-hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-dim", type=int, default=128)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--top-k-subspace", type=int, default=10)
    parser.add_argument("--early-epoch", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-json", type=str, default=str(output_dir / "imdb_fisher_future_learning.json"))
    return parser.parse_args()


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def _layerwise_fisher(opt_alignment: dict[str, Any]) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for layer_name in opt_alignment["sample_layer_gradcos"]:
        gradcos = np.asarray(opt_alignment["sample_layer_gradcos"][layer_name], dtype=np.float32)
        curvagree = np.asarray(opt_alignment["sample_layer_curvagree"][layer_name], dtype=np.float32)
        outputs[layer_name] = (0.5 * (_zscore(gradcos) + _zscore(curvagree))).astype(np.float32)
    return outputs


def _evaluate_checkpoint(
    dataset: Any,
    split: Any,
    model: Any,
    *,
    device: str,
    eval_batch_size: int,
    top_k_subspace: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    evaluation = evaluate_joint_model(dataset, split, model, device=device, batch_size=eval_batch_size)
    gradients = collect_gradient_signals(dataset, split, model, device=device, batch_size=eval_batch_size)
    opt_alignment = compute_optimization_alignment(
        gradients,
        evaluation["layer_representations"],
        modality_names=dataset.modality_names,
        top_k=top_k_subspace,
        neighborhood_k=max(2 * top_k_subspace, 25),
    )
    fisher_by_layer = _layerwise_fisher(opt_alignment)
    correctness = _correctness_from_probs(dataset.task, evaluation["joint_probs"], split.labels).astype(np.int64)
    return fisher_by_layer, correctness


def _quartile_delta(scores: np.ndarray, target: np.ndarray) -> dict[str, float]:
    q1 = np.quantile(scores, 0.25)
    q3 = np.quantile(scores, 0.75)
    low = target[scores <= q1]
    high = target[scores >= q3]
    if low.size == 0 or high.size == 0:
        return {"low_mean": float("nan"), "high_mean": float("nan"), "delta": float("nan")}
    return {
        "low_mean": float(low.mean()),
        "high_mean": float(high.mean()),
        "delta": float(high.mean() - low.mean()),
    }


def _binary_metrics(scores: np.ndarray, target: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float32)
    target = np.asarray(target, dtype=np.int64)
    quartile = _quartile_delta(scores, target.astype(np.float32))
    if scores.size < 3 or np.unique(target).size < 2:
        return {
            "auc": float("nan"),
            "rho": float("nan"),
            "rho_p_value": float("nan"),
            "quartile_delta": quartile["delta"],
            "quartile_low_mean": quartile["low_mean"],
            "quartile_high_mean": quartile["high_mean"],
        }
    corr = stats.spearmanr(scores, target)
    return {
        "auc": float(roc_auc_score(target, scores)),
        "rho": float(corr.statistic),
        "rho_p_value": float(corr.pvalue),
        "quartile_delta": quartile["delta"],
        "quartile_low_mean": quartile["low_mean"],
        "quartile_high_mean": quartile["high_mean"],
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(finite) == 1:
        return {"mean": finite[0], "std": 0.0, "n": 1}
    mean = sum(finite) / len(finite)
    var = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return {"mean": mean, "std": math.sqrt(max(var, 0.0)), "n": len(finite)}


def main() -> None:
    args = parse_args()
    early_name = f"epoch_{args.early_epoch}"
    per_seed: list[dict[str, Any]] = []
    aggregate_rows: dict[str, dict[str, dict[str, list[float]]]] = {
        "all_samples_final_correctness": {},
        "all_samples_current_correctness": {},
        "incorrect_subset_future_success": {},
    }
    layer_names: list[str] | None = None

    for seed in args.seeds:
        dataset = load_imdb(
            args.imdb_path,
            max_samples=args.max_samples,
            train_max_samples=args.train_max_samples,
            valid_max_samples=args.valid_max_samples,
            test_max_samples=args.test_max_samples,
            text_dropout_prob=args.imdb_text_dropout_prob,
            text_dropout_seed=seed + 17,
        )
        split = {"train": dataset.train, "valid": dataset.valid, "test": dataset.test}[args.analysis_split]
        _, train_metrics, checkpoints = train_joint_model_with_snapshots(
            dataset,
            device=args.device,
            encoder_hidden_dim=args.encoder_hidden_dim,
            fusion_dim=args.fusion_dim,
            fusion_layers=args.fusion_layers,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            seed=seed,
            capture_epochs=[args.early_epoch],
        )
        if early_name not in checkpoints:
            raise RuntimeError(f"Missing {early_name} checkpoint for seed {seed}")

        model = build_joint_model(
            dataset,
            encoder_hidden_dim=args.encoder_hidden_dim,
            fusion_dim=args.fusion_dim,
            fusion_layers=args.fusion_layers,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
        ).to(args.device)

        model.load_state_dict(checkpoints[early_name])
        early_fisher, early_correctness = _evaluate_checkpoint(
            dataset,
            split,
            model,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            top_k_subspace=args.top_k_subspace,
        )
        model.load_state_dict(checkpoints["best"])
        _, best_correctness = _evaluate_checkpoint(
            dataset,
            split,
            model,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            top_k_subspace=args.top_k_subspace,
        )

        if layer_names is None:
            layer_names = list(early_fisher.keys())

        incorrect_mask = early_correctness == 0
        seed_payload = {
            "seed": seed,
            "best_epoch": int(train_metrics["best_epoch"]),
            "early_epoch": args.early_epoch,
            "early_accuracy": float(early_correctness.mean()),
            "best_accuracy": float(best_correctness.mean()),
            "incorrect_subset_size": int(incorrect_mask.sum()),
            "metrics": {
                "all_samples_final_correctness": {},
                "all_samples_current_correctness": {},
                "incorrect_subset_future_success": {},
            },
        }

        for layer_name, fisher in early_fisher.items():
            all_metrics = _binary_metrics(fisher, best_correctness)
            current_metrics = _binary_metrics(fisher, early_correctness)
            future_metrics = _binary_metrics(fisher[incorrect_mask], best_correctness[incorrect_mask])
            seed_payload["metrics"]["all_samples_final_correctness"][layer_name] = all_metrics
            seed_payload["metrics"]["all_samples_current_correctness"][layer_name] = current_metrics
            seed_payload["metrics"]["incorrect_subset_future_success"][layer_name] = future_metrics
            for group_name, metrics in (
                ("all_samples_final_correctness", all_metrics),
                ("all_samples_current_correctness", current_metrics),
                ("incorrect_subset_future_success", future_metrics),
            ):
                layer_bucket = aggregate_rows[group_name].setdefault(layer_name, {})
                for metric_name, value in metrics.items():
                    layer_bucket.setdefault(metric_name, []).append(float(value))

        per_seed.append(seed_payload)

    aggregate: dict[str, Any] = {}
    for group_name, layers in aggregate_rows.items():
        aggregate[group_name] = {}
        for layer_name, metrics in layers.items():
            aggregate[group_name][layer_name] = {
                metric_name: _mean_std(values) for metric_name, values in metrics.items()
            }

    write_json(
        {
            "dataset": "imdb",
            "analysis_split": args.analysis_split,
            "seeds": args.seeds,
            "early_epoch": args.early_epoch,
            "layer_names": layer_names,
            "per_seed": per_seed,
            "aggregate": aggregate,
        },
        Path(args.output_json),
    )


if __name__ == "__main__":
    main()
