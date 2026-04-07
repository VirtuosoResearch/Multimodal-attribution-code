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
    _sample_quality_from_probs,
    auto_device,
    collect_gradient_signals,
    compute_optimization_alignment,
    evaluate_joint_model,
    load_affect,
    load_imdb,
    train_joint_model,
)


def parse_args() -> argparse.Namespace:
    output_dir = default_output_dir()
    parser = argparse.ArgumentParser(description="Fisher intervention analysis with disrupted cross-modal interaction.")
    parser.add_argument("--dataset", type=str, default="imdb", choices=["imdb", "humor", "sarcasm"])
    parser.add_argument("--imdb-path", type=str, default="/data/zhenshuo/datasets/multimodal_imdb.hdf5")
    parser.add_argument("--humor-path", type=str, default="/data/zhenshuo/datasets/humor/humor.pkl")
    parser.add_argument("--sarcasm-path", type=str, default="/data/zhenshuo/datasets/sarcasm/sarcasm.pkl")
    parser.add_argument("--analysis-split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--valid-max-samples", type=int, default=None)
    parser.add_argument("--test-max-samples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=50)
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-json", type=str, default=str(output_dir / "fisher_intervention.json"))
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


def _quartile_delta(scores: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    q1 = np.quantile(scores, 0.25)
    q3 = np.quantile(scores, 0.75)
    low = targets[scores <= q1]
    high = targets[scores >= q3]
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


def _continuous_metrics(scores: np.ndarray, target: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    quartile = _quartile_delta(scores, target)
    corr = stats.spearmanr(scores, target)
    return {
        "rho": float(corr.statistic),
        "rho_p_value": float(corr.pvalue),
        "quartile_delta": quartile["delta"],
        "quartile_low_mean": quartile["low_mean"],
        "quartile_high_mean": quartile["high_mean"],
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)),
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
    per_seed: list[dict[str, Any]] = []
    aggregate_rows: dict[str, dict[str, dict[str, list[float]]]] = {
        "interaction_sensitive": {},
        "interaction_critical": {},
        "correctness_drop": {},
        "negative_loss_drop": {},
    }
    layer_names: list[str] | None = None

    for seed in args.seeds:
        if args.dataset == "imdb":
            dataset = load_imdb(
                args.imdb_path,
                max_samples=args.max_samples,
                train_max_samples=args.train_max_samples,
                valid_max_samples=args.valid_max_samples,
                test_max_samples=args.test_max_samples,
                text_dropout_prob=args.imdb_text_dropout_prob,
                text_dropout_seed=seed + 17,
            )
        elif args.dataset == "humor":
            dataset = load_affect(args.humor_path, dataset_name="humor", max_seq_len=args.max_seq_len)
        elif args.dataset == "sarcasm":
            dataset = load_affect(args.sarcasm_path, dataset_name="sarcasm", max_seq_len=args.max_seq_len)
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        split = {"train": dataset.train, "valid": dataset.valid, "test": dataset.test}[args.analysis_split]

        full_model, full_train = train_joint_model(
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
            cross_modal_enabled=True,
        )
        disrupted_model, disrupted_train = train_joint_model(
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
            cross_modal_enabled=False,
        )

        full_eval = evaluate_joint_model(dataset, split, full_model, device=args.device, batch_size=args.eval_batch_size)
        disrupted_eval = evaluate_joint_model(dataset, split, disrupted_model, device=args.device, batch_size=args.eval_batch_size)
        gradients = collect_gradient_signals(dataset, split, full_model, device=args.device, batch_size=args.eval_batch_size)
        opt_alignment = compute_optimization_alignment(
            gradients,
            full_eval["layer_representations"],
            modality_names=dataset.modality_names,
            top_k=args.top_k_subspace,
            neighborhood_k=max(2 * args.top_k_subspace, 25),
        )
        fisher_by_layer = _layerwise_fisher(opt_alignment)
        if layer_names is None:
            layer_names = list(fisher_by_layer.keys())

        full_correctness = _correctness_from_probs(dataset.task, full_eval["joint_probs"], split.labels).astype(np.int64)
        disrupted_correctness = _correctness_from_probs(dataset.task, disrupted_eval["joint_probs"], split.labels).astype(np.int64)
        interaction_sensitive = (full_correctness != disrupted_correctness).astype(np.int64)
        interaction_critical = ((full_correctness == 1) & (disrupted_correctness == 0)).astype(np.int64)
        correctness_drop = (full_correctness - disrupted_correctness).astype(np.float32)

        full_quality = _sample_quality_from_probs(dataset.task, full_eval["joint_probs"], split.labels)
        disrupted_quality = _sample_quality_from_probs(dataset.task, disrupted_eval["joint_probs"], split.labels)
        negative_loss_drop = (full_quality["negative_loss"] - disrupted_quality["negative_loss"]).astype(np.float32)

        seed_payload = {
            "seed": seed,
            "sample_ids": [str(item) for item in split.ids],
            "full_best_valid_loss": float(full_train["best_valid_loss"]),
            "disrupted_best_valid_loss": float(disrupted_train["best_valid_loss"]),
            "full_accuracy": float(full_correctness.mean()),
            "disrupted_accuracy": float(disrupted_correctness.mean()),
            "interaction_sensitive_rate": float(interaction_sensitive.mean()),
            "interaction_critical_rate": float(interaction_critical.mean()),
            "metrics": {
                "interaction_sensitive": {},
                "interaction_critical": {},
                "correctness_drop": {},
                "negative_loss_drop": {},
            },
        }

        for layer_name, fisher in fisher_by_layer.items():
            metrics_sensitive = _binary_metrics(fisher, interaction_sensitive)
            metrics_binary = _binary_metrics(fisher, interaction_critical)
            metrics_drop = _continuous_metrics(fisher, correctness_drop)
            metrics_loss = _continuous_metrics(fisher, negative_loss_drop)
            seed_payload["metrics"]["interaction_sensitive"][layer_name] = metrics_sensitive
            seed_payload["metrics"]["interaction_critical"][layer_name] = metrics_binary
            seed_payload["metrics"]["correctness_drop"][layer_name] = metrics_drop
            seed_payload["metrics"]["negative_loss_drop"][layer_name] = metrics_loss
            for group_name, metrics in (
                ("interaction_sensitive", metrics_sensitive),
                ("interaction_critical", metrics_binary),
                ("correctness_drop", metrics_drop),
                ("negative_loss_drop", metrics_loss),
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
            "dataset": args.dataset,
            "analysis_split": args.analysis_split,
            "seeds": args.seeds,
            "layer_names": layer_names,
            "per_seed": per_seed,
            "aggregate": aggregate,
        },
        Path(args.output_json),
    )


if __name__ == "__main__":
    main()
