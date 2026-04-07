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
    parser = argparse.ArgumentParser(description="IMDb Fisher analysis across epochs and layers.")
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
    parser.add_argument("--capture-epochs", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-json", type=str, default=str(output_dir / "imdb_fisher_epoch_layer.json"))
    return parser.parse_args()


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def _quartile_delta(values: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    q1 = np.quantile(values, 0.25)
    q3 = np.quantile(values, 0.75)
    low = targets[values <= q1]
    high = targets[values >= q3]
    if low.size == 0 or high.size == 0:
        return {
            "low_mean": float("nan"),
            "high_mean": float("nan"),
            "delta": float("nan"),
        }
    return {
        "low_mean": float(low.mean()),
        "high_mean": float(high.mean()),
        "delta": float(high.mean() - low.mean()),
    }


def _auc(values: np.ndarray, targets: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=np.int64)
    if np.unique(targets).size < 2:
        return float("nan")
    return float(roc_auc_score(targets, np.asarray(values, dtype=np.float32)))


def _mean_std(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(finite) == 1:
        return {"mean": finite[0], "std": 0.0, "n": 1}
    mean = sum(finite) / len(finite)
    var = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return {"mean": mean, "std": math.sqrt(max(var, 0.0)), "n": len(finite)}


def _layerwise_fisher(opt_alignment: dict[str, Any]) -> dict[str, np.ndarray]:
    fisher: dict[str, np.ndarray] = {}
    for layer_name in opt_alignment["sample_layer_gradcos"]:
        gradcos = np.asarray(opt_alignment["sample_layer_gradcos"][layer_name], dtype=np.float32)
        curvagree = np.asarray(opt_alignment["sample_layer_curvagree"][layer_name], dtype=np.float32)
        fisher[layer_name] = (0.5 * (_zscore(gradcos) + _zscore(curvagree))).astype(np.float32)
    return fisher


def _layer_metrics(layer_fisher: dict[str, np.ndarray], targets: np.ndarray) -> dict[str, dict[str, float]]:
    outputs: dict[str, dict[str, float]] = {}
    targets = np.asarray(targets, dtype=np.float32)
    for layer_name, fisher in layer_fisher.items():
        corr = stats.spearmanr(fisher, targets)
        quartile = _quartile_delta(fisher, targets)
        outputs[layer_name] = {
            "auc": _auc(fisher, targets.astype(np.int64)),
            "rho": float(corr.statistic),
            "rho_p_value": float(corr.pvalue),
            "quartile_delta": float(quartile["delta"]),
            "quartile_low_mean": float(quartile["low_mean"]),
            "quartile_high_mean": float(quartile["high_mean"]),
            "fisher_mean": float(np.mean(fisher)),
            "fisher_std": float(np.std(fisher)),
        }
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
    layer_fisher = _layerwise_fisher(opt_alignment)
    correctness = _correctness_from_probs(dataset.task, evaluation["joint_probs"], split.labels).astype(np.float32)
    return layer_fisher, correctness


def main() -> None:
    args = parse_args()
    checkpoint_order = [f"epoch_{epoch}" for epoch in args.capture_epochs] + ["best"]
    per_seed: list[dict[str, Any]] = []
    aggregate_values: dict[str, dict[str, dict[str, list[float]]]] = {
        target: {checkpoint: {} for checkpoint in checkpoint_order}
        for target in ("current_correctness", "final_correctness")
    }
    layer_names: list[str] | None = None

    for seed in args.seeds:
        dataset_seed = load_imdb(
            args.imdb_path,
            max_samples=args.max_samples,
            train_max_samples=args.train_max_samples,
            valid_max_samples=args.valid_max_samples,
            test_max_samples=args.test_max_samples,
            text_dropout_prob=args.imdb_text_dropout_prob,
            text_dropout_seed=seed + 17,
        )
        split_seed = {"train": dataset_seed.train, "valid": dataset_seed.valid, "test": dataset_seed.test}[args.analysis_split]
        _, train_metrics, checkpoints = train_joint_model_with_snapshots(
            dataset_seed,
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
            capture_epochs=args.capture_epochs,
        )

        probe_model = build_joint_model(
            dataset_seed,
            encoder_hidden_dim=args.encoder_hidden_dim,
            fusion_dim=args.fusion_dim,
            fusion_layers=args.fusion_layers,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
        ).to(args.device)

        final_correctness: np.ndarray | None = None
        per_checkpoint: dict[str, Any] = {}
        checkpoint_cache: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}

        for checkpoint_name in checkpoint_order:
            if checkpoint_name not in checkpoints:
                continue
            probe_model.load_state_dict(checkpoints[checkpoint_name])
            fisher_by_layer, current_correctness = _evaluate_checkpoint(
                dataset_seed,
                split_seed,
                probe_model,
                device=args.device,
                eval_batch_size=args.eval_batch_size,
                top_k_subspace=args.top_k_subspace,
            )
            checkpoint_cache[checkpoint_name] = (fisher_by_layer, current_correctness)
            if layer_names is None:
                layer_names = list(fisher_by_layer.keys())
            current_metrics = _layer_metrics(fisher_by_layer, current_correctness)
            per_checkpoint[checkpoint_name] = {
                "current_correctness": current_metrics,
                "current_accuracy": float(np.mean(current_correctness)),
            }
            if checkpoint_name == "best":
                final_correctness = current_correctness

        if final_correctness is None:
            raise RuntimeError(f"Missing best checkpoint for seed {seed}")

        for checkpoint_name, checkpoint_payload in per_checkpoint.items():
            fisher_by_layer, _ = checkpoint_cache[checkpoint_name]
            final_metrics = _layer_metrics(fisher_by_layer, final_correctness)
            checkpoint_payload["final_correctness"] = final_metrics
            for target_name in ("current_correctness", "final_correctness"):
                metrics_by_layer = checkpoint_payload[target_name]
                for layer_name_local, metrics in metrics_by_layer.items():
                    layer_bucket = aggregate_values[target_name][checkpoint_name].setdefault(layer_name_local, {})
                    for metric_name, value in metrics.items():
                        layer_bucket.setdefault(metric_name, []).append(float(value))

        per_seed.append(
            {
                "seed": seed,
                "sample_ids": [str(item) for item in split_seed.ids],
                "best_epoch": int(train_metrics["best_epoch"]),
                "stopped_epoch": int(train_metrics["stopped_epoch"]),
                "checkpoint_order": [name for name in checkpoint_order if name in per_checkpoint],
                "layer_names": layer_names,
                "checkpoints": per_checkpoint,
            }
        )

    aggregate: dict[str, Any] = {}
    for target_name, checkpoints_for_target in aggregate_values.items():
        aggregate[target_name] = {}
        for checkpoint_name, layers_for_checkpoint in checkpoints_for_target.items():
            if not layers_for_checkpoint:
                continue
            aggregate[target_name][checkpoint_name] = {}
            for layer_name_local, metrics in layers_for_checkpoint.items():
                aggregate[target_name][checkpoint_name][layer_name_local] = {
                    metric_name: _mean_std(metric_values) for metric_name, metric_values in metrics.items()
                }

    write_json(
        {
            "dataset": "imdb",
            "analysis_split": args.analysis_split,
            "seeds": args.seeds,
            "capture_epochs": args.capture_epochs,
            "layer_names": layer_names,
            "per_seed": per_seed,
            "aggregate": aggregate,
        },
        Path(args.output_json),
    )


if __name__ == "__main__":
    main()
