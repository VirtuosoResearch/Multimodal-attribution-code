from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats

from motivation.core import write_json
from opt_motivation import default_output_dir
from opt_motivation.core import (
    _confidence_for_targets,
    _correctness_from_probs,
    _sample_quality_from_probs,
    auto_device,
    collect_empirical_fisher_signals,
    compute_conditional_predictive_power,
    compute_interaction_buckets,
    compute_matched_bin_analysis,
    evaluate_models,
    load_affect,
    load_imdb,
    load_mosi,
    train_joint_model,
    train_unimodal_models,
)


def parse_args() -> argparse.Namespace:
    output_dir = default_output_dir()
    parser = argparse.ArgumentParser(description="Analyze multimodal interaction with empirical parameter Fisher features.")
    parser.add_argument("--dataset", type=str, default="imdb", choices=["mosi", "imdb", "humor", "sarcasm"])
    parser.add_argument("--mosi-path", type=str, default="/data/zhenshuo/datasets/mosi/mosi_raw.pkl")
    parser.add_argument("--imdb-path", type=str, default="/data/zhenshuo/datasets/multimodal_imdb.hdf5")
    parser.add_argument("--humor-path", type=str, default="/data/zhenshuo/datasets/humor/humor.pkl")
    parser.add_argument("--sarcasm-path", type=str, default="/data/zhenshuo/datasets/sarcasm/sarcasm.pkl")
    parser.add_argument("--analysis-split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--max-seq-len", type=int, default=50)
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
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--pid-source", type=str, default="auto", choices=["auto", "confidence", "lsmi"])
    parser.add_argument("--num-pid-bins", type=int, default=5)
    parser.add_argument("--bucket-margin", type=float, default=0.0)
    parser.add_argument("--parameter-scope", type=str, default="shared", choices=["shared", "representation", "head", "classifier", "all"])
    parser.add_argument("--architecture", type=str, default="shared_head", choices=["shared_head", "cross_attention"])
    parser.add_argument("--num-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def _load_dataset(args: argparse.Namespace):
    if args.dataset == "mosi":
        return load_mosi(args.mosi_path, max_seq_len=args.max_seq_len)
    if args.dataset == "imdb":
        return load_imdb(
            args.imdb_path,
            max_samples=args.max_samples,
            train_max_samples=args.train_max_samples,
            valid_max_samples=args.valid_max_samples,
            test_max_samples=args.test_max_samples,
            text_dropout_prob=args.imdb_text_dropout_prob,
            text_dropout_seed=args.seed + 17,
        )
    if args.dataset == "humor":
        return load_affect(args.humor_path, dataset_name="humor", max_seq_len=args.max_seq_len)
    if args.dataset == "sarcasm":
        return load_affect(args.sarcasm_path, dataset_name="sarcasm", max_seq_len=args.max_seq_len)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def _epochs_for_dataset(args: argparse.Namespace) -> int:
    if args.epochs is not None:
        return int(args.epochs)
    return 10 if args.dataset == "imdb" else 50


def _extract_pid_features(
    dataset: Any,
    split: Any,
    interactions: dict[str, Any],
    joint_probs: np.ndarray,
    unimodal_probs: dict[str, np.ndarray],
    *,
    pid_source: str,
) -> tuple[str, dict[str, np.ndarray], np.ndarray]:
    labels = split.labels
    confidence_pid_redundancy = np.asarray(interactions["scores"]["redundancy"], dtype=np.float32)
    confidence_pid_synergy = np.asarray(interactions["scores"]["synergy"], dtype=np.float32)
    confidence_unique_matrix = np.stack(
        [np.asarray(values, dtype=np.float32) for values in interactions["scores"]["uniqueness"].values()],
        axis=1,
    )
    confidence_pid_best_unique = np.max(confidence_unique_matrix, axis=1).astype(np.float32)
    pid_source_used = "confidence"
    pid_feature_dict: dict[str, np.ndarray] = {
        "redundancy": confidence_pid_redundancy,
        "synergy": confidence_pid_synergy,
        "best_unique": confidence_pid_best_unique,
    }
    if pid_source in {"auto", "lsmi"} and interactions["lsmi"] is not None:
        lsmi = interactions["lsmi"]
        lsmi_redundancy = np.asarray(lsmi["redundancy"], dtype=np.float32)
        lsmi_synergy = np.asarray(lsmi["synergy"], dtype=np.float32)
        if "unique_left" in lsmi and "unique_right" in lsmi:
            pid_feature_dict = {
                "redundancy": lsmi_redundancy,
                "unique_left": np.asarray(lsmi["unique_left"], dtype=np.float32),
                "unique_right": np.asarray(lsmi["unique_right"], dtype=np.float32),
                "synergy": lsmi_synergy,
            }
            pid_source_used = "lsmi"
        else:
            unique_feature_names = sorted(
                [name for name in lsmi.keys() if name.startswith("unique_") and name != "unique_modality"]
            )
            pid_feature_dict = {"redundancy": lsmi_redundancy}
            for name in unique_feature_names:
                pid_feature_dict[name] = np.asarray(lsmi[name], dtype=np.float32)
            pid_feature_dict["synergy"] = lsmi_synergy
            pid_source_used = "lsmi_pairwise"
    pid_feature_names = list(pid_feature_dict.keys())
    pid_feature_matrix = np.stack([pid_feature_dict[name] for name in pid_feature_names], axis=1).astype(np.float32)
    return pid_source_used, pid_feature_dict, pid_feature_matrix


def _bucket_summary(buckets: np.ndarray, fisher_measure: np.ndarray) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for bucket in sorted(set(buckets.tolist())):
        mask = buckets == bucket
        values = fisher_measure[mask]
        output[str(bucket)] = {
            "count": int(mask.sum()),
            "mean": float(np.mean(values)) if values.size > 0 else float("nan"),
            "std": float(np.std(values)) if values.size > 0 else float("nan"),
        }
    return output


def _spearman_summary(values: np.ndarray, target: np.ndarray) -> dict[str, float]:
    result = stats.spearmanr(values, target)
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _single_feature_conditional_power(
    *,
    pid_features: np.ndarray,
    features: dict[str, np.ndarray],
    correctness: np.ndarray,
    negative_loss: np.ndarray,
    negative_loss_advantage: np.ndarray,
    correctness_advantage: np.ndarray,
    beats_best_unimodal: np.ndarray,
    seed: int,
    num_bootstrap: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for index, (name, values) in enumerate(features.items(), start=1):
        output[name] = compute_conditional_predictive_power(
            pid_features=pid_features,
            fisher_features=np.asarray(values, dtype=np.float32).reshape(-1, 1),
            correctness=correctness,
            negative_loss=negative_loss,
            negative_loss_advantage=negative_loss_advantage,
            correctness_advantage=correctness_advantage,
            beats_best_unimodal=beats_best_unimodal,
            seed=seed + index * 10000,
            num_bootstrap=num_bootstrap,
        )
    return output


def main() -> None:
    args = parse_args()
    output_dir = default_output_dir()
    output_json = Path(args.output_json) if args.output_json else output_dir / f"{args.dataset}_empirical_fisher_interaction.json"

    dataset = _load_dataset(args)
    split = {"train": dataset.train, "valid": dataset.valid, "test": dataset.test}[args.analysis_split]
    epochs = _epochs_for_dataset(args)

    joint_model, joint_train = train_joint_model(
        dataset,
        device=args.device,
        encoder_hidden_dim=args.encoder_hidden_dim,
        fusion_dim=args.fusion_dim,
        fusion_layers=args.fusion_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=epochs,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        architecture=args.architecture,
    )
    unimodal_models, unimodal_train = train_unimodal_models(
        dataset,
        device=args.device,
        encoder_hidden_dim=args.encoder_hidden_dim,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=epochs,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed + 1000,
    )
    evaluation = evaluate_models(dataset, split, joint_model, unimodal_models, device=args.device, batch_size=args.eval_batch_size)
    interactions = compute_interaction_buckets(
        dataset,
        split,
        evaluation["joint_probs"],
        evaluation["unimodal_probs"],
        evaluation["layer_representations"],
        device=args.device,
        bucket_margin=args.bucket_margin,
        seed=args.seed + 2000,
    )
    empirical_fisher = collect_empirical_fisher_signals(
        dataset,
        split,
        joint_model,
        device=args.device,
        batch_size=args.eval_batch_size,
        parameter_scope=args.parameter_scope,
    )

    pid_source_used, pid_feature_dict, pid_feature_matrix = _extract_pid_features(
        dataset,
        split,
        interactions,
        evaluation["joint_probs"],
        evaluation["unimodal_probs"],
        pid_source=args.pid_source,
    )

    correctness = _correctness_from_probs(dataset.task, evaluation["joint_probs"], split.labels)
    joint_quality = _sample_quality_from_probs(dataset.task, evaluation["joint_probs"], split.labels)
    unimodal_quality = {
        name: _sample_quality_from_probs(dataset.task, evaluation["unimodal_probs"][name], split.labels) for name in dataset.modality_names
    }
    unimodal_correctness = np.stack(
        [_correctness_from_probs(dataset.task, evaluation["unimodal_probs"][name], split.labels) for name in dataset.modality_names],
        axis=1,
    )
    unimodal_conf = np.stack(
        [_confidence_for_targets(dataset.task, evaluation["unimodal_probs"][name], split.labels) for name in dataset.modality_names],
        axis=1,
    )
    joint_gain = (joint_quality["confidence"] - unimodal_conf.max(axis=1)).astype(np.float32)
    unimodal_negative_loss = np.stack([unimodal_quality[name]["negative_loss"] for name in dataset.modality_names], axis=1)
    negative_loss_advantage = (joint_quality["negative_loss"] - unimodal_negative_loss.max(axis=1)).astype(np.float32)
    correctness_advantage = (correctness - unimodal_correctness.max(axis=1)).astype(np.float32)
    beats_best_unimodal = (correctness_advantage > 0.0).astype(np.int64)

    fisher_trace = np.asarray(empirical_fisher["sample_trace_mean"], dtype=np.float32)
    fisher_inner = np.asarray(empirical_fisher["sample_pair_inner_mean"], dtype=np.float32)
    fisher_inner_pos = np.asarray(empirical_fisher["sample_pair_inner_pos_mean"], dtype=np.float32)
    fisher_inner_neg = np.asarray(empirical_fisher["sample_pair_inner_neg_mean"], dtype=np.float32)
    fisher_cos = np.asarray(empirical_fisher["sample_pair_cos_mean"], dtype=np.float32)
    fisher_alignment_score = (0.5 * (_zscore(fisher_inner_pos) + _zscore(np.clip(fisher_cos, a_min=0.0, a_max=None)))).astype(np.float32)
    fisher_conflict_score = (0.5 * (_zscore(fisher_inner_neg) + _zscore(np.clip(-fisher_cos, a_min=0.0, a_max=None)))).astype(np.float32)
    empirical_fisher_features = np.stack(
        [fisher_trace, fisher_inner, fisher_inner_pos, fisher_inner_neg, fisher_cos],
        axis=1,
    ).astype(np.float32)
    empirical_feature_dict = {
        "trace_mean": fisher_trace,
        "pair_inner_mean": fisher_inner,
        "pair_inner_pos_mean": fisher_inner_pos,
        "pair_inner_neg_mean": fisher_inner_neg,
        "pair_cos_mean": fisher_cos,
        "alignment_score": fisher_alignment_score,
        "conflict_score": fisher_conflict_score,
    }

    valid_mask = np.isin(interactions["buckets"], ["RED", "SYN", "UNQ"])
    valid_mask &= np.isfinite(pid_feature_matrix).all(axis=1)
    valid_mask &= np.isfinite(empirical_fisher_features).all(axis=1)

    conditional_predictive_power = compute_conditional_predictive_power(
        pid_features=pid_feature_matrix[valid_mask],
        fisher_features=empirical_fisher_features[valid_mask],
        correctness=correctness[valid_mask],
        negative_loss=joint_quality["negative_loss"][valid_mask],
        negative_loss_advantage=negative_loss_advantage[valid_mask],
        correctness_advantage=correctness_advantage[valid_mask],
        beats_best_unimodal=beats_best_unimodal[valid_mask],
        seed=args.seed + 5000,
        num_bootstrap=args.num_bootstrap,
    )
    matched_bin_alignment = compute_matched_bin_analysis(
        pid_features=pid_feature_matrix[valid_mask],
        fisher_score=fisher_alignment_score[valid_mask],
        negative_loss_advantage=negative_loss_advantage[valid_mask],
        correctness=correctness[valid_mask],
        beats_best_unimodal=beats_best_unimodal[valid_mask],
        num_bins=args.num_pid_bins,
        seed=args.seed + 6000,
    )
    matched_bin_conflict = compute_matched_bin_analysis(
        pid_features=pid_feature_matrix[valid_mask],
        fisher_score=fisher_conflict_score[valid_mask],
        negative_loss_advantage=negative_loss_advantage[valid_mask],
        correctness=correctness[valid_mask],
        beats_best_unimodal=beats_best_unimodal[valid_mask],
        num_bins=args.num_pid_bins,
        seed=args.seed + 7000,
    )

    feature_spearman = {
        name: {
            "vs_pid_synergy": _spearman_summary(values[valid_mask], pid_feature_matrix[valid_mask, -1]),
            "vs_correctness": _spearman_summary(values[valid_mask], correctness[valid_mask]),
        }
        for name, values in empirical_feature_dict.items()
    }
    feature_bucket_summary = {
        name: _bucket_summary(np.asarray(interactions["buckets"], dtype=object), values) for name, values in empirical_feature_dict.items()
    }
    single_feature_power = _single_feature_conditional_power(
        pid_features=pid_feature_matrix[valid_mask],
        features={name: values[valid_mask] for name, values in empirical_feature_dict.items()},
        correctness=correctness[valid_mask],
        negative_loss=joint_quality["negative_loss"][valid_mask],
        negative_loss_advantage=negative_loss_advantage[valid_mask],
        correctness_advantage=correctness_advantage[valid_mask],
        beats_best_unimodal=beats_best_unimodal[valid_mask],
        seed=args.seed + 8000,
        num_bootstrap=args.num_bootstrap,
    )

    write_json(
        {
            "dataset": args.dataset,
            "analysis_split": args.analysis_split,
            "seed": args.seed,
            "architecture": args.architecture,
            "parameter_scope": args.parameter_scope,
            "training": {"joint": joint_train, "unimodal": unimodal_train},
            "pid_source_used": pid_source_used,
            "empirical_fisher": {
                "summary": empirical_fisher["summary"],
                "feature_names": ["trace_mean", "pair_inner_mean", "pair_inner_pos_mean", "pair_inner_neg_mean", "pair_cos_mean"],
                "feature_values": {
                    "trace_mean": fisher_trace.astype(float).tolist(),
                    "pair_inner_mean": fisher_inner.astype(float).tolist(),
                    "pair_inner_pos_mean": fisher_inner_pos.astype(float).tolist(),
                    "pair_inner_neg_mean": fisher_inner_neg.astype(float).tolist(),
                    "pair_cos_mean": fisher_cos.astype(float).tolist(),
                },
                "alignment_score": fisher_alignment_score.astype(float).tolist(),
                "conflict_score": fisher_conflict_score.astype(float).tolist(),
            },
            "interaction_analysis": {
                "bucket_summary": feature_bucket_summary,
                "spearman": feature_spearman,
                "conditional_predictive_power": conditional_predictive_power,
                "single_feature_conditional_power": single_feature_power,
                "matched_bin_analysis": {
                    "alignment_score": matched_bin_alignment,
                    "conflict_score": matched_bin_conflict,
                },
            },
        },
        output_json,
    )


if __name__ == "__main__":
    main()
