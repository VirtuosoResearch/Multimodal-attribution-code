from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motivation.core import write_json
from opt_motivation import default_output_dir
from opt_motivation.core import (
    auto_device,
    collect_empirical_fisher_signals,
    load_affect,
    load_imdb,
    load_mosi,
    train_joint_model,
)


def parse_args() -> argparse.Namespace:
    output_dir = default_output_dir()
    parser = argparse.ArgumentParser(description="Measure empirical parameter Fisher on shared or representation parameters.")
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--parameter-scope", type=str, default="shared", choices=["shared", "representation", "head", "classifier", "all"])
    parser.add_argument("--architecture", type=str, default="shared_head", choices=["shared_head", "cross_attention"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    output_dir = default_output_dir()
    output_json = Path(args.output_json) if args.output_json else output_dir / f"{args.dataset}_empirical_fisher_{args.parameter_scope}.json"

    dataset = _load_dataset(args)
    split = {"train": dataset.train, "valid": dataset.valid, "test": dataset.test}[args.analysis_split]
    model, train_metrics = train_joint_model(
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
        seed=args.seed,
        architecture=args.architecture,
    )
    fisher = collect_empirical_fisher_signals(
        dataset,
        split,
        model,
        device=args.device,
        batch_size=args.eval_batch_size,
        parameter_scope=args.parameter_scope,
    )
    write_json(
        {
            "dataset": args.dataset,
            "analysis_split": args.analysis_split,
            "seed": args.seed,
            "architecture": args.architecture,
            "parameter_scope": args.parameter_scope,
            "training": train_metrics,
            "empirical_fisher": {
                "parameter_scope": fisher["parameter_scope"],
                "parameter_names": fisher["parameter_names"],
                "num_parameters": fisher["num_parameters"],
                "summary": fisher["summary"],
                "sample_trace_mean": fisher["sample_trace_mean"].astype(float).tolist(),
                "sample_pair_inner_mean": fisher["sample_pair_inner_mean"].astype(float).tolist(),
                "sample_pair_cos_mean": fisher["sample_pair_cos_mean"].astype(float).tolist(),
                "sample_trace_by_modality": {name: values.astype(float).tolist() for name, values in fisher["sample_trace_by_modality"].items()},
                "sample_pair_inner": {name: values.astype(float).tolist() for name, values in fisher["sample_pair_inner"].items()},
                "sample_pair_cos": {name: values.astype(float).tolist() for name, values in fisher["sample_pair_cos"].items()},
            },
        },
        output_json,
    )


if __name__ == "__main__":
    main()
