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
    load_affect,
    load_imdb,
    load_mosi,
    measure_fisher_hessian_interactions,
    train_joint_model,
)


def build_parser() -> argparse.ArgumentParser:
    output_dir = default_output_dir()
    parser = argparse.ArgumentParser(description="Run fresh Fisher/Hessian measurements for multimodal interaction.")
    parser.add_argument("--dataset", type=str, required=True, choices=["mosi", "imdb", "humor", "sarcasm"])
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--architecture", type=str, default="shared_head", choices=["shared_head", "cross_attention"])
    parser.add_argument("--parameter-scope", type=str, default="shared_core", choices=["shared_core", "shared", "head", "representation", "classifier", "all"])
    parser.add_argument("--measurement-batch-size", type=int, default=32)
    parser.add_argument("--max-measurement-samples", type=int, default=None)
    parser.add_argument("--fisher-batches", type=int, default=32)
    parser.add_argument("--hessian-batches", type=int, default=8)
    parser.add_argument("--subspace-energy", type=float, default=0.9)
    parser.add_argument("--max-subspace-rank", type=int, default=16)
    parser.add_argument("--principal-top-k", type=int, default=8)
    parser.add_argument("--hessian-top-k", type=int, default=3)
    parser.add_argument("--hessian-power-iters", type=int, default=12)
    parser.add_argument("--hessian-trace-samples", type=int, default=12)
    parser.add_argument("--quadratic-form-samples", type=int, default=16)
    parser.add_argument("--random-subspace-samples", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--fisher-eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


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


def _default_epochs(dataset: str) -> int:
    if dataset == "imdb":
        return 10
    return 30


def run_single_experiment(args: argparse.Namespace) -> Path:
    output_dir = default_output_dir()
    output_json = Path(args.output_json) if args.output_json else output_dir / f"{args.dataset}_fisher_hessian_hypothesis.json"
    dataset = _load_dataset(args)
    split = {"train": dataset.train, "valid": dataset.valid, "test": dataset.test}[args.analysis_split]
    epochs = int(args.epochs) if args.epochs is not None else _default_epochs(args.dataset)

    joint_model, train_metrics = train_joint_model(
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

    metrics = measure_fisher_hessian_interactions(
        dataset,
        split,
        joint_model,
        device=args.device,
        train_lr=args.lr,
        parameter_scope=args.parameter_scope,
        measurement_batch_size=args.measurement_batch_size,
        max_measurement_samples=args.max_measurement_samples,
        fisher_batches=args.fisher_batches,
        hessian_batches=args.hessian_batches,
        subspace_energy=args.subspace_energy,
        max_subspace_rank=args.max_subspace_rank,
        principal_top_k=args.principal_top_k,
        hessian_top_k=args.hessian_top_k,
        hessian_power_iters=args.hessian_power_iters,
        hessian_trace_samples=args.hessian_trace_samples,
        quadratic_form_samples=args.quadratic_form_samples,
        random_subspace_samples=args.random_subspace_samples,
        beta=args.beta,
        fisher_eps=args.fisher_eps,
        seed=args.seed,
    )

    write_json(
        {
            "dataset": args.dataset,
            "analysis_split": args.analysis_split,
            "seed": args.seed,
            "architecture": args.architecture,
            "training": train_metrics,
            "config": {
                "device": args.device,
                "epochs": epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "measurement_batch_size": args.measurement_batch_size,
                "max_measurement_samples": args.max_measurement_samples,
                "fisher_batches": args.fisher_batches,
                "hessian_batches": args.hessian_batches,
                "parameter_scope": args.parameter_scope,
                "subspace_energy": args.subspace_energy,
                "max_subspace_rank": args.max_subspace_rank,
                "principal_top_k": args.principal_top_k,
                "hessian_top_k": args.hessian_top_k,
                "hessian_power_iters": args.hessian_power_iters,
                "hessian_trace_samples": args.hessian_trace_samples,
                "quadratic_form_samples": args.quadratic_form_samples,
                "random_subspace_samples": args.random_subspace_samples,
                "beta": args.beta,
                "fisher_eps": args.fisher_eps,
            },
            "metrics": metrics,
        },
        output_json,
    )
    return output_json


def main() -> None:
    args = parse_args()
    output_json = run_single_experiment(args)
    print(output_json)


if __name__ == "__main__":
    main()
