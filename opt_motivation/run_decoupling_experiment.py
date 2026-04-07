from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opt_motivation import default_output_dir
from opt_motivation.core import auto_device, run_experiment


def parse_args() -> argparse.Namespace:
    output_dir = default_output_dir()
    parser = argparse.ArgumentParser(description="Run the representational-vs-optimization alignment decoupling experiment.")
    parser.add_argument("--dataset", type=str, default="mosi", choices=["mosi", "imdb", "humor", "sarcasm"])
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
    parser.add_argument("--knn-device", type=str, default=None)
    parser.add_argument("--encoder-hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-dim", type=int, default=128)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-alignment-permutations", type=int, default=100)
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--top-k-subspace", type=int, default=10)
    parser.add_argument("--pid-source", type=str, default="auto", choices=["auto", "confidence", "lsmi"])
    parser.add_argument("--num-pid-bins", type=int, default=5)
    parser.add_argument("--bucket-margin", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--figure-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = default_output_dir()
    output_json = Path(args.output_json) if args.output_json else output_dir / f"{args.dataset}_decoupling_results.json"
    figure_dir = Path(args.figure_dir) if args.figure_dir else output_dir / "figures"

    run_experiment(
        dataset_name=args.dataset,
        mosi_path=args.mosi_path,
        imdb_path=args.imdb_path,
        humor_path=args.humor_path,
        sarcasm_path=args.sarcasm_path,
        analysis_split=args.analysis_split,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        train_max_samples=args.train_max_samples,
        valid_max_samples=args.valid_max_samples,
        test_max_samples=args.test_max_samples,
        imdb_text_dropout_prob=args.imdb_text_dropout_prob,
        device=args.device,
        knn_device=args.knn_device or args.device,
        encoder_hidden_dim=args.encoder_hidden_dim,
        fusion_dim=args.fusion_dim,
        fusion_layers=args.fusion_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        k=args.k,
        num_alignment_permutations=args.num_alignment_permutations,
        num_bootstrap=args.num_bootstrap,
        top_k_subspace=args.top_k_subspace,
        pid_source=args.pid_source,
        num_pid_bins=args.num_pid_bins,
        bucket_margin=args.bucket_margin,
        output_json=output_json,
        figure_dir=figure_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
