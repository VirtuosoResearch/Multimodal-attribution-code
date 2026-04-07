from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opt_motivation import default_output_dir
from opt_motivation.run_fisher_hessian_experiment import build_parser as build_single_parser
from opt_motivation.run_fisher_hessian_experiment import run_single_experiment


def parse_args() -> argparse.Namespace:
    single_parser = build_single_parser()
    defaults = {action.dest: action.default for action in single_parser._actions if action.dest != "help"}
    parser = argparse.ArgumentParser(description="Run the fresh Fisher/Hessian pipeline for multiple seeds.")
    parser.add_argument("--dataset", type=str, required=True, choices=["mosi", "imdb", "humor", "sarcasm"])
    parser.add_argument("--mosi-path", type=str, default=defaults["mosi_path"])
    parser.add_argument("--imdb-path", type=str, default=defaults["imdb_path"])
    parser.add_argument("--humor-path", type=str, default=defaults["humor_path"])
    parser.add_argument("--sarcasm-path", type=str, default=defaults["sarcasm_path"])
    parser.add_argument("--analysis-split", type=str, default=defaults["analysis_split"], choices=["train", "valid", "test"])
    parser.add_argument("--max-seq-len", type=int, default=defaults["max_seq_len"])
    parser.add_argument("--max-samples", type=int, default=defaults["max_samples"])
    parser.add_argument("--train-max-samples", type=int, default=defaults["train_max_samples"])
    parser.add_argument("--valid-max-samples", type=int, default=defaults["valid_max_samples"])
    parser.add_argument("--test-max-samples", type=int, default=defaults["test_max_samples"])
    parser.add_argument("--imdb-text-dropout-prob", type=float, default=defaults["imdb_text_dropout_prob"])
    parser.add_argument("--device", type=str, default=defaults["device"])
    parser.add_argument("--encoder-hidden-dim", type=int, default=defaults["encoder_hidden_dim"])
    parser.add_argument("--fusion-dim", type=int, default=defaults["fusion_dim"])
    parser.add_argument("--fusion-layers", type=int, default=defaults["fusion_layers"])
    parser.add_argument("--num-heads", type=int, default=defaults["num_heads"])
    parser.add_argument("--ffn-dim", type=int, default=defaults["ffn_dim"])
    parser.add_argument("--dropout", type=float, default=defaults["dropout"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--patience", type=int, default=defaults["patience"])
    parser.add_argument("--architecture", type=str, default=defaults["architecture"], choices=["shared_head", "cross_attention"])
    parser.add_argument("--parameter-scope", type=str, default=defaults["parameter_scope"], choices=["shared_core", "shared", "head", "representation", "classifier", "all"])
    parser.add_argument("--measurement-batch-size", type=int, default=defaults["measurement_batch_size"])
    parser.add_argument("--max-measurement-samples", type=int, default=defaults["max_measurement_samples"])
    parser.add_argument("--fisher-batches", type=int, default=defaults["fisher_batches"])
    parser.add_argument("--hessian-batches", type=int, default=defaults["hessian_batches"])
    parser.add_argument("--subspace-energy", type=float, default=defaults["subspace_energy"])
    parser.add_argument("--max-subspace-rank", type=int, default=defaults["max_subspace_rank"])
    parser.add_argument("--principal-top-k", type=int, default=defaults["principal_top_k"])
    parser.add_argument("--hessian-top-k", type=int, default=defaults["hessian_top_k"])
    parser.add_argument("--hessian-power-iters", type=int, default=defaults["hessian_power_iters"])
    parser.add_argument("--hessian-trace-samples", type=int, default=defaults["hessian_trace_samples"])
    parser.add_argument("--quadratic-form-samples", type=int, default=defaults["quadratic_form_samples"])
    parser.add_argument("--beta", type=float, default=defaults["beta"])
    parser.add_argument("--fisher-eps", type=float, default=defaults["fisher_eps"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir() / "fisher_hessian_multiseed_runs"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for seed in args.seeds:
        single_args = argparse.Namespace(**vars(args))
        single_args.seed = int(seed)
        single_args.output_json = str(output_dir / f"{args.dataset}_fisher_hessian_seed{seed}.json")
        written.append(run_single_experiment(single_args))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
