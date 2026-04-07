from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motivation.core import write_json
from opt_motivation import default_output_dir
from opt_motivation.core import auto_device, run_experiment


def parse_args() -> argparse.Namespace:
    output_dir = default_output_dir()
    parser = argparse.ArgumentParser(description="Run multi-seed IMDb LSMI-vs-Fisher experiments and aggregate the results.")
    parser.add_argument("--imdb-path", type=str, default="/data/zhenshuo/datasets/multimodal_imdb.hdf5")
    parser.add_argument("--analysis-split", type=str, default="test", choices=["train", "valid", "test"])
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-alignment-permutations", type=int, default=20)
    parser.add_argument("--num-bootstrap", type=int, default=200)
    parser.add_argument("--top-k-subspace", type=int, default=10)
    parser.add_argument("--pid-source", type=str, default="lsmi", choices=["auto", "confidence", "lsmi"])
    parser.add_argument("--num-pid-bins", type=int, default=5)
    parser.add_argument("--bucket-margin", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-json", type=str, default=str(output_dir / "imdb_lsmi_fisher_multiseed_summary.json"))
    parser.add_argument("--runs-dir", type=str, default=str(output_dir / "multiseed_runs"))
    parser.add_argument("--figure-dir", type=str, default=str(output_dir / "multiseed_figures"))
    return parser.parse_args()


def _mean_std(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(finite) == 1:
        return {"mean": finite[0], "std": 0.0, "n": 1}
    mean = sum(finite) / len(finite)
    var = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return {"mean": mean, "std": math.sqrt(max(var, 0.0)), "n": len(finite)}


def _extract_summary(result: dict[str, Any]) -> dict[str, float]:
    analysis = result["analysis"]
    conditional = analysis["conditional_predictive_power"]
    return {
        "overall_spearman": float(analysis["overall_spearman"]["rho"]),
        "rep_only_auc": float(analysis["auc_models"]["A_rep_only"]["auc"]),
        "opt_only_auc": float(analysis["auc_models"]["B_opt_only"]["auc"]),
        "both_auc": float(analysis["auc_models"]["C_both"]["auc"]),
        "correctness_pid_auc": float(conditional["classification"]["correctness"]["models"]["pid_only"]["auc"]),
        "correctness_fisher_auc": float(conditional["classification"]["correctness"]["models"]["fisher_only"]["auc"]),
        "correctness_both_auc": float(conditional["classification"]["correctness"]["models"]["pid_plus_fisher"]["auc"]),
        "correctness_both_minus_pid_auc": float(
            conditional["classification"]["correctness"]["bootstrap_tests"]["both_minus_pid_auc"]["mean_diff"]
        ),
        "neg_adv_pid_r2": float(conditional["regression"]["negative_loss_advantage"]["models"]["pid_only"]["r2"]),
        "neg_adv_fisher_r2": float(conditional["regression"]["negative_loss_advantage"]["models"]["fisher_only"]["r2"]),
        "neg_adv_both_r2": float(conditional["regression"]["negative_loss_advantage"]["models"]["pid_plus_fisher"]["r2"]),
        "neg_adv_both_minus_pid_r2": float(
            conditional["regression"]["negative_loss_advantage"]["bootstrap_tests"]["both_minus_pid_r2"]["mean_diff"]
        ),
        "beats_pid_auc": float(conditional["classification"]["beats_best_unimodal"]["models"]["pid_only"]["auc"]),
        "beats_fisher_auc": float(conditional["classification"]["beats_best_unimodal"]["models"]["fisher_only"]["auc"]),
        "beats_both_auc": float(conditional["classification"]["beats_best_unimodal"]["models"]["pid_plus_fisher"]["auc"]),
        "beats_both_minus_pid_auc": float(
            conditional["classification"]["beats_best_unimodal"]["bootstrap_tests"]["both_minus_pid_auc"]["mean_diff"]
        ),
        "matched_mean_correctness_delta": float(analysis["matched_bin_analysis"]["summary"]["mean_correctness_delta"]),
        "matched_mean_neg_adv_delta": float(analysis["matched_bin_analysis"]["summary"]["mean_negative_loss_advantage_delta"]),
        "matched_mean_beats_delta": float(analysis["matched_bin_analysis"]["summary"]["mean_beats_best_unimodal_delta"]),
    }


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json)
    runs_dir = Path(args.runs_dir)
    figure_dir = Path(args.figure_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    per_seed = []
    summaries = []
    for seed in args.seeds:
        run_output = runs_dir / f"imdb_lsmi_fisher_seed{seed}.json"
        run_figures = figure_dir / f"seed_{seed}"
        result = run_experiment(
            dataset_name="imdb",
            mosi_path="",
            imdb_path=args.imdb_path,
            humor_path="",
            sarcasm_path="",
            analysis_split=args.analysis_split,
            max_seq_len=50,
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
            output_json=run_output,
            figure_dir=run_figures,
            seed=seed,
        )
        summary = _extract_summary(result)
        summaries.append(summary)
        per_seed.append({"seed": seed, "output_json": str(run_output), "summary": summary})

    metric_names = list(summaries[0].keys()) if summaries else []
    aggregate = {name: _mean_std([summary[name] for summary in summaries]) for name in metric_names}
    payload = {
        "dataset": "imdb",
        "pid_source": args.pid_source,
        "analysis_split": args.analysis_split,
        "seeds": args.seeds,
        "config": {
            "imdb_path": args.imdb_path,
            "max_samples": args.max_samples,
            "train_max_samples": args.train_max_samples,
            "valid_max_samples": args.valid_max_samples,
            "test_max_samples": args.test_max_samples,
            "imdb_text_dropout_prob": args.imdb_text_dropout_prob,
            "device": args.device,
            "knn_device": args.knn_device or args.device,
            "encoder_hidden_dim": args.encoder_hidden_dim,
            "fusion_dim": args.fusion_dim,
            "fusion_layers": args.fusion_layers,
            "num_heads": args.num_heads,
            "ffn_dim": args.ffn_dim,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "patience": args.patience,
            "k": args.k,
            "num_alignment_permutations": args.num_alignment_permutations,
            "num_bootstrap": args.num_bootstrap,
            "top_k_subspace": args.top_k_subspace,
            "num_pid_bins": args.num_pid_bins,
            "bucket_margin": args.bucket_margin,
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    write_json(payload, output_json)


if __name__ == "__main__":
    main()
