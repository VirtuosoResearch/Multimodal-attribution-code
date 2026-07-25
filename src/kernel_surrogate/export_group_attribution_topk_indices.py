"""Export top-k group-index JSONs from IF/TRAK/TracIn group attribution outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METHOD_SPECS = {
    "influence_function": {
        "file": "influence_function_lds.json",
        "detail_key": None,
        "scores_key": "group_modality_if_scores",
    },
    "trak": {
        "file": "trak_group_lds.json",
        "detail_key": None,
        "scores_key": "group_modality_scores",
    },
    "tracin_loss_gradient": {
        "file": "tracin_group_lds.json",
        "detail_key": "tracin_loss_gradient",
        "scores_key": "group_modality_scores",
    },
    "tracin_output_gradient": {
        "file": "tracin_group_lds.json",
        "detail_key": "tracin_output_gradient",
        "scores_key": "group_modality_scores",
    },
}

RUN_NAMES = {
    "kmeans_data80_mod80": "kmeans_grouped25_subset100_data80_mod80",
    "kmeans_data90_mod90": "kmeans_grouped25_subset100_data90_mod90",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert IF/TRAK/TracIn group attribution scores into top-k group index JSONs."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--surrogate-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--aux-modality",
        choices=["x", "y"],
        default="x",
        help="Auxiliary modality to score. Column order is x=0, y=1.",
    )
    parser.add_argument("--methods", nargs="+", default=list(METHOD_SPECS))
    return parser.parse_args()


def load_detail(payload: dict[str, Any], dataset: str, setting: str) -> dict[str, Any]:
    for detail in payload.get("dataset_setting_details", []):
        if str(detail.get("dataset")) == dataset and str(detail.get("setting")) == setting:
            return dict(detail)
    raise KeyError(f"missing detail for dataset={dataset!r}, setting={setting!r}")


def method_scores(detail: dict[str, Any], method: str) -> np.ndarray:
    spec = METHOD_SPECS[method]
    if spec["detail_key"] is None:
        scores = detail[spec["scores_key"]]
    else:
        scores = detail["methods"][spec["detail_key"]][spec["scores_key"]]
    return np.asarray(scores, dtype=np.float64)


def indices_for_groups(group_indices: list[list[int]], group_ids: list[int]) -> list[int]:
    selected: list[int] = []
    for group_id in group_ids:
        selected.extend(int(index) for index in group_indices[group_id])
    return sorted(set(selected))


def main() -> None:
    args = parse_args()
    if args.setting not in RUN_NAMES:
        raise KeyError(f"unknown setting {args.setting!r}")
    modality_col = 0 if args.aux_modality == "x" else 1

    group_payload_path = (
        Path("src/kernel_surrogate/subset_results/lds_real_train")
        / RUN_NAMES[args.setting]
        / f"{args.dataset}_seed{int(args.surrogate_seed)}.json"
    )
    group_payload = json.loads(group_payload_path.read_text())
    group_indices = group_payload["group_partition"]["group_indices"]
    num_train_samples = int(
        sum(int(size) for size in group_payload["group_partition"].get("group_sizes", []))
    )
    if num_train_samples <= 0:
        num_train_samples = max(
            (int(index) for indices in group_indices for index in indices),
            default=-1,
        ) + 1

    for method in args.methods:
        spec = METHOD_SPECS[method]
        payload = json.loads((args.input_root / spec["file"]).read_text())
        detail = load_detail(payload, args.dataset, args.setting)
        scores = method_scores(detail, method)
        if scores.ndim != 2 or scores.shape[1] <= modality_col:
            raise ValueError(f"{method} scores have invalid shape {scores.shape}")
        group_scores = scores[:, modality_col]
        order = np.argsort(-group_scores, kind="stable")
        selected_group_ids = [int(group_id) for group_id in order[: int(args.top_k)].tolist()]
        selected_indices = indices_for_groups(group_indices, selected_group_ids)

        output = {
            "selected_indices": selected_indices,
            "selected_group_ids": selected_group_ids,
            "selected_group_scores": [float(group_scores[group_id]) for group_id in selected_group_ids],
            "candidate_group_ids": selected_group_ids,
            "candidate_pool_size": int(len(selected_indices)),
            "selected_fraction": float(len(selected_indices) / max(num_train_samples, 1)),
            "metadata": {
                "dataset": str(args.dataset),
                "surrogate_seed": int(args.surrogate_seed),
                "surrogate_method": str(method),
                "score": f"top {int(args.top_k)} by {args.aux_modality}-modality group attribution score",
                "hparams": {
                    "run_name": RUN_NAMES[args.setting],
                    "setting": str(args.setting),
                },
                "selection_unit": "whole_groups",
                "fallback_group_count": int(args.top_k),
                "num_train_samples": int(num_train_samples),
                "direction": "highest_aux_modality_score",
                "data_selection_seed": int(args.seed),
                "source_file": str(args.input_root / spec["file"]),
                "aux_modality": str(args.aux_modality),
                "aux_modality_column": int(modality_col),
            },
        }

        out_path = args.output_root / method / args.dataset / f"seed{int(args.seed)}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
        print(
            f"{method}: groups={selected_group_ids} scores={output['selected_group_scores']} "
            f"indices={len(selected_indices)} -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
