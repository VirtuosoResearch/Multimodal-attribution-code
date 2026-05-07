from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _get_by_path(payload: dict[str, Any], key_path: str) -> Any:
    value: Any = payload
    for part in key_path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise KeyError(f"missing score key path: {key_path}")
    return value


def _load_group_partition(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "group_partition" not in payload:
        raise KeyError(f"{path} does not contain group_partition")
    return payload["group_partition"]


def _target_count(num_samples: int, *, budget: float | None, top_k: int | None) -> int | None:
    if top_k is not None:
        return None
    if budget is None:
        raise ValueError("pass --budget for sample-budget selection or --top-k for whole-group selection")
    if not 0.0 < float(budget) <= 1.0:
        raise ValueError(f"budget must be in (0, 1], got {budget}")
    return min(int(num_samples), max(1, int(round(int(num_samples) * float(budget)))))


def ranked_group_selection(
    *,
    scores: np.ndarray,
    group_indices: list[list[int]],
    direction: str,
    target_count: int | None,
    top_k: int | None,
    seed: int,
) -> tuple[list[int], list[int], int]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.shape[0] != len(group_indices):
        raise ValueError(
            f"score count {values.shape[0]} does not match group count {len(group_indices)}"
        )
    order = np.argsort(values, kind="mergesort")
    if direction == "largest":
        order = order[::-1]
    elif direction != "smallest":
        raise ValueError(f"unsupported direction: {direction}")

    selected_group_ids: list[int] = []
    candidate_indices: list[int] = []
    seen: set[int] = set()
    group_limit = int(top_k) if top_k is not None else len(order)
    for group_id in order[:group_limit].tolist():
        selected_group_ids.append(int(group_id))
        for index in group_indices[int(group_id)]:
            resolved = int(index)
            if resolved not in seen:
                candidate_indices.append(resolved)
                seen.add(resolved)
        if target_count is not None and len(candidate_indices) >= int(target_count):
            break

    if target_count is None:
        return sorted(candidate_indices), selected_group_ids, len(candidate_indices)
    if len(candidate_indices) < int(target_count):
        raise ValueError(
            f"ranked groups provide {len(candidate_indices)} samples, "
            f"but target_count={target_count}"
        )
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        np.asarray(candidate_indices, dtype=np.int64),
        size=int(target_count),
        replace=False,
    )
    return sorted(int(index) for index in selected.tolist()), selected_group_ids, len(candidate_indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert attribution scores into a downsampled data-selection index JSON."
    )
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--group-payload", type=Path, required=True)
    parser.add_argument(
        "--score-key",
        default="modality_group_scores.combined",
        help="Dotted key path in score JSON, e.g. modality_group_scores.y.",
    )
    parser.add_argument("--direction", choices=["largest", "smallest"], default="largest")
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.budget is not None and args.top_k is not None:
        raise ValueError("use either --budget or --top-k, not both")
    score_payload = json.loads(args.score_file.read_text(encoding="utf-8"))
    partition = _load_group_partition(args.group_payload)
    group_indices = [[int(index) for index in group] for group in partition["group_indices"]]
    num_samples = int(partition.get("dataset_size") or sum(len(group) for group in group_indices))
    scores = np.asarray(_get_by_path(score_payload, args.score_key), dtype=np.float64)
    target_count = _target_count(num_samples, budget=args.budget, top_k=args.top_k)
    selected_indices, selected_group_ids, candidate_pool_size = ranked_group_selection(
        scores=scores,
        group_indices=group_indices,
        direction=args.direction,
        target_count=target_count,
        top_k=args.top_k,
        seed=int(args.seed),
    )
    flat_scores = scores.reshape(-1)
    output = {
        "selected_indices": selected_indices,
        "selected_group_ids": selected_group_ids,
        "selected_group_scores": [float(flat_scores[group_id]) for group_id in selected_group_ids],
        "candidate_pool_size": int(candidate_pool_size),
        "selected_fraction": float(len(selected_indices) / max(num_samples, 1)),
        "metadata": {
            "score_file": str(args.score_file),
            "group_payload": str(args.group_payload),
            "score_key": str(args.score_key),
            "direction": str(args.direction),
            "budget": args.budget,
            "top_k": args.top_k,
            "seed": int(args.seed),
            "num_train_samples": int(num_samples),
            "selection_unit": "ranked_groups_then_samples" if args.budget is not None else "whole_groups",
            "hparams": {
                "run_name": score_payload.get("metadata", {}).get("run_name"),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    print(
        f"saved {len(selected_indices)} indices from {len(selected_group_ids)} groups to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
