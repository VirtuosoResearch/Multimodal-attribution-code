"""Best structured_rbf surrogate hyperparameters from 80/20 LDS grid search."""

from __future__ import annotations

from typing import Any

BEST_STRUCTURED_RBF_HPARAMS_80_20: dict[str, dict[str, Any]] = {
    "avmnist": {
        "gamma_scale": 0.01,
        "method": "structured_rbf",
        "num_test": 20,
        "num_train": 80,
        "ridge_lambda": 0.1,
        "run_name": "kmeans_grouped25_subset100_data90_mod90",
        "setting": "kmeans_data90_mod90",
        "split": "0.8/0.2",
        "split_seed": 0,
        "structured_shared_weight": 2.0,
        "test_lds": 0.3037593984962406,
        "test_pearson": 0.11709379392725645,
        "test_rmse": 0.02408980156272485
    },
    "humor": {
        "gamma_scale": 10.0,
        "method": "structured_rbf",
        "num_test": 20,
        "num_train": 80,
        "ridge_lambda": 1e-08,
        "run_name": "random_grouped25_subset100_data90_mod90",
        "setting": "random_data90_mod90",
        "split": "0.8/0.2",
        "split_seed": 0,
        "structured_shared_weight": 0.0,
        "test_lds": 0.5112781954887218,
        "test_pearson": 0.5572702042512648,
        "test_rmse": 0.007869711364102595
    },
    "imdb": {
        "gamma_scale": 0.0001,
        "method": "structured_rbf",
        "num_test": 20,
        "num_train": 80,
        "ridge_lambda": 100.0,
        "run_name": "random_grouped25_subset100_data80_mod80",
        "setting": "random_data80_mod80",
        "split": "0.8/0.2",
        "split_seed": 0,
        "structured_shared_weight": 5.0,
        "test_lds": 0.5278195488721804,
        "test_pearson": 0.5434641764071925,
        "test_rmse": 0.005773418315719939
    },
    "mosei": {
        "gamma_scale": 0.0001,
        "method": "structured_rbf",
        "num_test": 20,
        "num_train": 80,
        "ridge_lambda": 1e-08,
        "run_name": "random_grouped25_subset100_data80_mod80",
        "setting": "random_data80_mod80",
        "split": "0.8/0.2",
        "split_seed": 0,
        "structured_shared_weight": 5.0,
        "test_lds": 0.45263157894736844,
        "test_pearson": 0.17270281507243038,
        "test_rmse": 0.025410129968524382
    },
    "mosi": {
        "gamma_scale": 100.0,
        "method": "structured_rbf",
        "num_test": 20,
        "num_train": 80,
        "ridge_lambda": 1e-06,
        "run_name": "kmeans_grouped25_subset100_data90_mod90",
        "setting": "kmeans_data90_mod90",
        "split": "0.8/0.2",
        "split_seed": 0,
        "structured_shared_weight": 1.0,
        "test_lds": 0.644033527065794,
        "test_pearson": 0.4793726167751698,
        "test_rmse": 0.08685582443868087
    },
    "sarcasm": {
        "gamma_scale": 1.0,
        "method": "structured_rbf",
        "num_test": 20,
        "num_train": 80,
        "ridge_lambda": 100.0,
        "run_name": "kmeans_grouped25_subset100_data90_mod90",
        "setting": "kmeans_data90_mod90",
        "split": "0.8/0.2",
        "split_seed": 0,
        "structured_shared_weight": 5.0,
        "test_lds": 0.5593984962406015,
        "test_pearson": 0.4727204996918626,
        "test_rmse": 0.03389115540222521
    }
}


def get_best_structured_rbf_hparams(dataset: str) -> dict[str, Any]:
    key = str(dataset).lower()
    if key not in BEST_STRUCTURED_RBF_HPARAMS_80_20:
        raise KeyError(f"no fixed structured_rbf hparams for dataset {dataset!r}")
    return dict(BEST_STRUCTURED_RBF_HPARAMS_80_20[key])
