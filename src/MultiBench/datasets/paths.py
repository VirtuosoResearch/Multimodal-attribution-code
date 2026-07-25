"""Dataset path definitions shared by MultiBench loaders and downstream code."""

from __future__ import annotations

import os
from pathlib import Path


DATASETS_ROOT = Path(os.environ.get("MULTIBENCH_DATASETS_ROOT", "/data/zhenshuo/datasets"))

IMDB_PATH = DATASETS_ROOT / "multimodal_imdb.hdf5"
MOSI_PATH = DATASETS_ROOT / "mosi" / "mosi_data.pkl"
HUMOR_PATH = DATASETS_ROOT / "humor" / "humor.pkl"
SARCASM_PATH = DATASETS_ROOT / "sarcasm" / "sarcasm.pkl"
SARCAM_PATH = SARCASM_PATH
MOSEI_PATH = DATASETS_ROOT / "mosei" / "mosei_senti_data.pkl"
AVMNIST_ROOT = DATASETS_ROOT / "avmnist"
FOOD101_ROOT = DATASETS_ROOT / "food101"
OXFORD_PET_ROOT = DATASETS_ROOT / "oxford_pet"
