# Where the data comes from: the local copy when it is in the repo, otherwise streamed from the
# private Hugging Face dataset (run `hf auth login` once). Used by every notebook and script.
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HF_DATASET_URL = "hf://datasets/DSCI-Capstone/county-outage-data"

# Model-table files by split. The 2025 test labels live in holdout/ and are deliberately not listed:
# they are opened once, for the final evaluation, and never through this loader.
MODEL_TABLE = {
    "train": "processed/model_table/train.parquet",
    "validation": "processed/model_table/validation.parquet",
    "test": "processed/model_table/test_features.parquet",
}


def data_path(rel):
    """Local path if the file or folder is in the repo, otherwise its hf:// URL."""
    local = ROOT / rel
    return local.as_posix() if local.exists() else f"{HF_DATASET_URL}/{rel}"


def model_table_paths(splits=("train", "validation")):
    return [data_path(MODEL_TABLE[s]) for s in splits]


def load_model_table(splits=("train", "validation")):
    """Model-table splits as DataFrames, e.g. train, val = load_model_table(). 'test' = features only."""
    frames = [pd.read_parquet(p) for p in model_table_paths(splits)]
    return frames[0] if len(frames) == 1 else frames
