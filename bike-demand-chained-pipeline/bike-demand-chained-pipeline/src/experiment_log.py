"""Small experiment registry so model decisions are evidence-backed."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


DEFAULT_COLUMNS = [
    "experiment_id",
    "phase",
    "feature_set",
    "split_type",
    "polynomial_degree",
    "learning_rate",
    "lambda",
    "l1_ratio",
    "train_r2",
    "val_r2",
    "val_rmse",
    "iterations",
    "convergence_status",
    "selected_features",
    "notes",
]


def append_experiment(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{c: record.get(c) for c in DEFAULT_COLUMNS}])
    if path.exists():
        existing = pd.read_csv(path)
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(path, index=False)
