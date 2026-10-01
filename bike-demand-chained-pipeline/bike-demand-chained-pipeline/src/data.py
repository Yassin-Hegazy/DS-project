"""Data loading and split utilities."""

from __future__ import annotations

import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split


def compute_team_seed(student_ids: list[str]) -> int:
    """Compute the assignment-defined team seed."""
    if not student_ids:
        raise ValueError("Add the registered team student IDs before running.")
    joined = "_".join(sorted(map(str, student_ids)))
    return int(hashlib.sha256(joined.encode()).hexdigest(), 16) % 100000


def load_data(train_path, test_path):
    return pd.read_csv(train_path), pd.read_csv(test_path)


def seeded_split(train_df: pd.DataFrame, seed: int, test_size: float = 0.20):
    """The assignment's single seeded 80/20 split."""
    return train_test_split(
        train_df,
        test_size=test_size,
        random_state=seed,
    )


def chronological_split(df: pd.DataFrame, validation_fraction: float = 0.20):
    """Chronological split for Phase 3.

    TODO: choose and justify the exact chronological cut used by the team.
    """
    ordered = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    cut = int(len(ordered) * (1 - validation_fraction))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
