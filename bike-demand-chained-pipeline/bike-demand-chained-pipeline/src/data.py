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
    """The one chronological split (Phase 3): the last 20% of labeled DAYS.

    The cut falls between whole days, so no day has hours on both sides.
    Uses dates only, never labels. Returns (earlier, later, first_validation_day).
    """
    days = sorted(df["dteday"].unique())
    first_validation_day = days[-round(validation_fraction * len(days))]
    earlier = df[df["dteday"] < first_validation_day].copy()
    later = df[df["dteday"] >= first_validation_day].copy()
    return earlier, later, first_validation_day
