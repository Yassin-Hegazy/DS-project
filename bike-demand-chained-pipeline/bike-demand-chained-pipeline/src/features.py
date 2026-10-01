"""Feature engineering.

Add only transformations the team can explain and support with experiments.
Fit train-dependent transformations on training data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["dteday"])
    out["day"] = dt.dt.day
    out["dayofyear"] = dt.dt.dayofyear
    return out


def add_cyclic_hour(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hr_sin"] = np.sin(2 * np.pi * out["hr"] / 24)
    out["hr_cos"] = np.cos(2 * np.pi * out["hr"] / 24)
    return out


def add_cyclic_month(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mnth_sin"] = np.sin(2 * np.pi * (out["mnth"] - 1) / 12)
    out["mnth_cos"] = np.cos(2 * np.pi * (out["mnth"] - 1) / 12)
    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline feature pipeline.

    TODO: evolve this through controlled experiments rather than adding
    every possible feature at once.
    """
    out = add_date_parts(df)
    out = add_cyclic_hour(out)
    out = add_cyclic_month(out)
    return out
