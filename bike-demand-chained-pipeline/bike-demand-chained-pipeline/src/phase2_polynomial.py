"""Phase 2 — polynomial expansion using the Phase 1 GD implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import group_rare_weather

# A cell with fewer training rows than this has an unreliable mean residual:
# with a residual SD near 100 bikes, the mean of 30 rows is still +/- 18 bikes.
MIN_CELL_ROWS = 30

# Candidate combinations for the residual diagnosis: name -> grouping columns.
CANDIDATE_GROUPS = {
    "hour x workingday": ["hr", "workingday"],
    "hour x yr": ["hr", "yr"],
    "hour x season": ["hr", "season"],
    "hour x month": ["hr", "mnth"],
    "hour x weathersit": ["hr", "weathersit"],
    "hour x holiday": ["hr", "holiday"],
    "hour x weekday": ["hr", "weekday"],
    "temp bins (10)": ["temp_bin10"],
    "temp bins (5) x hour": ["temp_bin5", "hr"],
    "hum bins (5) x hour": ["hum_bin5", "hr"],
    "windspeed bins (5) x hour": ["windspeed_bin5", "hr"],
    "temp bins (5) x hum bins (5)": ["temp_bin5", "hum_bin5"],
}


def quantile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    """Cut a training column into (at most) n_bins equal-count bins.

    Tied values can merge neighbouring bins, so fewer bins may result.
    """
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    return pd.cut(values, edges, include_lowest=True)


def diagnosis_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    """Grouping columns for the residual diagnosis (training rows only)."""
    out = group_rare_weather(train_df)[
        ["hr", "workingday", "yr", "season", "mnth", "weathersit", "holiday", "weekday"]
    ].copy()
    out["temp_bin10"] = quantile_bins(train_df["temp"], 10)
    out["temp_bin5"] = quantile_bins(train_df["temp"], 5)
    out["hum_bin5"] = quantile_bins(train_df["hum"], 5)
    out["windspeed_bin5"] = quantile_bins(train_df["windspeed"], 5)
    return out


def between_group_share(residual: np.ndarray, groups: pd.DataFrame) -> dict:
    """Share of residual variance explained by the group means, plus cell support.

    between share = sum over cells of n_cell * (cell mean - overall mean)^2,
                    divided by the total sum of squares of the residuals.
    adjusted share removes what G cells would capture from pure noise, like
    adjusted R^2: 1 - (1 - share) * (n - 1) / (n - G).
    """
    frame = groups.assign(residual=residual)
    cells = frame.groupby(list(groups.columns), observed=True)["residual"]
    cell_mean = cells.transform("mean")
    centered = residual - residual.mean()
    share = float(np.sum((cell_mean - residual.mean()) ** 2) / np.sum(centered ** 2))
    sizes = cells.size()
    n, g = len(residual), len(sizes)
    sparse = sizes < MIN_CELL_ROWS
    return {
        "cells": g,
        "between_share": share,
        "adjusted_share": 1 - (1 - share) * (n - 1) / (n - g),
        "min_cell_rows": int(sizes.min()),
        "median_cell_rows": float(sizes.median()),
        f"cells_under_{MIN_CELL_ROWS}": int(sparse.sum()),
        "rows_in_sparse_cells": int(sizes[sparse].sum()),
    }


def residual_diagnosis(train_df: pd.DataFrame, residual: np.ndarray) -> pd.DataFrame:
    """Rank every candidate combination by its between-group variance share."""
    frame = diagnosis_frame(train_df)
    rows = [{"combination": name, **between_group_share(residual, frame[columns])}
            for name, columns in CANDIDATE_GROUPS.items()]
    table = pd.DataFrame(rows).sort_values("between_share", ascending=False)
    return table.reset_index(drop=True)


def residual_cell_means(train_df: pd.DataFrame, residual: np.ndarray, columns) -> pd.DataFrame:
    """Mean residual and training-row count per cell of one combination."""
    frame = diagnosis_frame(train_df)[columns].assign(residual=residual)
    return (frame.groupby(columns, observed=True)["residual"]
            .agg(mean_residual="mean", rows="size").reset_index())


def build_polynomial_features(X, feature_names, degree, expand_features):
    """TODO:
    - expand only justified features,
    - preserve a mapping from original to expanded columns,
    - initialize compatible terms from Phase 1 weights.
    """
    raise NotImplementedError
