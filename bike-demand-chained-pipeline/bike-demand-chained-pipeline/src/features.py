"""Feature engineering.

Add only transformations the team can explain and support with experiments.
Fit train-dependent transformations on training data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed calendar origin (first labeled date), not a fitted quantity.
START_DATE = pd.Timestamp("2011-01-01")

# Phase 1 main effects. One level of every one-hot group is dropped (the
# reference level) so the dummies plus intercept are not exactly collinear.
# workingday is left out: it is an exact function of weekday and holiday.
PHASE1_NUMERIC = ["yr", "holiday", "temp", "hum", "windspeed"]
PHASE1_CATEGORICAL = ["hr", "weekday", "weathersit"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the dteday text into a number: days elapsed since START_DATE."""
    out = df.copy()
    out["days_since_start"] = (pd.to_datetime(out["dteday"]) - START_DATE).dt.days
    return out


def group_rare_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Map weathersit 4 (heavy rain/snow) into 3, the nearest severe level.

    Level 4 has a single row in all labeled data, and none in our training
    portion, so no separate coefficient can be learned for it.
    """
    out = df.copy()
    out["weathersit"] = out["weathersit"].clip(upper=3)
    return out


class Phase1Features:
    """One-hot encoder plus standard scaler, fitted on the training portion only.

    seasonal: "mnth" or "season" (season is an exact function of month here,
    so only one of them can be used).
    use_trend: include days_since_start, the continuous time from dteday.
    """

    def __init__(self, seasonal: str = "mnth", use_trend: bool = True):
        if seasonal not in ("mnth", "season"):
            raise ValueError("seasonal must be 'mnth' or 'season'")
        self.seasonal = seasonal
        self.use_trend = use_trend
        self.levels = None
        self.mean = None
        self.std = None
        self.feature_names = None

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return group_rare_weather(add_calendar_features(df))

    def _numeric_columns(self) -> list[str]:
        return PHASE1_NUMERIC + (["days_since_start"] if self.use_trend else [])

    def _categorical_columns(self) -> list[str]:
        return PHASE1_CATEGORICAL + [self.seasonal]

    def _unscaled(self, prepared: pd.DataFrame) -> pd.DataFrame:
        columns = {c: prepared[c].astype(float) for c in self._numeric_columns()}
        for c in self._categorical_columns():
            for level in self.levels[c]:
                columns[f"{c}={level}"] = prepared[c].eq(level).astype(float)
        return pd.DataFrame(columns, index=prepared.index)

    def fit(self, train_df: pd.DataFrame) -> "Phase1Features":
        prepared = self._prepare(train_df)
        # Levels come from training only; the smallest level is the reference.
        self.levels = {
            c: sorted(int(v) for v in prepared[c].unique())[1:]
            for c in self._categorical_columns()
        }
        raw = self._unscaled(prepared)
        self.mean = raw.mean(axis=0).to_numpy()
        self.std = raw.std(axis=0, ddof=0).to_numpy()
        if np.any(self.std == 0):
            raise ValueError("A feature is constant in the training portion.")
        self.feature_names = ["intercept"] + raw.columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return the design matrix: intercept column, then scaled features."""
        if self.mean is None:
            raise RuntimeError("Phase1Features has not been fitted.")
        raw = self._unscaled(self._prepare(df)).to_numpy()
        scaled = (raw - self.mean) / self.std
        return np.column_stack([np.ones(len(scaled)), scaled])

    def to_dict(self) -> dict:
        return {
            "seasonal": self.seasonal,
            "use_trend": self.use_trend,
            "levels": self.levels,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, state: dict) -> "Phase1Features":
        features = cls(seasonal=state["seasonal"], use_trend=state["use_trend"])
        features.levels = {c: list(v) for c, v in state["levels"].items()}
        features.mean = np.asarray(state["mean"])
        features.std = np.asarray(state["std"])
        features.feature_names = list(state["feature_names"])
        return features
