"""Operator-facing error checks, applied to every fitted model.

The costly failure for the operator is too few bikes at busy times, so we
look at errors by demand level and by hour x day-type, not only at R2.
Error = predicted - actual (negative = under-prediction).
Shortfall = bikes missing in an hour = max(actual - predicted, 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def demand_decile_edges(y_train) -> np.ndarray:
    """Decile edges of actual demand, fitted on the training portion only."""
    edges = np.quantile(y_train, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf  # validation values outside the training range
    return edges


def _error_summary(frame: pd.DataFrame, by) -> pd.DataFrame:
    return frame.groupby(by, observed=True).agg(
        hours=("error", "size"),
        mean_actual=("actual", "mean"),
        mean_predicted=("predicted", "mean"),
        mean_error=("error", "mean"),
        under_prediction_rate=("under", "mean"),
        mean_shortfall_bikes=("shortfall", "mean"),
    ).reset_index()


def _error_frame(y, pred) -> pd.DataFrame:
    error = pred - y
    return pd.DataFrame({
        "actual": y,
        "predicted": pred,
        "error": error,
        "under": error < 0,
        "shortfall": np.maximum(-error, 0),
    })


def errors_by_demand_decile(y, pred, edges, bin_on="actual") -> pd.DataFrame:
    """Errors per decile of demand.

    bin_on="actual" groups hours by what really happened (the operator's
    question: are the busiest hours under-supplied?). Hours land in the top
    actual bin partly because they were unusually busy by chance, so any
    imperfect model looks under-predicted there (regression to the mean).
    bin_on="predicted" groups by what the model said and has no such
    selection effect: under-prediction there is a real bias.
    """
    frame = _error_frame(y, pred)
    frame["decile"] = pd.cut(frame[bin_on], edges, labels=range(1, 11))
    return _error_summary(frame, "decile")


def errors_by_hour_daytype(df, y, pred) -> pd.DataFrame:
    frame = _error_frame(y, pred)
    frame["workingday"] = df["workingday"].to_numpy()
    frame["hr"] = df["hr"].to_numpy()
    return _error_summary(frame, ["workingday", "hr"])


def top_decile_shortfall(y, pred, edges) -> float:
    """Mean bikes short per hour among hours whose actual demand is in the top decile."""
    top = y > edges[-2]
    return float(np.mean(np.maximum(y[top] - pred[top], 0)))


def unscaled_coefficients(weights, feature_names, mean, std) -> pd.Series:
    """Convert weights on standardized columns back to original units.

    For a 0/1 dummy the result is the bikes added when that level is on,
    relative to the reference level. The intercept is the prediction when
    every original column is 0.
    """
    weights = np.asarray(weights)
    slopes = weights[1:] / np.asarray(std)
    intercept = weights[0] - np.sum(slopes * np.asarray(mean))
    return pd.Series(np.concatenate([[intercept], slopes]), index=feature_names)
