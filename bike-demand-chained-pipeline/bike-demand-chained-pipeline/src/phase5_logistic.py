"""Phase 5 — logistic regression on a contextual high-demand label."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.phase1_gradient_descent import LinearRegressionGD, largest_curvature

# "High demand" is judged against hours of the same kind: same hour of day,
# same day type (working / not) and same year.
GROUP_COLUMNS = ["hr", "workingday", "yr"]
HIGH_QUANTILE = 0.75


def high_demand_cutoffs(train_df: pd.DataFrame, columns=GROUP_COLUMNS,
                        quantile=HIGH_QUANTILE) -> pd.Series:
    """The cnt cutoff of every group, from training rows only."""
    return train_df.groupby(columns)["cnt"].quantile(quantile).rename("cutoff")


def high_demand_labels(df: pd.DataFrame, cutoffs: pd.Series, columns=GROUP_COLUMNS) -> np.ndarray:
    """1 if the hour's cnt exceeds its group's training cutoff, else 0.

    Validation hours are looked up in the cutoffs learned on training rows;
    nothing is recomputed from validation.
    """
    cutoff = df.join(cutoffs, on=columns)["cutoff"]
    if cutoff.isna().any():
        raise ValueError("a row's group has no training cutoff")
    return (df["cnt"] > cutoff).astype(int).to_numpy()


def label_definition_comparison(train_df: pd.DataFrame) -> pd.DataFrame:
    """Why the label is contextual and includes year (training rows only).

    For three definitions of "above the training 75th percentile": overall
    positive rate, the rate in each year, and how many of the 24 hours of the
    day are never / always positive (a label the clock alone decides).
    """
    definitions = {
        "global cutoff": [],
        "hour x working day": ["hr", "workingday"],
        "hour x working day x year (chosen)": GROUP_COLUMNS,
    }
    rows = []
    for name, columns in definitions.items():
        if columns:
            labels = high_demand_labels(train_df, high_demand_cutoffs(train_df, columns), columns)
        else:
            labels = (train_df["cnt"] > train_df["cnt"].quantile(HIGH_QUANTILE)).astype(int).to_numpy()
        by_hour = pd.Series(labels).groupby(train_df["hr"].to_numpy()).mean()
        by_year = pd.Series(labels).groupby(train_df["yr"].to_numpy()).mean()
        rows.append({"definition": name, "positive_rate": labels.mean(),
                     "rate_2011": by_year.loc[0], "rate_2012": by_year.loc[1],
                     "hours_never_positive": int((by_hour == 0).sum()),
                     "hours_always_positive": int((by_hour == 1).sum()),
                     "lowest_hourly_rate": by_hour.min(), "highest_hourly_rate": by_hour.max()})
    return pd.DataFrame(rows)


def sigmoid(z):
    """1 / (1 + e^-z), written so that large |z| cannot overflow."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class LogisticRegressionGD(LinearRegressionGD):
    """Phase 1's gradient descent loop on the log-loss.

    p = sigmoid(Xw); loss = -mean(y log p + (1 - y) log(1 - p));
    gradient = X^T (p - y) / m. p changes with w at every step, so there is
    no fixed table of sums as for the squared error: the gradient is
    computed explicitly at every update.
    """

    QUADRATIC = False

    def loss_and_gradient(self, X, y, w):
        z = X @ w
        # log(1 + e^z) - y z equals -[y log p + (1 - y) log(1 - p)], without
        # taking the log of a probability that rounds to 0.
        loss = np.mean(np.logaddexp(0.0, z) - y * z)
        gradient = X.T @ (sigmoid(z) - y) / len(y)
        return loss, gradient

    def predict_proba(self, X):
        if self.weights is None:
            raise RuntimeError("Model has not been fitted.")
        return sigmoid(X @ self.weights)


def logistic_curvature_bound(X) -> float:
    """The log-loss curvature is X^T diag(p(1-p)) X / m and p(1-p) <= 1/4, so
    it never exceeds largest_curvature(X) / 4."""
    return largest_curvature(X) / 4


def classification_scores(y, prob, threshold) -> dict:
    """Confusion matrix and scores at one probability cutoff."""
    predicted = (prob >= threshold).astype(int)
    tp = int(np.sum((predicted == 1) & (y == 1)))
    fp = int(np.sum((predicted == 1) & (y == 0)))
    fn = int(np.sum((predicted == 0) & (y == 1)))
    tn = int(np.sum((predicted == 0) & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall,
            "f1": f1, "roc_auc": float(roc_auc_score(y, prob)),
            "always_0_accuracy": float(np.mean(y == 0))}


def threshold_scan(y, prob, thresholds) -> pd.DataFrame:
    return pd.DataFrame([classification_scores(y, prob, t) for t in thresholds])


def best_f1_threshold(scan: pd.DataFrame) -> float:
    """Cutoff with the highest F1 (ties: the one closest to 0.5)."""
    best = scan[scan["f1"] == scan["f1"].max()]
    return float(best.loc[(best["threshold"] - 0.5).abs().idxmin(), "threshold"])


def save_phase5_artifact(path, artifact: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, default=float))
