"""Phase 4 — L1, L2 and Elastic Net experiments.

Objective (one convention for all three methods; the intercept w[0] is never
penalized):

    J(w) = ||Xw - y||^2 / (2n) + lam * ( rho * sum_j |w_j| + (1 - rho) * sum_j w_j^2 / 2 )

Ridge: rho = 0. Lasso: rho = 1. Elastic Net: 0 < rho < 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import regression_metrics
from src.operator_checks import errors_by_demand_decile, top_decile_shortfall
from src.phase1_gradient_descent import (
    DIVERGENCE_FACTOR,
    GDResult,
    LinearRegressionGD,
    largest_curvature,
)
from src.phase2_polynomial import Phase2Features


class PenalizedGD(LinearRegressionGD):
    """Phase 1's gradient descent with an L2 and/or L1 penalty.

    Each update: a gradient step on the smooth part (the squared error plus
    the L2 penalty), then soft-thresholding for the L1 part: every penalized
    weight is pulled toward 0 by learning_rate * lam * rho, and set to exactly
    0 if it would cross 0. With rho = 0 this is plain gradient descent.

    Stopping rule: the largest entry of (w - w_next) / learning_rate is below
    tol. Without an L1 part this is exactly the gradient; with one, it is the
    size of the move the algorithm would still make, and it is 0 only at the
    minimum.
    """

    def __init__(self, lam, l1_ratio, learning_rate, max_iter, tol):
        super().__init__(learning_rate=learning_rate, max_iter=max_iter, tol=tol)
        self.lam = lam
        self.l1_ratio = l1_ratio

    def fit(self, X, y, init_weights=None) -> GDResult:
        w = np.zeros(X.shape[1]) if init_weights is None else np.array(init_weights, float)
        squared_error = self._objective(X, y)  # w -> (MSE loss, H w - b)
        self._assert_matches_readable_form(squared_error, X, y, w)
        penalized = np.ones(len(w))
        penalized[0] = 0.0  # the intercept is not penalized
        l1 = self.lam * self.l1_ratio
        l2 = self.lam * (1 - self.l1_ratio)
        threshold = self.learning_rate * l1 * penalized
        objective_history, move_history = [], []
        status, iterations = "max_iter", 0
        while True:
            loss, gradient = squared_error(w)
            # Ridge part: its gradient adds l2 to the diagonal of H (not the intercept's).
            gradient = gradient + l2 * penalized * w
            objective = (loss + l2 / 2 * np.sum(penalized * w ** 2)
                         + l1 * np.sum(penalized * np.abs(w)))
            step = w - self.learning_rate * gradient
            w_next = np.sign(step) * np.maximum(np.abs(step) - threshold, 0.0)
            move = float(np.max(np.abs(w - w_next))) / self.learning_rate
            objective_history.append(float(objective))
            move_history.append(move)
            if not np.isfinite(objective) or objective > DIVERGENCE_FACTOR * objective_history[0]:
                status = "diverged"
                break
            if move < self.tol:
                status = "converged"
                break
            if iterations == self.max_iter:
                break
            w = w_next
            iterations += 1
        if status != "diverged":
            self._assert_matches_readable_form(squared_error, X, y, w)
        self.weights = w
        self.loss_history = objective_history
        return GDResult(w, self.learning_rate, iterations, objective_history,
                        move_history, status)


def lambda_max_l1(X, y, l1_ratio) -> float:
    """Smallest lam at which every penalized weight is exactly 0 (rho > 0).

    At w = (mean of y, 0, ..., 0) the squared-error gradient of a centered
    column j is -X_j^T (y - mean) / n; soft-thresholding keeps all weights at
    0 once lam * rho is at least the largest of these in absolute value.
    """
    gradient = X[:, 1:].T @ (y - y.mean()) / len(y)
    return float(np.max(np.abs(gradient)) / l1_ratio)


def regularization_path(X_train, y_train, X_val, y_val, lambdas, l1_ratio,
                        step_multiplier, max_iter, tol, decile_edges):
    """Fit one method for every lam, from the largest down.

    The first fit starts with every penalized weight at 0 (intercept at the
    mean); every later fit starts from the previous fit's weights.
    """
    curvature = largest_curvature(X_train)
    w = np.zeros(X_train.shape[1])
    w[0] = y_train.mean()
    rows, weights = [], []
    for lam in lambdas:
        # The penalty's L2 part raises the largest curvature by lam * (1 - rho).
        learning_rate = step_multiplier / (curvature + lam * (1 - l1_ratio))
        result = PenalizedGD(lam, l1_ratio, learning_rate, max_iter, tol).fit(
            X_train, y_train, init_weights=w)
        w = result.weights
        weights.append(w)
        rows.append({"lambda": lam, "l1_ratio": l1_ratio,
                     **score_weights(w, X_train, y_train, X_val, y_val, decile_edges),
                     "status": result.status, "iterations": result.iterations,
                     "final_max_move": result.grad_history[-1],
                     "objective_ever_increased":
                         bool(np.any(np.diff(result.loss_history) > 0))})
    return pd.DataFrame(rows), weights


def score_weights(w, X_train, y_train, X_val, y_val, decile_edges) -> dict:
    """Scores, sparsity and operator checks for one weight vector."""
    pred_train, pred_val = X_train @ w, X_val @ w
    train, val = regression_metrics(y_train, pred_train), regression_metrics(y_val, pred_val)
    by_predicted = errors_by_demand_decile(y_val, pred_val, decile_edges, bin_on="predicted")
    top_predicted = by_predicted.loc[by_predicted["decile"] == 10, "mean_shortfall_bikes"]
    return {
        "nonzero": int(np.count_nonzero(w[1:])),
        "train_r2": train["r2"], "val_r2": val["r2"],
        "train_rmse": train["rmse"], "val_rmse": val["rmse"],
        "top_decile_shortfall": top_decile_shortfall(y_val, pred_val, decile_edges),
        "top_predicted_decile_shortfall":
            float(top_predicted.iloc[0]) if len(top_predicted) else np.nan,
    }


def select(path: pd.DataFrame, r2_tolerance: float | None) -> pd.Series:
    """The chosen fit on one path; only converged fits are eligible.

    r2_tolerance None (Ridge): the highest validation R2. Otherwise (Lasso,
    Elastic Net): the largest lam whose validation R2 is within r2_tolerance
    of that path's best.
    """
    converged = path[path["status"] == "converged"]
    if r2_tolerance is None:
        return converged.loc[converged["val_r2"].idxmax()]
    close = converged[converged["val_r2"] >= converged["val_r2"].max() - r2_tolerance]
    return close.loc[close["lambda"].idxmax()]


# ---------------------------------------------------------------------------
# Column verdicts
# ---------------------------------------------------------------------------

ORIGINAL_COLUMNS = ["instant", "dteday", "season", "yr", "mnth", "hr", "holiday",
                    "weekday", "workingday", "weathersit", "temp", "atemp", "hum",
                    "windspeed"]

# How each column not (directly) in the target is added back.
ADD_BACK_GROUPS = {
    "instant": "add instant",
    "dteday": "add dteday (elapsed days, day of month)",
    "atemp": "add atemp",
    "workingday": "add workingday main effect",
    "season": "add season main effect",
}

# A column on its own: one-hot (reference level dropped) or numeric.
ONE_HOT_ALONE = {"hr", "weekday", "weathersit", "mnth", "season"}
ALONE_COLUMNS = {"dteday": ["days_since_start", "day"]}


def alone_design(train_df, val_df, column):
    """Intercept plus the column's own encoding, standardized on training rows."""
    prepared = {name: Phase2Features._prepare(df) for name, df in
                (("train", train_df), ("val", val_df))}
    blocks = {"train": [], "val": []}
    for source in ALONE_COLUMNS.get(column, [column]):
        if column in ONE_HOT_ALONE:
            levels = sorted(prepared["train"][source].unique())[1:]
            for name in blocks:
                blocks[name] += [prepared[name][source].eq(v).to_numpy(float) for v in levels]
        else:
            for name in blocks:
                blocks[name].append(prepared[name][source].to_numpy(float))
    raw_train, raw_val = np.column_stack(blocks["train"]), np.column_stack(blocks["val"])
    mean, std = raw_train.mean(axis=0), raw_train.std(axis=0)
    def design(raw):
        return np.column_stack([np.ones(len(raw)), (raw - mean) / std])
    return design(raw_train), design(raw_val)


def refit_like(features, train_df, y_train, val_df, y_val, start_names, start_weights,
               lam, l1_ratio, step_multiplier, max_iter, tol, decile_edges):
    """Fit a modified design with the verdict model's settings, starting from
    the verdict model's weights wherever a column has the same name."""
    X_tr, X_va = features.transform(train_df), features.transform(val_df)
    lookup = dict(zip(start_names, start_weights))
    init = np.array([lookup.get(name, 0.0) for name in features.feature_names])
    learning_rate = step_multiplier / (largest_curvature(X_tr) + lam * (1 - l1_ratio))
    result = PenalizedGD(lam, l1_ratio, learning_rate, max_iter, tol).fit(
        X_tr, y_train, init_weights=init)
    return result, score_weights(result.weights, X_tr, y_train, X_va, y_val, decile_edges)


def verdict(in_model: bool, ablation_change: float, add_back_change: float,
            add_back_all_dropped: bool, alone_r2: float, r2_tolerance: float,
            alone_r2_threshold: float) -> str:
    """Pre-registered rule turning the numbers into a verdict."""
    if in_model:
        if ablation_change <= -r2_tolerance:
            return "useful"
    else:
        if add_back_all_dropped:
            return "redundant (exact combination of other columns)"
        if add_back_change >= r2_tolerance:
            return "useful (missing from the model)"
    return "redundant" if alone_r2 >= alone_r2_threshold else "uninformative"


def save_phase4_artifact(path, artifact: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, default=float))


def load_phase4_artifact(path) -> dict:
    return json.loads(Path(path).read_text())
