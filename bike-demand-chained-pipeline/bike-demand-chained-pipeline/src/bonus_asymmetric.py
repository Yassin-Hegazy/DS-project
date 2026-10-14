"""Phase 1 optional bonus — asymmetric-cost loss (under-prediction costs k times more).

Separate from the main pipeline: nothing here is read by Phase 2 or used for
the submission file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import regression_metrics
from src.phase1_gradient_descent import LinearRegressionGD


def asymmetric_cost(y, pred, under_cost=3.0) -> float:
    """Mean of c * r^2 / 2, with r = pred - y and c = under_cost when r < 0."""
    residual = pred - y
    weight = np.where(residual < 0, under_cost, 1.0)
    return float(np.mean(weight * residual ** 2) / 2)


class AsymmetricLinearRegressionGD(LinearRegressionGD):
    """Phase 1 gradient descent with only the loss/gradient hook replaced.

    Gradient: X^T (c * r) / n, where c = under_cost for under-predicted rows.
    under_cost = 1 gives exactly the Phase 1 MSE model.
    """

    def __init__(self, under_cost=3.0, **kwargs):
        super().__init__(**kwargs)
        self.under_cost = under_cost

    def loss_and_gradient(self, X, y, w):
        n = len(y)
        residual = X @ w - y
        weight = np.where(residual < 0, self.under_cost, 1.0)
        loss = np.sum(weight * residual ** 2) / (2 * n)
        gradient = X.T @ (weight * residual) / n
        return loss, gradient


def finite_difference_gradient(model, X, y, w, eps=1e-4) -> np.ndarray:
    """Central-difference gradient, used only to check the analytic one."""
    numeric = np.zeros_like(w)
    for j in range(len(w)):
        step = np.zeros_like(w)
        step[j] = eps
        loss_plus, _ = model.loss_and_gradient(X, y, w + step)
        loss_minus, _ = model.loss_and_gradient(X, y, w - step)
        numeric[j] = (loss_plus - loss_minus) / (2 * eps)
    return numeric


def prediction_summary(y, pred, under_cost=3.0) -> dict[str, float]:
    """Accuracy and operator-facing numbers for one model's predictions."""
    residual = pred - y
    return {
        **regression_metrics(y, pred),
        "asymmetric_cost": asymmetric_cost(y, pred, under_cost),
        "mean_prediction": float(np.mean(pred)),
        "under_prediction_rate": float(np.mean(residual < 0)),
        "mean_shortfall_bikes": float(np.mean(np.maximum(-residual, 0))),
        "mean_surplus_bikes": float(np.mean(np.maximum(residual, 0))),
    }


def shift_by_hour(df, y, pred_mse, pred_asym) -> pd.DataFrame:
    """Mean actual demand, both models' predictions and the shift per hour x day type."""
    table = pd.DataFrame({
        "hr": df["hr"].to_numpy(),
        "workingday": df["workingday"].to_numpy(),
        "actual": y,
        "pred_mse": pred_mse,
        "pred_asym": pred_asym,
        "shift": pred_asym - pred_mse,
    })
    return table.groupby(["workingday", "hr"]).mean().reset_index()


def buffer_by_hour(df, y, pred_mse, pred_asym) -> pd.DataFrame:
    """Per hour x day type: upward shift of the asymmetric model over its MSE
    model, and the asymmetric model's mean bikes short and idle per hour."""
    error = pred_asym - y
    table = pd.DataFrame({
        "workingday": df["workingday"].to_numpy(),
        "hr": df["hr"].to_numpy(),
        "shift": pred_asym - pred_mse,
        "short": np.maximum(-error, 0),
        "idle": np.maximum(error, 0),
    })
    return table.groupby(["workingday", "hr"]).mean().reset_index()


def residual_structure(df, residual) -> tuple[pd.DataFrame, dict[str, float]]:
    """Split MSE residuals into hour x day-type cell means and within-cell spread.

    A cell mean far from zero is systematic error a main-effects model cannot
    remove (underfitting). The within-cell spread is what remains: genuine
    demand variability plus any other effect the model is missing.
    """
    frame = pd.DataFrame({
        "hr": df["hr"].to_numpy(),
        "workingday": df["workingday"].to_numpy(),
        "residual": residual,
    })
    cells = frame.groupby(["workingday", "hr"])["residual"]
    cell_mean = cells.transform("mean")
    cell_table = cells.agg(
        mean_residual="mean",
        within_sd=lambda r: float(np.sqrt(np.mean((r - r.mean()) ** 2))),
    ).reset_index()
    total = float(np.mean(residual ** 2))
    between = float(np.mean(cell_mean ** 2))
    within = float(np.mean((residual - cell_mean) ** 2))
    shares = {
        "mean_squared_residual": total,
        "between_cell_share": between / total,
        "within_cell_share": within / total,
    }
    return cell_table, shares
