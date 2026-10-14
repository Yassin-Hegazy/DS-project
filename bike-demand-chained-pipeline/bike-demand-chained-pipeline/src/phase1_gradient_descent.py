"""Phase 1 — linear regression with gradient descent implemented from scratch."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

# A loss this many times larger than the starting loss is treated as divergence.
DIVERGENCE_FACTOR = 1e6


@dataclass
class GDResult:
    weights: np.ndarray
    learning_rate: float
    iterations: int
    loss_history: list[float]
    grad_history: list[float]
    status: str  # "converged", "diverged" or "max_iter"

    @property
    def converged(self) -> bool:
        return self.status == "converged"


class LinearRegressionGD:
    """Full-batch gradient descent on L(w) = ||Xw - y||^2 / (2n).

    X must already contain the intercept column. Stopping rule: stop when
    the largest absolute gradient entry is below tol (converged), when the
    loss becomes non-finite or explodes (diverged), or after max_iter updates.
    """

    def __init__(self, learning_rate=1e-2, max_iter=10_000, tol=1e-8):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.loss_history = []

    def loss_and_gradient(self, X, y, w):
        """MSE loss and its gradient X^T (Xw - y) / n.

        This is the hook the asymmetric-cost bonus overrides.
        """
        n = len(y)
        residual = X @ w - y
        loss = residual @ residual / (2 * n)
        gradient = X.T @ residual / n
        return loss, gradient

    def fit(self, X, y, init_weights=None) -> GDResult:
        if init_weights is None:
            w = np.zeros(X.shape[1])
        else:
            w = np.array(init_weights, dtype=float)
        loss_history, grad_history = [], []
        status = "max_iter"
        iterations = 0
        while True:
            loss, gradient = self.loss_and_gradient(X, y, w)
            max_grad = float(np.max(np.abs(gradient)))
            loss_history.append(float(loss))
            grad_history.append(max_grad)
            if not np.isfinite(loss) or loss > DIVERGENCE_FACTOR * loss_history[0]:
                status = "diverged"
                break
            if max_grad < self.tol:
                status = "converged"
                break
            if iterations == self.max_iter:
                break
            w = w - self.learning_rate * gradient
            iterations += 1
        self.weights = w
        self.loss_history = loss_history
        return GDResult(w, self.learning_rate, iterations, loss_history,
                        grad_history, status)

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("Model has not been fitted.")
        return X @ self.weights


def largest_curvature(X) -> float:
    """Largest eigenvalue of X^T X / n, the largest curvature of the MSE loss.

    Gradient descent on the MSE loss is stable only for learning rates
    below 2 / largest_curvature.
    """
    return float(np.linalg.eigvalsh(X.T @ X / len(X))[-1])


def save_phase1_artifact(path, features, result: GDResult, settings: dict,
                         metrics: dict) -> None:
    """Write everything Phase 2 consumes: weights in feature order, the fitted
    encoder/scaler, and the GD hyperparameters that produced the weights."""
    artifact = {
        "feature_names": features.feature_names,
        "weights": result.weights.tolist(),
        "learning_rate": result.learning_rate,
        "iterations": result.iterations,
        "status": result.status,
        "final_loss": result.loss_history[-1],
        "final_max_abs_gradient": result.grad_history[-1],
        "settings": settings,
        "metrics": metrics,
        "features": features.to_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2))


def load_phase1_artifact(path) -> dict:
    artifact = json.loads(Path(path).read_text())
    artifact["weights"] = np.asarray(artifact["weights"])
    return artifact
