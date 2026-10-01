"""Phase 1 — linear regression with gradient descent implemented from scratch."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class GDResult:
    weights: np.ndarray
    learning_rate: float
    iterations: int
    loss_history: list[float]
    converged: bool


class LinearRegressionGD:
    def __init__(self, learning_rate=1e-2, max_iter=10_000, tol=1e-8):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.loss_history = []

    def fit(self, X, y, init_weights=None):
        """TODO: implement and understand every line before submission."""
        raise NotImplementedError

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("Model has not been fitted.")
        return X @ self.weights
