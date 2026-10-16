"""Phase 4 mechanics on synthetic data (no project data, no official numbers)."""

import numpy as np
import pytest

from src.features import Phase1Features
from src.phase1_gradient_descent import largest_curvature
from src.phase4_regularization import PenalizedGD, lambda_max_l1
from test_phase2 import synthetic_hours


@pytest.fixture(scope="module")
def design():
    data = synthetic_hours()
    X = Phase1Features("mnth", False).fit(data).transform(data)
    return X, data["cnt"].to_numpy(float)


def fit(X, y, lam, rho, init=None):
    rate = 1.9 / (largest_curvature(X) + lam * (1 - rho))
    return PenalizedGD(lam, rho, rate, 200_000, 1e-6).fit(X, y, init_weights=init)


def test_ridge_matches_its_closed_form(design):
    X, y = design
    lam = 0.5
    result = fit(X, y, lam, 0.0)
    penalty = lam * np.eye(X.shape[1])
    penalty[0, 0] = 0.0  # intercept not penalized
    exact = np.linalg.solve(X.T @ X / len(y) + penalty, X.T @ y / len(y))
    assert result.converged
    assert np.max(np.abs(result.weights - exact)) < 1e-5


def test_lasso_satisfies_optimality_conditions(design):
    X, y = design
    lam = 2.0
    w = fit(X, y, lam, 1.0).weights
    gradient = X.T @ (X @ w - y) / len(y)
    nonzero = np.flatnonzero(w[1:]) + 1
    zero = np.setdiff1d(np.arange(1, len(w)), nonzero)
    assert abs(gradient[0]) < 1e-5                      # intercept: plain gradient is 0
    assert np.allclose(gradient[nonzero], -lam * np.sign(w[nonzero]), atol=1e-5)
    assert np.all(np.abs(gradient[zero]) <= lam + 1e-5)  # zeros: |gradient| <= lam
    assert 0 < len(nonzero) < X.shape[1] - 1           # some, not all, weights survive


def test_lambda_max_zeroes_every_penalized_weight(design):
    X, y = design
    for rho in (1.0, 0.5):
        lam = lambda_max_l1(X, y, rho)
        start = np.zeros(X.shape[1])
        start[0] = y.mean()
        assert np.count_nonzero(fit(X, y, lam * 1.0001, rho, start).weights[1:]) == 0
        assert np.count_nonzero(fit(X, y, lam * 0.9, rho, start).weights[1:]) > 0
