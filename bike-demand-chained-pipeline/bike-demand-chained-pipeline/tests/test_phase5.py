"""Phase 5 mechanics on synthetic data (no project data, no official numbers)."""

import numpy as np
import pandas as pd

from src.bonus_asymmetric import finite_difference_gradient
from src.phase5_logistic import (
    LogisticRegressionGD,
    best_f1_threshold,
    high_demand_cutoffs,
    high_demand_labels,
    logistic_curvature_bound,
    sigmoid,
    threshold_scan,
)
from test_phase2 import synthetic_hours


def logistic_data(seed=0, m=2000):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(m), rng.normal(size=(m, 3))])
    true_w = np.array([-1.0, 2.0, -1.0, 0.0])
    y = (rng.uniform(size=m) < sigmoid(X @ true_w)).astype(float)
    return X, y, true_w


def test_gradient_matches_finite_differences():
    X, y, _ = logistic_data()
    model = LogisticRegressionGD()
    w = np.array([0.3, -0.2, 0.5, 0.1])
    analytic = model.loss_and_gradient(X, y, w)[1]
    numeric = finite_difference_gradient(model, X, y, w, eps=1e-6)
    assert np.max(np.abs(analytic - numeric)) < 1e-7


def test_gd_with_curvature_step_recovers_the_weights():
    X, y, true_w = logistic_data(m=20_000)
    step = 1.9 / logistic_curvature_bound(X)
    result = LogisticRegressionGD(step, 50_000, 1e-6).fit(X, y)
    assert result.converged
    assert not np.any(np.diff(result.loss_history) > 1e-12)  # the safe step never raises the loss
    assert np.max(np.abs(result.weights - true_w)) < 0.1


def test_sigmoid_is_stable_for_large_inputs():
    assert sigmoid(np.array([-1000.0]))[0] == 0.0
    assert sigmoid(np.array([1000.0]))[0] == 1.0


def test_labels_use_training_cutoffs_only():
    data = synthetic_hours()
    train, val = data.iloc[:2400], data.iloc[2400:]
    cutoffs = high_demand_cutoffs(train)
    labels = high_demand_labels(val, cutoffs)
    looked_up = val.join(cutoffs, on=["hr", "workingday", "yr"])["cutoff"].to_numpy()
    assert np.array_equal(labels, (val["cnt"].to_numpy() > looked_up).astype(int))
    # About a quarter: small groups and tied counts move it a little either way.
    assert 0.2 < high_demand_labels(train, cutoffs).mean() < 0.3


def test_threshold_scan_and_best_f1():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.4, 0.35, 0.8])
    scan = threshold_scan(y, prob, [0.3, 0.5, 0.9])
    assert list(scan["f1"].round(3)) == [0.8, 0.667, 0.0]
    assert best_f1_threshold(scan) == 0.3
    assert scan.loc[0, "always_0_accuracy"] == 0.5
