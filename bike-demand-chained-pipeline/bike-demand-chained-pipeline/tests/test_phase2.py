"""Phase 2 mechanics on synthetic data (no project data, no official numbers)."""

import numpy as np
import pandas as pd
import pytest

from src.features import Phase1Features
from src.operator_checks import demand_decile_edges
from src.phase1_gradient_descent import LinearRegressionGD, largest_curvature
from src.phase2_polynomial import (
    PHASE2_GROUPS,
    Phase2Features,
    fit_and_score,
    group_terms,
)


def synthetic_hours(n=3000, seed=0):
    """Hourly rows shaped like train.csv, with an hour x working-day effect."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2011-01-01", "2012-12-19")
    days = days[days.day <= 19]
    date = pd.Series(rng.choice(days, n))
    hr = rng.integers(0, 24, n)
    weekday = ((date.dt.dayofweek + 1) % 7).to_numpy()
    holiday = (date.dt.strftime("%m-%d").isin(["07-04", "01-17"])
               & (weekday >= 1) & (weekday <= 5)).astype(int).to_numpy()
    workingday = ((weekday >= 1) & (weekday <= 5) & (holiday == 0)).astype(int)
    temp = rng.uniform(0.05, 0.95, n)
    peak = np.isin(hr, [8, 17, 18])
    cnt = (50 + 200 * peak * workingday + 150 * temp + 60 * (date.dt.year == 2012)
           + rng.normal(0, 20, n))
    return pd.DataFrame({
        "instant": np.arange(n),
        "dteday": date.dt.strftime("%Y-%m-%d"),
        "season": (date.dt.month - 1) // 3 + 1,
        "yr": (date.dt.year - 2011).to_numpy(),
        "mnth": date.dt.month.to_numpy(),
        "hr": hr,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": rng.integers(1, 4, n),
        "temp": temp,
        "atemp": temp,
        "hum": rng.uniform(0.2, 1.0, n),
        "windspeed": rng.uniform(0.0, 0.5, n),
        "cnt": np.clip(cnt, 0, None).round(),
    })


@pytest.fixture(scope="module")
def fitted():
    data = synthetic_hours()
    train, val = data.iloc[:2400], data.iloc[2400:]
    y_train, y_val = train["cnt"].to_numpy(float), val["cnt"].to_numpy(float)
    phase1 = Phase1Features("mnth", False).fit(train)
    X1 = phase1.transform(train)
    phase1_result = LinearRegressionGD(1.9 / largest_curvature(X1), 50_000, 1e-4).fit(X1, y_train)
    return train, val, y_train, y_val, phase1, phase1_result


def test_terms_never_repeat_a_column():
    levels = {"hr": list(range(1, 24)), "season": [2, 3, 4],
              "hour_block": [1, 2, 3, 4], "weathersit": [2, 3]}
    for group in PHASE2_GROUPS:
        for term in group_terms(group, levels):
            columns = [column for column, _ in term]
            assert len(set(columns)) == len(columns)


def test_warm_start_and_convergence(fitted):
    train, val, y_train, y_val, phase1, phase1_result = fitted
    features = Phase2Features(phase1, list(PHASE2_GROUPS)).fit(train)
    edges = demand_decile_edges(y_train)
    # fit_and_score itself asserts that the warm start reproduces Phase 1.
    result, row = fit_and_score(features, train, y_train, val, y_val,
                                phase1_result.weights, 1.9, 50_000, 1e-4, edges)
    assert row["rank_ok"]
    assert result.converged
    assert row["gap_to_exact_bikes"] < 0.1
    phase1_val = phase1.transform(val) @ phase1_result.weights
    phase1_rmse = np.sqrt(np.mean((phase1_val - y_val) ** 2))
    assert row["val_rmse"] < phase1_rmse  # the hour x working-day effect is learned


def test_artifact_round_trip(fitted):
    train, val, *_ = fitted
    features = Phase2Features(fitted[4], list(PHASE2_GROUPS)).fit(train)
    restored = Phase2Features.from_dict(features.to_dict())
    assert np.array_equal(restored.transform(val), features.transform(val))
    assert list(features.lineage) == features.feature_names
    assert features.lineage["hr=8*workingday*temp"] == ["hr", "temp", "workingday"]


def test_no_new_groups_is_phase1(fitted):
    train, val, *_ = fitted
    phase1 = fitted[4]
    features = Phase2Features(phase1, []).fit(train)
    assert np.array_equal(features.transform(val), phase1.transform(val))


def test_precomputed_and_explicit_gradient_reach_the_same_weights(fitted):
    """Phase 1's loop (H w - b) and the explicit X^T (Xw - y) / n loop."""
    from src.bonus_asymmetric import AsymmetricLinearRegressionGD
    train, _, y_train, _, phase1, phase1_result = fitted
    X = phase1.transform(train)
    explicit = AsymmetricLinearRegressionGD(
        under_cost=1.0, learning_rate=phase1_result.learning_rate,
        max_iter=50_000, tol=1e-4).fit(X, y_train)
    assert explicit.iterations == phase1_result.iterations
    assert np.max(np.abs(explicit.weights - phase1_result.weights)) < 1e-8
