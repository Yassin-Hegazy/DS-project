"""Phase 3 mechanics on synthetic data (no project data, no official numbers)."""

import numpy as np

from src.data import chronological_split
from src.features import Phase1Features
from src.phase2_polynomial import Phase2Features
from src.phase3_bias_variance import choose_target, learning_curve
from test_phase2 import synthetic_hours


def test_chronological_split_keeps_whole_days():
    data = synthetic_hours()
    earlier, later, first_day = chronological_split(data)
    assert set(earlier["dteday"]).isdisjoint(set(later["dteday"]))
    assert earlier["dteday"].max() < first_day <= later["dteday"].min()
    days = data["dteday"].nunique()
    assert later["dteday"].nunique() == round(0.2 * days)


def test_exact_combinations_are_dropped():
    data = synthetic_hours()
    phase1 = Phase1Features("mnth", False).fit(data)
    # hour x month contains hour x season: 23 hours x 3 seasons are implied.
    features = Phase2Features(phase1, ["hour x season", "hour x month"]).fit(data)
    assert len(features.dropped) == 23 * 3
    X = features.transform(data)
    assert np.linalg.matrix_rank(X) == X.shape[1]


def test_learning_curve_and_target_rule():
    data = synthetic_hours()
    train, val = data.iloc[:2400], data.iloc[2400:]
    y_train, y_val = train["cnt"].to_numpy(float), val["cnt"].to_numpy(float)
    curve = learning_curve(["hour x workingday"], "mnth", False, train, y_train, val, y_val,
                           [0.5, 1.0], 0, 1.9, 50_000, 1e-4)
    assert list(curve["n_train"]) == [1200, 2400]
    assert (curve["status"] == "converged").all()
    sweep = curve.assign(n_features=[10, 20], val_r2=[0.900, 0.901])
    sweep.loc[0, "status"] = "max_iter"
    assert choose_target(sweep, 0.002)["n_features"] == 20  # never a flagged fit
