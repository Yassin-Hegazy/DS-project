"""Phase 3 — empirical bias/variance diagnosis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import Phase1Features
from src.metrics import regression_metrics
from src.operator_checks import demand_decile_edges
from src.phase1_gradient_descent import LinearRegressionGD, largest_curvature
from src.phase2_polynomial import Phase2Features, fit_and_score

# Groups added on top of Phase 2's chosen design to probe the high-variance
# side: the four rejected in Phase 2 Step A, then one degree-4 group.
SWEEP_EXTENSIONS = {
    "hour x holiday": 2,
    "hour x weekday": 2,
    "hour x month": 2,
    "hour x weathersit": 2,
    "hour x workingday x yr x season": 4,
}


def sweep_steps(phase2_artifact: dict) -> list[dict]:
    """The complexity sweep: Phase 2's steps in its own order (read from its
    artifact), then the extensions on top of Phase 2's chosen design."""
    steps, groups = [], []
    for record in phase2_artifact["sequence"]:
        label = record["added group"]
        if label != "(Phase 1)":
            groups.append(label)
        steps.append({"added group": label, "degree": int(record["degree"]),
                      "groups": list(groups), "part": "Phase 2"})
    # Extensions build on Phase 2's chosen design and start from its fit.
    groups = list(phase2_artifact["groups"])
    chosen = next(s["added group"] for s in steps if s["groups"] == groups)
    for group, degree in SWEEP_EXTENSIONS.items():
        groups.append(group)
        steps.append({"added group": group, "degree": degree, "groups": list(groups),
                      "part": "extension", "start_from": chosen})
    return steps


def fit_phase1(train_df, y_train, seasonal, use_trend, step_multiplier, max_iter, tol):
    """Phase 1 representation and GD rule, refitted on another training set."""
    features = Phase1Features(seasonal=seasonal, use_trend=use_trend).fit(train_df)
    X = features.transform(train_df)
    model = LinearRegressionGD(step_multiplier / largest_curvature(X), max_iter, tol)
    return features, model.fit(X, y_train)


def run_sweep(steps, train_df, y_train, val_df, y_val, phase1_features, phase1_weights,
              step_multiplier, max_iter, tol):
    """Fit and score every sweep step on one split.

    Each step starts from phase1_weights plus zeros, as in Phase 2; a step
    with "start_from" starts from that earlier step's fitted weights instead.
    """
    edges = demand_decile_edges(y_train)
    rows, fitted = [], {}
    for step in steps:
        features = Phase2Features(phase1_features, step["groups"]).fit(train_df)
        if "start_from" in step:
            base_features, base_result = fitted[step["start_from"]]
            base_weights = base_result.weights
        else:
            base_features, base_weights = phase1_features, phase1_weights
        result, row = fit_and_score(features, train_df, y_train, val_df, y_val,
                                    base_weights, step_multiplier, max_iter, tol, edges,
                                    base_features=base_features)
        fitted[step["added group"]] = (features, result)
        rows.append({"added group": step["added group"], "degree": step["degree"],
                     "part": step["part"], **row})
    return pd.DataFrame(rows), fitted


def group_support(features: Phase2Features, train_df, group: str) -> dict:
    """Training-row support of one group's kept terms (see term_support)."""
    support = features.term_support(train_df)
    rows = support.loc[support["group"] == group, "support_rows"]
    return {"terms_kept": int(len(rows)),
            "min_support_rows": int(rows.min()) if len(rows) else None,
            "terms_under_30_rows": int((rows < 30).sum())}


def learning_curve(groups, seasonal, use_trend, train_df, y_train, val_df, y_val,
                   fractions, seed, step_multiplier, max_iter, tol) -> pd.DataFrame:
    """Train/validation error of one design as the training set grows.

    Each subset is a prefix of one seeded shuffle of the training portion, so
    smaller subsets are contained in larger ones. Everything (encoder, scaler,
    centering means) is refitted on the subset; the validation set is fixed.
    GD starts from zero: the loss is convex, so the start does not change the
    answer, only the number of updates.
    """
    order = np.random.default_rng(seed).permutation(len(train_df))
    rows = []
    for fraction in fractions:
        rows_used = order[: int(round(fraction * len(order)))]
        subset, y_subset = train_df.iloc[rows_used], y_train[rows_used]
        phase1 = Phase1Features(seasonal=seasonal, use_trend=use_trend).fit(subset)
        features = Phase2Features(phase1, groups).fit(subset)
        X_sub, X_val = features.transform(subset), features.transform(val_df)
        model = LinearRegressionGD(step_multiplier / largest_curvature(X_sub), max_iter, tol)
        result = model.fit(X_sub, y_subset)
        exact = np.linalg.lstsq(X_sub, y_subset, rcond=None)[0]
        rows.append({
            "fraction": fraction,
            "n_train": len(subset),
            "n_features": X_sub.shape[1] - 1,
            "dropped_columns": len(features.dropped),
            "status": result.status,
            "iterations": result.iterations,
            "final_max_abs_gradient": result.grad_history[-1],
            "train_rmse": regression_metrics(y_subset, X_sub @ result.weights)["rmse"],
            "val_rmse": regression_metrics(y_val, X_val @ result.weights)["rmse"],
            "val_r2": regression_metrics(y_val, X_val @ result.weights)["r2"],
            # Reference only: the exact least-squares fit on the same subset.
            "exact_val_rmse": regression_metrics(y_val, X_val @ exact)["rmse"],
        })
    return pd.DataFrame(rows)


def choose_target(sweep: pd.DataFrame, r2_tolerance: float) -> pd.Series:
    """Fewest features among converged fits within r2_tolerance of the best
    converged validation R2. Non-converged fits are never chosen."""
    converged = sweep[sweep["status"] == "converged"]
    best = converged["val_r2"].max()
    close = converged[converged["val_r2"] >= best - r2_tolerance]
    return close.sort_values("n_features").iloc[0]


def step_contributions(seeded: pd.DataFrame, chronological: pd.DataFrame) -> pd.DataFrame:
    """Change in validation R2 and top-decile shortfall from adding each group,
    on both splits (each row compared with the step before it)."""
    out = pd.DataFrame({"added group": seeded["added group"], "part": seeded["part"],
                        "n_features": seeded["n_features"]})
    for name, table in (("seeded", seeded), ("chrono", chronological)):
        table = table.set_index("added group").loc[out["added group"]].reset_index()
        out[f"{name}_val_r2_change"] = table["val_r2"].diff().to_numpy()
        out[f"{name}_top_decile_shortfall_change"] = table["top_decile_shortfall"].diff().to_numpy()
        out[f"{name}_converged"] = (table["status"] == "converged").to_numpy()
    return out


def save_phase3_artifact(path, target: pd.Series, features: Phase2Features, weights,
                         diagnosis: dict, noise_candidates: list[str], tables: dict,
                         settings: dict) -> None:
    """Everything Phase 4 consumes: the target design (fitted encoder/scaler,
    feature names, lineage), a converged starting point, the diagnosis and the
    groups the penalty should be allowed to remove."""
    artifact = {
        "target_step": target["added group"],
        "target_degree": int(target["degree"]),
        "target_groups": features.groups,
        "feature_names": features.feature_names,
        "lineage": features.lineage,
        "weights": np.asarray(weights).tolist(),
        "diagnosis": diagnosis,
        "noise_candidates": noise_candidates,
        "tables": {name: table.to_dict(orient="records") for name, table in tables.items()},
        "settings": settings,
        "features": features.to_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, default=float))


def load_phase3_artifact(path) -> dict:
    artifact = json.loads(Path(path).read_text())
    artifact["weights"] = np.asarray(artifact["weights"])
    return artifact
