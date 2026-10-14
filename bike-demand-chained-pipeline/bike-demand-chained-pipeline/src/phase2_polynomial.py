"""Phase 2 — polynomial expansion using the Phase 1 GD implementation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import Phase1Features, group_rare_weather
from src.metrics import regression_metrics
from src.operator_checks import errors_by_demand_decile, top_decile_shortfall
from src.phase1_gradient_descent import LinearRegressionGD

# A cell with fewer training rows than this has an unreliable mean residual:
# with a residual SD near 100 bikes, the mean of 30 rows is still +/- 18 bikes.
MIN_CELL_ROWS = 30

# Candidate combinations for the residual diagnosis: name -> grouping columns.
CANDIDATE_GROUPS = {
    "hour x workingday": ["hr", "workingday"],
    "hour x yr": ["hr", "yr"],
    "hour x season": ["hr", "season"],
    "hour x month": ["hr", "mnth"],
    "hour x weathersit": ["hr", "weathersit"],
    "hour x holiday": ["hr", "holiday"],
    "hour x weekday": ["hr", "weekday"],
    "temp bins (10)": ["temp_bin10"],
    "temp bins (5) x hour": ["temp_bin5", "hr"],
    "hum bins (5) x hour": ["hum_bin5", "hr"],
    "windspeed bins (5) x hour": ["windspeed_bin5", "hr"],
    "temp bins (5) x hum bins (5)": ["temp_bin5", "hum_bin5"],
}


def quantile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    """Cut a training column into (at most) n_bins equal-count bins.

    Tied values can merge neighbouring bins, so fewer bins may result.
    """
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    return pd.cut(values, edges, include_lowest=True)


def hour_block(hours: pd.Series) -> pd.Series:
    """Coarse day parts: 0 night (0-5), 1 morning (6-9), 2 midday (10-15),
    3 evening (16-19), 4 late (20-23)."""
    return pd.cut(hours, [-1, 5, 9, 15, 19, 23], labels=False).astype(int)


def diagnosis_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    """Grouping columns for the residual diagnosis (training rows only)."""
    out = group_rare_weather(train_df)[
        ["hr", "workingday", "yr", "season", "mnth", "weathersit", "holiday", "weekday",
         "temp", "hum"]
    ].copy()
    out["hour_block"] = hour_block(out["hr"])
    out["temp_bin10"] = quantile_bins(train_df["temp"], 10)
    out["temp_bin5"] = quantile_bins(train_df["temp"], 5)
    out["hum_bin5"] = quantile_bins(train_df["hum"], 5)
    out["windspeed_bin5"] = quantile_bins(train_df["windspeed"], 5)
    return out


def between_group_share(residual: np.ndarray, groups: pd.DataFrame) -> dict:
    """Share of residual variance explained by the group means, plus cell support.

    between share = sum over cells of n_cell * (cell mean - overall mean)^2,
                    divided by the total sum of squares of the residuals.
    adjusted share removes what G cells would capture from pure noise, like
    adjusted R^2: 1 - (1 - share) * (n - 1) / (n - G).
    """
    frame = groups.assign(residual=residual)
    cells = frame.groupby(list(groups.columns), observed=True)["residual"]
    cell_mean = cells.transform("mean")
    centered = residual - residual.mean()
    share = float(np.sum((cell_mean - residual.mean()) ** 2) / np.sum(centered ** 2))
    sizes = cells.size()
    n, g = len(residual), len(sizes)
    sparse = sizes < MIN_CELL_ROWS
    return {
        "cells": g,
        "between_share": share,
        "adjusted_share": 1 - (1 - share) * (n - 1) / (n - g),
        "min_cell_rows": int(sizes.min()),
        "median_cell_rows": float(sizes.median()),
        f"cells_under_{MIN_CELL_ROWS}": int(sparse.sum()),
        "rows_in_sparse_cells": int(sizes[sparse].sum()),
    }


def residual_diagnosis(train_df: pd.DataFrame, residual: np.ndarray) -> pd.DataFrame:
    """Rank every candidate combination by its between-group variance share."""
    frame = diagnosis_frame(train_df)
    rows = [{"combination": name, **between_group_share(residual, frame[columns])}
            for name, columns in CANDIDATE_GROUPS.items()]
    table = pd.DataFrame(rows).sort_values("between_share", ascending=False)
    return table.reset_index(drop=True)


def residual_cell_means(train_df: pd.DataFrame, residual: np.ndarray, columns) -> pd.DataFrame:
    """Mean residual and training-row count per cell of one combination."""
    frame = diagnosis_frame(train_df)[columns].assign(residual=residual)
    return (frame.groupby(columns, observed=True)["residual"]
            .agg(mean_residual="mean", rows="size").reset_index())


# Re-ranking candidates. "cells": one indicator per observed cell of these
# columns; "slopes": a separate slope on these (centered) columns inside every
# cell; "columns": plain centered numeric columns ("a*b" is a product).
RERANK_CANDIDATES = {
    "hour x season": {"cells": ["hr", "season"]},
    "hour x temp": {"cells": ["hr"], "slopes": ["temp"]},
    "hour x weathersit": {"cells": ["hr", "weathersit"]},
    "hour-block x weathersit": {"cells": ["hour_block", "weathersit"]},
    "temp x hum": {"columns": ["temp", "hum", "temp*hum"]},
    "temp^2": {"columns": ["temp", "temp*temp"]},
    "hour x workingday x yr": {"cells": ["hr", "workingday", "yr"]},
    "hour x workingday x season": {"cells": ["hr", "workingday", "season"]},
    "hour x workingday x temp": {"cells": ["hr", "workingday"], "slopes": ["temp"]},
}


def _cell_indicators(frame: pd.DataFrame, columns) -> pd.DataFrame:
    keys = frame[columns].astype(str).agg("|".join, axis=1)
    return pd.get_dummies(keys, dtype=float)


def _centered(frame: pd.DataFrame, expression: str) -> np.ndarray:
    value = np.ones(len(frame))
    for column in expression.split("*"):
        value = value * (frame[column] - frame[column].mean()).to_numpy()
    return value


def candidate_design(frame: pd.DataFrame, spec: dict) -> np.ndarray:
    """Columns that let a candidate explain residuals (intercept included)."""
    parts = [np.ones((len(frame), 1))]
    if "cells" in spec:
        cells = _cell_indicators(frame, spec["cells"]).to_numpy()
        parts.append(cells)
        for slope in spec.get("slopes", []):
            parts.append(cells * _centered(frame, slope)[:, None])
    for expression in spec.get("columns", []):
        parts.append(_centered(frame, expression)[:, None])
    return np.hstack(parts)


def project_out(residual: np.ndarray, frame: pd.DataFrame, spec: dict) -> np.ndarray:
    """Remove what one candidate can explain (used to measure overlap)."""
    design = candidate_design(frame, spec)
    return residual - design @ np.linalg.lstsq(design, residual, rcond=None)[0]


def remove_group_means(residual: np.ndarray, frame: pd.DataFrame, groupings) -> np.ndarray:
    """Subtract the best-fitting sum of group means (least squares on the
    groups' cell indicators). For a single grouping this is just its cell means."""
    design = np.hstack([_cell_indicators(frame, g).to_numpy() for g in groupings])
    fitted = design @ np.linalg.lstsq(design, residual, rcond=None)[0]
    return residual - fitted


def rerank(residual: np.ndarray, frame: pd.DataFrame, candidates=RERANK_CANDIDATES) -> pd.DataFrame:
    """Share of the (remaining) residual variance each candidate could explain.

    share = R^2 of the residual regressed on the candidate's columns; for pure
    cell candidates this equals the between-group share of residual_diagnosis.
    adjusted share uses p = rank of the candidate's columns.
    """
    n = len(residual)
    total = np.sum((residual - residual.mean()) ** 2)
    rows = []
    for name, spec in candidates.items():
        design = candidate_design(frame, spec)
        fitted = design @ np.linalg.lstsq(design, residual, rcond=None)[0]
        share = 1 - np.sum((residual - fitted) ** 2) / total
        p = np.linalg.matrix_rank(design)
        row = {"candidate": name, "columns": int(p), "share": float(share),
               "adjusted_share": float(1 - (1 - share) * (n - 1) / (n - p))}
        if "cells" in spec:
            sizes = frame.groupby(spec["cells"]).size()
            row.update({"min_cell_rows": int(sizes.min()),
                        f"cells_under_{MIN_CELL_ROWS}": int((sizes < MIN_CELL_ROWS).sum()),
                        "rows_in_sparse_cells": int(sizes[sizes < MIN_CELL_ROWS].sum())})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("adjusted_share", ascending=False)
    return table.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 2 feature expansion
# ---------------------------------------------------------------------------

# Interaction groups chosen from the residual diagnosis, in the order they are
# added, with their degree (interaction order). Degree-3 groups also bring the
# lower-order terms they need (e.g. hour x temp for hour x workingday x temp),
# so the model stays hierarchical.
PHASE2_GROUPS = {
    "hour x workingday": 2,
    "hour x yr": 2,
    "hour x season": 2,
    "hour-block x weathersit": 2,
    "temp x hum": 2,
    "hour x workingday x yr": 3,
    "hour x workingday x temp": 3,
}

# Hour (or hour-block) indicators select "which part of the day" and are used
# as 0/1. Every other factor in a product is centered at its training mean
# first. That leaves the set of models the columns can express unchanged (the
# hour indicator itself is already in the model) but makes each product far
# less correlated with its hour indicator, so gradient descent converges faster.
SELECTORS = ("hr", "hour_block")

# Original data column behind each prepared column (for lineage).
SOURCE_COLUMN = {"hour_block": "hr"}


def group_terms(group: str, levels: dict) -> list[tuple]:
    """Explicit term list of one group.

    A term is a tuple of factors; a factor is (column, level) for an indicator
    or (column, None) for a numeric value. Hour, season, hour-block and
    weathersit levels exclude their reference level, so no term duplicates a
    Phase 1 column or another term.
    """
    hours, seasons = levels["hr"], levels["season"]
    if group == "hour x workingday":
        return [(("hr", h), ("workingday", None)) for h in hours]
    if group == "hour x yr":
        return [(("hr", h), ("yr", None)) for h in hours]
    if group == "hour x season":
        return [(("hr", h), ("season", s)) for h in hours for s in seasons]
    if group == "hour-block x weathersit":
        return [(("hour_block", b), ("weathersit", w))
                for b in levels["hour_block"] for w in levels["weathersit"]]
    if group == "temp x hum":
        return [(("temp", None), ("hum", None))]
    if group == "hour x workingday x yr":
        return ([(("workingday", None), ("yr", None))]
                + [(("hr", h), ("workingday", None), ("yr", None)) for h in hours])
    if group == "hour x workingday x temp":
        return ([(("hr", h), ("temp", None)) for h in hours]
                + [(("workingday", None), ("temp", None))]
                + [(("hr", h), ("workingday", None), ("temp", None)) for h in hours])
    raise ValueError(f"unknown group: {group}")


def term_name(term: tuple) -> str:
    return "*".join(column if level is None else f"{column}={level}"
                    for column, level in term)


def term_lineage(term: tuple) -> list[str]:
    return sorted({SOURCE_COLUMN.get(column, column) for column, _ in term})


def phase1_lineage(feature_name: str) -> list[str]:
    """Raw column behind a Phase 1 feature name such as 'hr=8' or 'temp'."""
    return [] if feature_name == "intercept" else [feature_name.split("=")[0]]


class Phase2Features:
    """Phase 1 design matrix plus standardized interaction columns.

    Phase 1 columns come from the Phase 1 encoder/scaler, unchanged. Each new
    column is the product of its term's factors (non-hour factors centered at
    their training mean), then standardized with the product's training mean
    and standard deviation.
    """

    def __init__(self, phase1, groups: list[str], center_factors: bool = True):
        self.phase1 = phase1
        self.groups = list(groups)
        self.center_factors = center_factors  # False only to show why we center
        self.levels = None
        self.centers = None
        self.mean = None
        self.std = None

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        out = group_rare_weather(df)
        out["hour_block"] = hour_block(out["hr"])
        return out

    @property
    def terms(self) -> list[tuple]:
        return [term for g in self.groups for term in group_terms(g, self.levels)]

    def _factor(self, prepared, factor, centered=True) -> np.ndarray:
        column, level = factor
        value = prepared[column].to_numpy(dtype=float)
        if level is not None:
            value = (value == level).astype(float)
        if centered and column not in SELECTORS:
            value = value - self.centers[term_name((factor,))]
        return value

    def _products(self, prepared, centered=True) -> np.ndarray:
        columns = []
        for term in self.terms:
            value = np.ones(len(prepared))
            for factor in term:
                value = value * self._factor(prepared, factor, centered)
            columns.append(value)
        if not columns:
            return np.empty((len(prepared), 0))
        return np.column_stack(columns)

    def fit(self, train_df: pd.DataFrame) -> "Phase2Features":
        prepared = self._prepare(train_df)
        self.levels = {
            "hr": list(self.phase1.levels["hr"]),
            "weathersit": list(self.phase1.levels["weathersit"]),
            "season": sorted(int(v) for v in prepared["season"].unique())[1:],
            "hour_block": sorted(int(v) for v in prepared["hour_block"].unique())[1:],
        }
        for term in self.terms:
            # No product of two levels of one column (always 0) and no square
            # of a 0/1 column (identical to the column).
            columns = [column for column, _ in term]
            assert len(set(columns)) == len(columns), f"repeated factor: {term_name(term)}"
        self.centers = {}
        for term in self.terms:
            for factor in term:
                if factor[0] not in SELECTORS:
                    center = self._factor(prepared, factor, centered=False).mean()
                    self.centers[term_name((factor,))] = (
                        float(center) if self.center_factors else 0.0)
        raw = self._products(prepared)
        self.mean = raw.mean(axis=0)
        self.std = raw.std(axis=0)
        if np.any(self.std == 0):
            raise ValueError("An interaction column is constant in the training portion.")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        phase1_part = self.phase1.transform(df)
        new_part = (self._products(self._prepare(df)) - self.mean) / self.std
        return np.hstack([phase1_part, new_part])

    @property
    def feature_names(self) -> list[str]:
        return self.phase1.feature_names + [term_name(t) for t in self.terms]

    @property
    def lineage(self) -> dict[str, list[str]]:
        out = {name: phase1_lineage(name) for name in self.phase1.feature_names}
        out.update({term_name(t): term_lineage(t) for t in self.terms})
        return out

    def term_support(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Training rows where each new term is non-zero before centering
        (e.g. hr=8*workingday: working-day rows at 08:00)."""
        raw = self._products(self._prepare(train_df), centered=False)
        groups = [g for g in self.groups for _ in group_terms(g, self.levels)]
        return pd.DataFrame({
            "group": groups,
            "term": [term_name(t) for t in self.terms],
            "support_rows": np.count_nonzero(raw, axis=0),
        })

    def warm_start(self, phase1_weights) -> np.ndarray:
        """Phase 1 weights in their own columns, zero for every new column."""
        return np.concatenate([np.asarray(phase1_weights), np.zeros(len(self.terms))])

    def to_dict(self) -> dict:
        return {
            "phase1": self.phase1.to_dict(),
            "groups": self.groups,
            "center_factors": self.center_factors,
            "levels": self.levels,
            "centers": self.centers,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_names": self.feature_names,
            "lineage": self.lineage,
        }

    @classmethod
    def from_dict(cls, state: dict) -> "Phase2Features":
        features = cls(Phase1Features.from_dict(state["phase1"]), state["groups"],
                       state["center_factors"])
        features.levels = {k: list(v) for k, v in state["levels"].items()}
        features.centers = dict(state["centers"])
        features.mean = np.asarray(state["mean"])
        features.std = np.asarray(state["std"])
        return features


def fit_and_score(features: Phase2Features, train_df, y_train, val_df, y_val,
                  phase1_weights, step_multiplier, max_iter, tol, decile_edges):
    """Fit one expanded design with the Phase 1 GD and score it.

    Returns the GD result and one row of numbers: design checks, convergence,
    train/validation scores and the operator checks.
    """
    X_tr, X_va = features.transform(train_df), features.transform(val_df)
    eigenvalues = np.linalg.eigvalsh(X_tr.T @ X_tr / len(X_tr))

    # Warm start: before any update the expanded model must predict exactly
    # what Phase 1 predicts.
    init = features.warm_start(phase1_weights)
    phase1_pred = features.phase1.transform(train_df) @ np.asarray(phase1_weights)
    init_gap = float(np.max(np.abs(X_tr @ init - phase1_pred)))
    assert init_gap < 1e-8, f"warm start changes predictions by {init_gap}"

    model = LinearRegressionGD(learning_rate=step_multiplier / eigenvalues[-1],
                               max_iter=max_iter, tol=tol)
    result = model.fit(X_tr, y_train, init_weights=init)
    pred_tr, pred_va = X_tr @ result.weights, X_va @ result.weights

    # Independent check (not used for training): the exact least-squares fit.
    exact = np.linalg.lstsq(X_tr, y_train, rcond=None)[0]
    by_predicted = errors_by_demand_decile(y_val, pred_va, decile_edges, bin_on="predicted")
    top_predicted = by_predicted[by_predicted["decile"] == 10]
    top_predicted = (top_predicted.iloc[0] if len(top_predicted)
                     else pd.Series({"mean_shortfall_bikes": np.nan, "mean_error": np.nan}))
    train_scores = regression_metrics(y_train, pred_tr)
    val_scores = regression_metrics(y_val, pred_va)
    row = {
        "n_features": X_tr.shape[1] - 1,
        "rank_ok": bool(np.linalg.matrix_rank(X_tr) == X_tr.shape[1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "learning_rate": result.learning_rate,
        "status": result.status,
        "iterations": result.iterations,
        "final_max_abs_gradient": result.grad_history[-1],
        "gap_to_exact_bikes": float(np.max(np.abs(pred_va - X_va @ exact))),
        "train_r2": train_scores["r2"],
        "val_r2": val_scores["r2"],
        "train_val_r2_gap": train_scores["r2"] - val_scores["r2"],
        "val_rmse": val_scores["rmse"],
        "top_decile_shortfall": top_decile_shortfall(y_val, pred_va, decile_edges),
        "top_predicted_decile_shortfall": float(top_predicted["mean_shortfall_bikes"]),
        "top_predicted_decile_mean_error": float(top_predicted["mean_error"]),
        "negative_predictions": int(np.sum(pred_va < 0)),
    }
    return result, row


def save_phase2_artifact(path, features: Phase2Features, result, degree: int,
                         row: dict, sequence: pd.DataFrame, settings: dict) -> None:
    """Write everything Phase 3 consumes: chosen degree, expanded features
    with lineage, the fitted encoders/scalers, weights and scores."""
    artifact = {
        "degree": degree,
        "groups": features.groups,
        "feature_names": features.feature_names,
        "lineage": features.lineage,
        "weights": result.weights.tolist(),
        "learning_rate": result.learning_rate,
        "iterations": result.iterations,
        "status": result.status,
        "final_max_abs_gradient": result.grad_history[-1],
        "metrics": {key: float(row[key]) for key in
                    ("train_r2", "val_r2", "val_rmse", "train_val_r2_gap",
                     "top_decile_shortfall", "top_predicted_decile_shortfall")},
        "sequence": sequence.to_dict(orient="records"),
        "settings": settings,
        "features": features.to_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2))


def load_phase2_artifact(path) -> dict:
    artifact = json.loads(Path(path).read_text())
    artifact["weights"] = np.asarray(artifact["weights"])
    return artifact
