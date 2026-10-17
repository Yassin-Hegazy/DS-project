"""Build report/report.pdf (LaTeX) from the executed notebook and the saved artifacts.

1. Every number is read from the executed notebook's outputs or from an artifact
   the notebook saved. Values recomputed here (the Phase 1 sweep and design table,
   a decile table, the Phase 5 cutoff scan and label table) are asserted to appear
   in the notebook's own printed output, so the report cannot drift from the run.
2. The numbers are written to report/numbers.tex as al{key} macros and the tables
   to report/tab_*.tex; report/report.tex contains no hand-typed results.
3. Figures are drawn in greyscale (line styles and markers, no colour) as PDF.
4. pdflatex runs twice; the build fails on an undefined al key, any overfull box,
   or more than 6 pages.

Run after executing notebooks/final_submission.ipynb:  python report/build_report.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.data import (  # noqa: E402
    chronological_split, compute_team_seed, load_data, seeded_split)
from src.features import Phase1Features  # noqa: E402
from src.operator_checks import demand_decile_edges, errors_by_demand_decile  # noqa: E402
from src.phase1_gradient_descent import (  # noqa: E402
    LinearRegressionGD, largest_curvature, load_phase1_artifact)
from src.phase2_polynomial import Phase2Features, load_phase2_artifact  # noqa: E402
from src.phase3_bias_variance import load_phase3_artifact  # noqa: E402
from src.phase4_regularization import load_phase4_artifact  # noqa: E402
from src.phase5_logistic import (  # noqa: E402
    high_demand_cutoffs, high_demand_labels, label_definition_comparison, sigmoid,
    threshold_scan)

OUT = ROOT / "report"
FIGURES = OUT / "figures"
NOTEBOOK = ROOT / "notebooks" / "final_submission.ipynb"

# ---------------------------------------------------------------------------
# Reading the executed notebook
# ---------------------------------------------------------------------------

def notebook_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf8"))
    cells = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        texts = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                raise RuntimeError("the notebook contains an error output")
            if "text" in out:
                texts.append("".join(out["text"]))
            data = out.get("data", {})
            if "text/plain" in data and "image/png" not in data:
                texts.append("".join(data["text/plain"]))
        cells.append(("".join(cell["source"]), "\n".join(texts)))
    if not any(text for _, text in cells):
        raise RuntimeError("run the notebook first: it has no outputs")
    return cells


CELLS = notebook_cells()


def output_of(marker: str) -> str:
    found = [text for source, text in CELLS if marker in source]
    assert len(found) == 1, f"{len(found)} cells contain {marker!r}"
    return found[0]


def grab(marker: str, pattern: str, cast=float, group=1):
    match = re.search(pattern, output_of(marker), re.M)
    assert match, f"{pattern!r} not found in the output of {marker!r}"
    return cast(match.group(group))


def grab_all(marker: str, pattern: str, cast=float):
    return [cast(x) for x in re.findall(pattern, output_of(marker), re.M)]


def appears(marker: str, text: str) -> None:
    assert text in output_of(marker), f"{text!r} not in the output of {marker!r}"


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

v = {}
seed = compute_team_seed(config.TEAM_IDS)
train_all, test_X = load_data(config.TRAIN_PATH, config.TEST_PATH)
train_df, val_df = seeded_split(train_all, seed)
y_train, y_val = train_df["cnt"].to_numpy(float), val_df["cnt"].to_numpy(float)
v.update(seed=seed, n_all=len(train_all), n_test=len(test_X), n_train=len(train_df),
         n_val=len(val_df), n_days=train_all["dteday"].nunique())
appears("Team seed", f"Team seed: {seed}")

p1 = load_phase1_artifact(config.PHASE1_ARTIFACT)
p2 = load_phase2_artifact(config.PHASE2_ARTIFACT)
p3 = load_phase3_artifact(config.PHASE3_ARTIFACT)
p4 = load_phase4_artifact(config.PHASE4_ARTIFACT)
p5 = json.loads(config.PHASE5_ARTIFACT.read_text())

# Phase 1 -------------------------------------------------------------------
lam = p1["settings"]["largest_curvature"]
v.update(p1_lam=lam, p1_limit=2 / lam, p1_lr=p1["learning_rate"], p1_iter=p1["iterations"],
         p1_train_r2=p1["metrics"]["train"]["r2"], p1_val_r2=p1["metrics"]["validation"]["r2"],
         p1_val_rmse=p1["metrics"]["validation"]["rmse"])
SWEEP = "MULTIPLIERS = [0.005"
sweep = {}
phase1 = Phase1Features.from_dict(p1["features"])
X1_train, X1_val = phase1.transform(train_df), phase1.transform(val_df)
for multiplier in (0.005, 0.1, 0.5, 1.0, 1.9, 2.1):
    result = LinearRegressionGD(multiplier / lam, p1["settings"]["max_iter"],
                                p1["settings"]["tol"]).fit(X1_train, y_train)
    sweep[multiplier] = result
    appears(SWEEP, f"{multiplier:g}/curvature")
    assert re.search(rf"{multiplier:g}/curvature\s+[\d.]+\s+{result.status}\s+{result.iterations}\b",
                     output_of(SWEEP)), multiplier
v.update(p1_div_iter=sweep[2.1].iterations, p1_slow_grad=sweep[0.005].grad_history[-1])
v["p1_gap_exact"] = grab("chosen_step = min", r"exact minimum: ([\d.]+) bikes")
DESIGN = "design_rows = []"
design = {}
for seasonal in ("season", "mnth"):
    for trend in (False, True):
        f = Phase1Features(seasonal, trend).fit(train_df)
        Xt, Xv = f.transform(train_df), f.transform(val_df)
        ev = np.linalg.eigvalsh(Xt.T @ Xt / len(Xt))
        r = LinearRegressionGD(1.0 / ev[-1], 50_000, 1e-4).fit(Xt, y_train)
        val_r2 = 1 - np.sum((Xv @ r.weights - y_val) ** 2) / np.sum((y_val - y_val.mean()) ** 2)
        design[(seasonal, trend)] = (val_r2, ev[-1] / ev[0], r.iterations)
        appears(DESIGN, f"{val_r2:.6f}")
v.update(p1_season_r2=design[("season", False)][0], p1_month_r2=design[("mnth", False)][0],
         p1_trend_gain=design[("mnth", True)][0] - design[("mnth", False)][0],
         p1_cond=design[("mnth", False)][1], p1_cond_trend=design[("mnth", True)][1],
         p1_iter_trend=design[("mnth", True)][2], p1_iter_notrend=design[("mnth", False)][2])
OPS1 = "decile_edges = demand_decile_edges(y_train)"
v["top_cut"] = grab(OPS1, r"actual > (\d+) bikes", int)
v["p1_top_short"] = grab(OPS1, r"bikes\): ([\d.]+) bikes per hour")
v["p1_negative"] = grab(OPS1, r"predicted below zero: (\d+)", int)
edges = demand_decile_edges(y_train)
by_pred = errors_by_demand_decile(y_val, X1_val @ p1["weights"], edges, bin_on="predicted")
v["p1_toppred_err"] = float(by_pred.iloc[-1]["mean_error"])
v["p1_toppred_short"] = float(by_pred.iloc[-1]["mean_shortfall_bikes"])
appears(OPS1, f"{v['p1_toppred_err']:.2f}")
v["resid_between"] = grab("train_residual = Xb_train @ w_phase1",
                          r"residual variance: ([\d.]+)% between")

# Bonus ---------------------------------------------------------------------
BONUS_LR = 'for label, step in (("Phase 1 step"'
v["b_div_iter"] = grab(BONUS_LR, r"^\d\s+Phase 1 step\s+[\d.]+\s+diverged\s+(\d+)", int)
v["b_lr"] = grab(BONUS_LR, r"Phase 1 step / k\s+([\d.]+)\s+converged")
v["b_iter"] = grab(BONUS_LR, r"Phase 1 step / k\s+[\d.]+\s+converged\s+(\d+)", int)
comparison = pd.read_csv(config.BONUS_DIR / "validation_comparison.csv", index_col=0)
v["bonus"] = comparison
v["b_shift"] = grab("pred_asym = Xb_val @ asym_result.weights", r"mean ([\d.]+), median")
TOPB = "Operator check for the bonus"
v["b_top_mse"] = grab(TOPB, r"MSE model: top-decile shortfall ([\d.]+)")
v["b_top_asym"] = grab(TOPB, r"Asymmetric model: top-decile shortfall ([\d.]+)")
FOLLOW = "Phase 1 features vs Phase 2 features: where"
r2s = grab_all(FOLLOW, r"^r2\s+([\d.]+)\s+[\d.]+")
r2s_asym = grab_all(FOLLOW, r"^r2\s+[\d.]+\s+([\d.]+)")
means = grab_all(FOLLOW, r"^mean_prediction\s+([\d.]+)\s+[\d.]+")
means_asym = grab_all(FOLLOW, r"^mean_prediction\s+[\d.]+\s+([\d.]+)")
tops = grab_all(FOLLOW, r"^top_decile_shortfall\s+([\d.]+)\s+[\d.]+")
tops_asym = grab_all(FOLLOW, r"^top_decile_shortfall\s+[\d.]+\s+([\d.]+)")
v.update(f_r2=r2s[1], f_r2_asym=r2s_asym[1], f_shift=means_asym[1] - means[1],
         f_top=tops[1], f_top_asym=tops_asym[1])

# Phase 2 -------------------------------------------------------------------
seq = pd.DataFrame(p2["sequence"]).set_index("added group")
v["p2"] = seq
v["p2_cond_unc"] = grab("uncentered = Phase2Features", r"with centering, (\d+) without", int)
v["p2_cond"] = grab("uncentered = Phase2Features", r"condition number: (\d+) with centering", int)

# Phase 3 -------------------------------------------------------------------
seeded = pd.DataFrame(p3["tables"]["seeded_sweep"]).set_index("added group")
chrono = pd.DataFrame(p3["tables"]["chronological_sweep"]).set_index("added group")
curves = pd.read_csv(config.PHASE3_DIR / "learning_curves.csv")
v.update(p3_seeded=seeded, p3_chrono=chrono, p3_curves=curves,
         p3_target=p3["target_step"], p3_target_n=len(p3["feature_names"]) - 1,
         p3_first_day=p3["settings"]["first_chronological_validation_day"])

# Phase 4 -------------------------------------------------------------------
v["p4_sel"] = p4["selected"]
v["p4_verdicts"] = pd.DataFrame(p4["column_verdicts"]).set_index("column")
v["p4_survivors"] = len(p4["lasso_surviving_features"])
v["p4_columns"] = p4["lasso_surviving_original_columns"]
paths = {label: pd.read_csv(config.PHASE4_DIR / f"path_{label.replace(' ', '_').lower()}.csv")
         for label in ("Ridge", "Lasso", "Elastic Net 0.9", "Elastic Net 0.5", "Elastic Net 0.1")}

# Phase 5 -------------------------------------------------------------------
LABEL = "label_definition_comparison(train_df)"
v["p5_pos"] = grab(LABEL, r"training positives: (\d+) of", int)
v["p5_pos_rate"] = grab(LABEL, r"training positives: \d+ of \d+ \(([\d.]+)%\)")
FIT5 = "phase5_columns = [0]"
v["p5_iter"] = grab(FIT5, r"converged after (\d+) updates", int)
v["p5_curv"] = grab(FIT5, r"at most L = ([\d.]+)")
v["p5_lr"] = grab(FIT5, r"learning rate 1.9/L = ([\d.]+)")
v["p5_val_pos"] = grab(FIT5, r"validation ([\d.]+)% high")
val5 = p5["validation"]
tuned_key = [k for k in val5 if k.startswith("tuned")][0]
v.update(p5_tuned=val5[tuned_key], p5_half=val5["cutoff 0.50"], p5_cut=p5["threshold"])
DIAG = "how much of the ROC-AUC does the calendar"
v["p5_auc_cal"] = grab(DIAG, r"calendar only \(month x weekday\) ([\d.]+)")
v["p5_auc_month"] = grab(DIAG, r"within a month, averaged over months: ([\d.]+)")
v["p5_auc_season_lo"] = grab(DIAG, r"within a season: ([\d.]+) to")
v["p5_auc_season_hi"] = grab(DIAG, r"within a season: [\d.]+ to ([\d.]+)")
v["p5_jan"] = grab(DIAG, r"\{1: ([\d.]+),")
v["p5_jun"] = grab(DIAG, r"6: ([\d.]+),")
odds = pd.DataFrame({"feature": p5["features"][1:], "beta": np.asarray(p5["weights"][1:])})
odds["odds"] = np.exp(odds["beta"])
odds = odds.reindex(odds["beta"].abs().sort_values(ascending=False).index)
v["p5_odds"] = odds

# Submission ----------------------------------------------------------------
SUB = "final_lambda = ridge"
v["s_iter"] = grab(SUB, r"converged after (\d+) updates", int)
v["s_r2"] = grab(SUB, r"training R2 ([\d.]+)")
v["s_rmse"] = grab(SUB, r"RMSE ([\d.]+) on all labeled rows")
v["s_clipped"] = grab(SUB, r"(\d+) below zero clipped", int)
v["s_mean"] = grab(SUB, r"mean ([\d.]+), max")
v["s_max"] = grab(SUB, r"max ([\d.]+)\n")
v["s_dropped"] = grab(SUB, r"\((\d+) exact combinations dropped\)", int)
submission = pd.read_csv(config.SUBMISSION_PATH)
assert len(submission) == 574 and np.isfinite(submission["cnt"]).all() and (submission["cnt"] >= 0).all()


# Phase 2 Step A and the chronological split (sizes only)
step_a = pd.read_csv(config.PHASE2_DIR / "residual_diagnosis.csv").set_index("combination")
_, chrono_val, _first_day = chronological_split(train_all)
assert _first_day == v["p3_first_day"]

# Phase 5 label evidence (recomputed with the pipeline's function, checked against the notebook)
label_table = label_definition_comparison(train_df).set_index("definition")
for value in label_table["positive_rate"]:
    appears(LABEL, f"{value:.3f}")

# ---------------------------------------------------------------------------
# Macros: every number the text uses, formatted once
# ---------------------------------------------------------------------------

def num(x, d=4):
    """Fixed decimals with thousands separators; a minus sign typeset as a real minus."""
    text = f"{x:,.{d}f}"
    return f"$-${text[1:]}" if text.startswith("-") else text


def count(x):
    return f"{int(x):,}"


S, C = seeded, chrono
deg3, deg4, target = "hour x workingday x temp", "hour x workingday x yr x season", v["p3_target"]
ridge, lasso, enet = v["p4_sel"]["Ridge"], v["p4_sel"]["Lasso"], v["p4_sel"]["Elastic Net"]
cmp_ = v["bonus"]
mse_col, asym_col = cmp_.columns[0], cmp_.columns[1]
tuned, half = v["p5_tuned"], v["p5_half"]
lc = curves.set_index(["design", "fraction"])
big = "largest extended design"
temp_odds = float(odds.loc[odds["feature"] == "temp", "odds"].iloc[0])
lasso_path = paths["Lasso"]

M = {
    # data and seed
    "seed": str(v["seed"]), "nAll": count(v["n_all"]), "nTest": count(v["n_test"]),
    "nTrain": count(v["n_train"]), "nVal": count(v["n_val"]), "nDays": count(v["n_days"]),
    # Phase 1
    "pOneLam": num(v["p1_lam"]), "pOneLimit": num(v["p1_limit"]), "pOneLr": num(v["p1_lr"], 3),
    "pOneIter": count(v["p1_iter"]), "pOneTrainR": num(v["p1_train_r2"]), "pOneValR": num(v["p1_val_r2"]),
    "pOneValRmse": num(v["p1_val_rmse"], 1), "pOneDivIter": count(v["p1_div_iter"]),
    "pOneSlowGrad": num(v["p1_slow_grad"]), "pOneGapExact": num(v["p1_gap_exact"]),
    "pOneSeasonR": num(v["p1_season_r2"]), "pOneMonthR": num(v["p1_month_r2"]),
    "pOneTrendGain": num(v["p1_trend_gain"]), "pOneCond": num(v["p1_cond"], 0),
    "pOneCondTrend": num(v["p1_cond_trend"], 0), "pOneIterTrend": count(v["p1_iter_trend"]),
    "pOneIterNoTrend": count(v["p1_iter_notrend"]), "topCut": count(v["top_cut"]),
    "pOneTopShort": num(v["p1_top_short"], 1), "pOneTopPredErr": num(-v["p1_toppred_err"], 1),
    "pOneNegative": count(v["p1_negative"]), "residBetween": num(v["resid_between"], 1),
    # bonus
    "bDivIter": count(v["b_div_iter"]), "bLr": num(v["b_lr"], 3), "bIter": count(v["b_iter"]),
    "bShift": num(v["b_shift"], 1),
    "bRmse": num(cmp_.loc["r2", mse_col], 3), "bRasym": num(cmp_.loc["r2", asym_col], 3),
    "bRmseMse": num(cmp_.loc["rmse", mse_col], 1), "bRmseAsym": num(cmp_.loc["rmse", asym_col], 1),
    "bCostMse": num(cmp_.loc["asymmetric_cost", mse_col], 0),
    "bCostAsym": num(cmp_.loc["asymmetric_cost", asym_col], 0),
    "bCostDrop": num(100 * (1 - cmp_.loc["asymmetric_cost", asym_col] / cmp_.loc["asymmetric_cost", mse_col]), 1),
    "bShortMse": num(cmp_.loc["mean_shortfall_bikes", mse_col], 1),
    "bShortAsym": num(cmp_.loc["mean_shortfall_bikes", asym_col], 1),
    "bIdleMse": num(cmp_.loc["mean_surplus_bikes", mse_col], 1),
    "bIdleAsym": num(cmp_.loc["mean_surplus_bikes", asym_col], 1),
    "bTopMse": num(v["b_top_mse"], 1), "bTopAsym": num(v["b_top_asym"], 1),
    "fShift": num(v["f_shift"], 1), "fR": num(v["f_r2"], 3), "fRasym": num(v["f_r2_asym"], 3),
    "fRcost": num(v["f_r2"] - v["f_r2_asym"], 3), "fTop": num(v["f_top"], 1), "fTopAsym": num(v["f_top_asym"], 1),
    # Phase 2
    "pTwoCondUnc": count(v["p2_cond_unc"]), "pTwoCond": count(v["p2_cond"]),
    "pTwoHrWdR": num(seq.loc["hour x workingday", "val_r2"]),
    "pTwoHrWdGain": num(seq.loc["hour x workingday", "val_r2"] - seq.loc["(Phase 1)", "val_r2"], 3),
    "pTwoDegTwoR": num(seq.loc["temp x hum", "val_r2"]), "pTwoDegThreeR": num(seq.loc[deg3, "val_r2"]),
    "pTwoDegThreeRmse": num(seq.loc[deg3, "val_rmse"], 1),
    "pTwoDegThreeTrainRmse": num(seq.loc[deg3, "train_rmse"], 2),
    "pTwoDegThreeValRmse": num(seq.loc[deg3, "val_rmse"], 2), "pTwoDegThreeN": count(seq.loc[deg3, "n_features"]),
    "pTwoDegThreeIter": count(seq.loc[deg3, "iterations"]), "pTwoDegThreeGap": num(seq.loc[deg3, "train_val_r2_gap"]),
    "pTwoTopPredStart": num(seq.loc["(Phase 1)", "top_predicted_decile_shortfall"], 1),
    "pTwoTopPredEnd": num(seq.loc[deg3, "top_predicted_decile_shortfall"], 1),
    "pTwoTempHumGain": num(seq.loc["temp x hum", "val_r2"] - seq.loc["hour-block x weathersit", "val_r2"]),
    # Phase 3
    "pThreeTargetN": count(len(p3["feature_names"]) - 1), "pThreeTargetR": num(S.loc[target, "val_r2"]),
    "pThreeTargetChronoR": num(C.loc[target, "val_r2"]), "pThreeDegFourR": num(S.loc[deg4, "val_r2"]),
    "pThreeDegFourN": count(S.loc[deg4, "n_features"]), "pThreeDegFourGap": num(S.loc[deg4, "train_val_r2_gap"]),
    "pThreeDegThreeChronoR": num(C.loc[deg3, "val_r2"]), "pThreeDegThreeChronoRmse": num(C.loc[deg3, "val_rmse"], 1),
    "pThreeWeatherChronoR": num(C.loc["hour x weathersit", "val_r2"]),
    "pThreeChronoGain": num(C.loc["hour x weathersit", "val_r2"] - C.loc[deg3, "val_r2"]),
    "pThreeSeededGain": num(S.loc["hour x weathersit", "val_r2"] - S.loc[deg3, "val_r2"]),
    "pThreeDegFourChronoR": num(C.loc[deg4, "val_r2"]),
    "pThreeDegFourChronoDrop": num(C.loc[deg4, "val_r2"] - C.loc["hour x weathersit", "val_r2"]),
    "pThreeDegFourChronoShort": num(C.loc[deg4, "top_decile_shortfall"]
                                    - C.loc["hour x weathersit", "top_decile_shortfall"], 1),
    "pThreeDegFourChronoN": count(C.loc[deg4, "n_features"]),
    "pThreeMaxIter": count(S["iterations"].max()),
    "pThreeLcGapSmall": num(lc.loc[(big, 0.25), "val_rmse"] - lc.loc[(big, 0.25), "train_rmse"], 1),
    "pThreeLcGapFull": num(lc.loc[(big, 1.0), "val_rmse"] - lc.loc[(big, 1.0), "train_rmse"], 1),
    "pThreeFirstDay": v["p3_first_day"],
    "pThreeTargetChronoErr": num(C.loc[target, "top_predicted_decile_mean_error"], 1),
    "pThreeTargetSeededErr": num(S.loc[target, "top_predicted_decile_mean_error"], 1),
    "pThreeDegThreeChronoShort": num(C.loc[deg3, "top_decile_shortfall"], 1),
    "pThreeDegThreeSeededShort": num(S.loc[deg3, "top_decile_shortfall"], 1),
    # Phase 4
    "ridgeLam": f"{ridge['lambda']:.2g}", "ridgeR": num(ridge["val_r2"], 6),
    "unpenR": num(S.loc[target, "val_r2"], 6), "lassoLam": num(lasso["lambda"], 3),
    "lassoN": count(lasso["nonzero"]), "lassoR": num(lasso["val_r2"]),
    "enetRho": f"{enet['l1_ratio']:g}", "enetLam": num(enet["lambda"], 4), "enetR": num(enet["val_r2"]),
    "enetN": count(enet["nonzero"]), "unpenShort": num(S.loc[target, "top_decile_shortfall"], 1),
    "lassoShort": num(lasso["top_decile_shortfall"], 1), "enetShort": num(enet["top_decile_shortfall"], 1),
    "ridgeShort": num(ridge["top_decile_shortfall"], 1), "lassoLamMax": num(lasso_path["lambda"].iloc[0], 1),
    "survivors": count(v["p4_survivors"]), "survivorColumns": count(len(v["p4_columns"])),
    # Phase 5
    "pFivePos": count(v["p5_pos"]), "pFivePosRate": num(v["p5_pos_rate"], 1),
    "pFiveValPos": num(v["p5_val_pos"], 1), "pFiveCurv": num(v["p5_curv"]), "pFiveLr": num(v["p5_lr"], 3),
    "pFiveIter": count(v["p5_iter"]), "pFiveCut": num(v["p5_cut"], 2),
    "pFiveAuc": num(tuned["roc_auc"], 3), "pFiveAcc": num(tuned["accuracy"], 3),
    "pFiveBase": num(tuned["always_0_accuracy"], 3), "pFivePrec": num(tuned["precision"], 3),
    "pFiveRec": num(tuned["recall"], 3), "pFiveFone": num(tuned["f1"], 3),
    "pFiveHalfAcc": num(half["accuracy"], 3), "pFiveHalfFone": num(half["f1"], 3),
    "pFiveHalfRec": num(half["recall"], 3), "pFiveMissed": count(tuned["tp"] - half["tp"]),
    "pFiveSavedAlarms": count(tuned["fp"] - half["fp"]),
    "pFiveAucCal": num(v["p5_auc_cal"], 3), "pFiveAucMonth": num(v["p5_auc_month"], 3),
    "pFiveAucSeasonLo": num(v["p5_auc_season_lo"], 3), "pFiveAucSeasonHi": num(v["p5_auc_season_hi"], 3),
    "pFiveJan": num(v["p5_jan"], 1), "pFiveJun": num(v["p5_jun"], 1), "pFiveTempOdds": num(temp_odds, 2),
    "labelGlobalNever": count(label_table.loc["global cutoff", "hours_never_positive"]),
    "labelNoYearOld": num(100 * label_table.loc["hour x working day", "rate_2011"], 1),
    "labelNoYearNew": num(100 * label_table.loc["hour x working day", "rate_2012"], 1),
    "labelOld": num(100 * label_table.loc["hour x working day x year (chosen)", "rate_2011"], 1),
    "labelNew": num(100 * label_table.loc["hour x working day x year (chosen)", "rate_2012"], 1),
    # submission
    "sIter": count(v["s_iter"]), "sR": num(v["s_r2"]), "sRmse": num(v["s_rmse"], 2),
    "sClipped": count(v["s_clipped"]), "sMean": num(v["s_mean"], 1), "sMax": num(v["s_max"], 1),
    "sDropped": count(v["s_dropped"]),
    # design constants and rules, read from the artifacts
    "slowMult": f"{min(sweep):g}", "pOneN": count(len(p1["feature_names"]) - 1), "labelGroups": count(len(p5["label"]["cutoffs"])),
    "rTol": f"{p3['diagnosis']['r2_tolerance']:g}", "aloneTol": f"{p4['rules']['alone_r2_threshold']:g}",
    "selTol": f"{p4['rules']['select_r2_tolerance']:g}",
    "stepAshare": num(step_a.loc["hour x workingday", "between_share"], 3),
    "pThreeChronoDays": count(chrono_val["dteday"].nunique()), "pThreeChronoRows": count(len(chrono_val)),
}


def write_numbers():
    lines = ["% Generated by report/build_report.py from the executed notebook and the artifacts.",
             "% Do not edit: rebuild instead."]
    for key, value in M.items():
        assert re.fullmatch(r"[A-Za-z]+", key), key
        lines.append(rf"\expandafter\def\csname val@{key}\endcsname{{{value}}}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf8")


def table(name, rows):
    (OUT / f"tab_{name}.tex").write_text(
        "% Generated by report/build_report.py. Do not edit.\n" + "\n".join(rows)
        # \bottomrule lives here: LaTeX cannot place it directly after an \input in a tabular.
        + "\n\\bottomrule\n", encoding="utf8")


def write_tables():
    pretty = {"(Phase 1)": "Phase 1 (main effects)", "hour x workingday": r"+ hour $\times$ workingday",
              "hour x yr": r"+ hour $\times$ yr", "hour x season": r"+ hour $\times$ season",
              "hour-block x weathersit": r"+ hour-block $\times$ weathersit",
              "temp x hum": r"+ temp $\times$ hum",
              "hour x workingday x yr": r"+ hour $\times$ workingday $\times$ yr",
              "hour x workingday x temp": r"+ hour $\times$ workingday $\times$ temp"}
    table("phase2", [rf"{pretty[g]} & {int(r.degree)} & {count(r.n_features)} & {num(r.train_r2)} & "
                     rf"{num(r.val_r2)} & {num(r.top_predicted_decile_shortfall, 1)} \\"
                     for g, r in seq.iterrows()])
    method_rows = [("Unpenalised (Phase 3)", "0", "--", S.loc[target, "n_features"], S.loc[target, "val_r2"],
                    S.loc[target, "val_rmse"], S.loc[target, "top_decile_shortfall"])]
    for label, row in (("Ridge", ridge), ("Lasso", lasso), ("Elastic Net", enet)):
        method_rows.append((label, f"{row['lambda']:.3g}", f"{row['l1_ratio']:g}", row["nonzero"],
                            row["val_r2"], row["val_rmse"], row["top_decile_shortfall"]))
    table("phase4", [rf"{m} & {lam} & {rho} & {count(n)} & {num(r2)} & {num(rmse, 1)} & {num(short, 1)} \\"
                     for m, lam, rho, n, r2, rmse, short in method_rows])
    rows = []
    for column, row in v["p4_verdicts"].iterrows():
        if row["terms_in_target"] > 0:
            test, change = "removal", num(row["ablation_val_r2_change"])
        elif row.get("add_back_columns_kept", 0) == 0:
            test, change = "add back", "exact copy"
        else:
            test, change = "add back", num(row["add_back_val_r2_change"])
        kept = (f"{int(row['terms_surviving_lasso'])}/{int(row['terms_in_target'])}"
                if row["terms_in_target"] > 0 else "--")
        rows.append(rf"\texttt{{{column}}} & {test} & {change} & {num(row['alone_val_r2'], 3)} & {kept} & "
                    rf"{row['verdict'].split(' (')[0]} \\")
    table("verdicts", rows)
    table("phase5", [rf"{num(r['threshold'], 2)} & {count(r['tp'])} & {count(r['fp'])} & {count(r['fn'])} & "
                     rf"{count(r['tn'])} & {num(r['accuracy'], 3)} & {num(r['precision'], 3)} & "
                     rf"{num(r['recall'], 3)} & {num(r['f1'], 3)} \\" for r in (tuned, half)])


# ---------------------------------------------------------------------------
# Figures (greyscale only: line styles and markers carry identity)
# ---------------------------------------------------------------------------

FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.1, "pdf.fonttype": 42, "axes.edgecolor": "black",
                     "text.color": "black", "axes.labelcolor": "black",
                     "xtick.color": "black", "ytick.color": "black"})
STYLES = [("black", "-", "o"), ("0.45", "--", "s"), ("black", ":", "^"), ("0.55", "-.", "D"),
          ("0.25", (0, (5, 1)), "v"), ("0.7", "-", "x")]


def save(fig, name):
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)


def fig_phase1():
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    for (color, ls, _), m in zip(STYLES, (0.005, 0.1, 0.5, 1.0, 1.9, 2.1)):
        r = sweep[m]
        ax.plot(range(1, len(r.grad_history) + 1), r.grad_history, color=color, ls=ls,
                label=rf"{m:g}/$\lambda_{{max}}$ ({r.status.replace('_', ' ')}, {r.iterations:,})")
    ax.axhline(p1["settings"]["tol"], color="black", lw=0.5)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e10)  # tolerance line visible; empty band above the curves for the legend
    ax.set_xlabel("update"), ax.set_ylabel("max |gradient|")
    ax.legend(fontsize=5.6, frameon=False, loc="upper right")
    save(fig, "fig_phase1")


def fig_phase3():
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    ax.plot(S["n_features"], S["train_r2"], color="black", ls="-", marker="o", ms=2.6, label="train (seeded)")
    ax.plot(S["n_features"], S["val_r2"], color="0.45", ls="--", marker="s", ms=2.6,
            label="validation (seeded)")
    ax.plot(S["n_features"], C["val_r2"], color="black", ls=":", marker="^", ms=2.6,
            label="validation (chronological)")
    ax.axvline(len(p3["feature_names"]) - 1, color="0.6", lw=0.6)
    ax.set_ylim(0.85, 0.945)
    ax.set_xlabel("number of features (seeded design)"), ax.set_ylabel(r"$R^2$")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    save(fig, "fig_phase3")


def fig_phase4():
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.0))
    for (color, ls, marker), (label, path) in zip(STYLES, paths.items()):
        for ax, column in zip(axes, ("val_r2", "nonzero")):
            ax.plot(path["lambda"], path[column], color=color, ls=ls, marker=marker, ms=2.2,
                    markevery=6, label=label)
    for ax, ylabel in zip(axes, (r"validation $R^2$", "nonzero weights")):
        ax.set_xscale("log"), ax.set_xlabel(r"$\lambda$"), ax.set_ylabel(ylabel)
    axes[0].set_ylim(0.85, 0.935)
    axes[1].legend(fontsize=6, frameon=False, loc="lower left")
    save(fig, "fig_phase4")


def fig_phase5():
    design = Phase2Features.from_dict(p4["features"])
    columns = [0] + [f["index"] for f in p4["lasso_surviving_features"]]
    prob = sigmoid(design.transform(val_df)[:, columns] @ np.asarray(p5["weights"]))
    label = high_demand_labels(val_df, high_demand_cutoffs(train_df))
    scan = threshold_scan(label, prob, np.round(np.arange(0.05, 0.951, 0.01), 2))
    best = scan.loc[scan["threshold"] == p5["threshold"]].iloc[0]
    assert abs(best["f1"] - tuned["f1"]) < 1e-12, "cutoff scan does not reproduce the notebook"
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    for (color, ls, _), column in zip(STYLES, ("f1", "precision", "recall")):
        ax.plot(scan["threshold"], scan[column], color=color, ls=ls, label=column)
    ax.axvline(p5["threshold"], color="0.6", lw=0.6)
    ax.set_xlabel("probability cutoff"), ax.set_ylabel("validation score")
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    save(fig, "fig_phase5")


# ---------------------------------------------------------------------------
# Build and checks
# ---------------------------------------------------------------------------

def check_tex_keys():
    tex = (OUT / "report.tex").read_text(encoding="utf8")
    used = set(re.findall(r"\\val\{([A-Za-z]+)\}", tex))
    missing = used - set(M)
    assert not missing, f"report.tex uses undefined values: {sorted(missing)}"
    body = re.sub(r"%.*", "", tex)
    body = re.sub(r"\d*\.?\d+\s*(\\textwidth|\\linewidth|cm|pt)", "", body)  # layout lengths
    hand_typed = re.findall(r"(?<![\w.])\d+\.\d{3,}", body)
    assert not hand_typed, f"report.tex contains hand-typed results {hand_typed}; use \\val{{...}}"
    return used


def compile_pdf():
    tex_bin = Path.home() / "AppData" / "Roaming" / "TinyTeX" / "bin" / "windows" / "pdflatex.exe"
    pdflatex = str(tex_bin) if tex_bin.exists() else "pdflatex"
    for _ in range(2):
        run = subprocess.run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                             cwd=OUT, capture_output=True, text=True, encoding="latin-1")
        if run.returncode != 0:
            print(run.stdout[-3000:])
            raise RuntimeError("pdflatex failed")
    log = (OUT / "report.log").read_text(encoding="latin-1")
    overfull = re.findall(r"Overfull \\[hv]box.*", log)
    assert not overfull, "overfull boxes:\n" + "\n".join(overfull)
    assert "There were undefined references" not in log, "undefined references"
    pages = int(re.search(r"Output written on report\.pdf \((\d+) pages?", log).group(1))
    assert pages <= 6, f"the report has {pages} pages; the limit is 6"
    return pages, len(re.findall(r"Underfull \\[hv]box", log))


if __name__ == "__main__":
    write_numbers()
    write_tables()
    for draw in (fig_phase1, fig_phase3, fig_phase4, fig_phase5):
        draw()
    used = check_tex_keys()
    pages, underfull = compile_pdf()
    print(f"report/report.pdf: {pages} pages; {len(used)} of {len(M)} values used; "
          f"no overfull boxes; {underfull} underfull warnings")
