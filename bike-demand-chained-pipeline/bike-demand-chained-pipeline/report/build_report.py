"""Build report/report.pdf from the executed notebook and the saved artifacts.

Every number is read from the executed notebook's outputs or from an
artifact the notebook saved. The few values recomputed here (the Phase 1
sweep and design table, decile tables, the Phase 5 cutoff scan, all used for
the figures) are asserted to appear in the notebook's own printed output.
Run after executing notebooks/final_submission.ipynb:  python report/build_report.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.data import compute_team_seed, load_data, seeded_split  # noqa: E402
from src.features import Phase1Features  # noqa: E402
from src.operator_checks import demand_decile_edges, errors_by_demand_decile  # noqa: E402
from src.phase1_gradient_descent import (  # noqa: E402
    LinearRegressionGD, largest_curvature, load_phase1_artifact)
from src.phase2_polynomial import Phase2Features, load_phase2_artifact  # noqa: E402
from src.phase3_bias_variance import load_phase3_artifact  # noqa: E402
from src.phase4_regularization import load_phase4_artifact  # noqa: E402
from src.phase5_logistic import (  # noqa: E402
    high_demand_cutoffs, high_demand_labels, sigmoid, threshold_scan)

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

# ---------------------------------------------------------------------------
# Figures (greyscale only)
# ---------------------------------------------------------------------------

FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.2, "savefig.dpi": 200})
STYLES = [("black", "-"), ("0.45", "--"), ("black", ":"), ("0.6", "-."), ("0.3", (0, (5, 1)))]


def fig_phase1():
    fig, ax = plt.subplots(figsize=(4.8, 2.3))
    for (color, ls), m in zip(STYLES + [("0.75", "-")], (0.005, 0.1, 0.5, 1.0, 1.9, 2.1)):
        r = sweep[m]
        ax.plot(range(1, len(r.grad_history) + 1), r.grad_history, color=color, ls=ls,
                label=f"{m:g}/λmax ({r.status}, {r.iterations})")
    ax.axhline(p1["settings"]["tol"], color="black", lw=0.6)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("update"), ax.set_ylabel("max |gradient|")
    ax.legend(fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(), fig.savefig(FIGURES / "fig1.png"), plt.close(fig)


def fig_phase3():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    s, c = seeded, chrono
    ax.plot(s["n_features"], s["train_r2"], color="black", ls="-", marker="o", ms=2.5, label="train (seeded)")
    ax.plot(s["n_features"], s["val_r2"], color="0.45", ls="--", marker="s", ms=2.5, label="validation (seeded)")
    ax.plot(s["n_features"], c["val_r2"], color="black", ls=":", marker="^", ms=2.5, label="validation (chronological)")
    ax.axvline(v["p3_target_n"], color="0.6", lw=0.6)
    ax.set_ylim(0.85, 0.945)
    ax.set_xlabel("number of features (seeded design)"), ax.set_ylabel("R²")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(), fig.savefig(FIGURES / "fig2.png"), plt.close(fig)


def fig_phase4():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2))
    for (color, ls), (label, path) in zip(STYLES, paths.items()):
        axes[0].plot(path["lambda"], path["val_r2"], color=color, ls=ls, label=label)
        axes[1].plot(path["lambda"], path["nonzero"], color=color, ls=ls, label=label)
    for ax, ylabel in zip(axes, ("validation R²", "nonzero weights")):
        ax.set_xscale("log"), ax.set_xlabel("λ"), ax.set_ylabel(ylabel)
    axes[0].set_ylim(0.85, 0.935)
    axes[1].legend(fontsize=6, frameon=False)
    fig.tight_layout(), fig.savefig(FIGURES / "fig3.png"), plt.close(fig)


def fig_phase5():
    design = Phase2Features.from_dict(p4["features"])
    columns = [0] + [f["index"] for f in p4["lasso_surviving_features"]]
    prob = sigmoid(design.transform(val_df)[:, columns] @ np.asarray(p5["weights"]))
    label = high_demand_labels(val_df, high_demand_cutoffs(train_df))
    scan = threshold_scan(label, prob, np.round(np.arange(0.05, 0.951, 0.01), 2))
    best = scan.loc[scan["threshold"] == p5["threshold"]].iloc[0]
    assert abs(best["f1"] - v["p5_tuned"]["f1"]) < 1e-12
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for (color, ls), column in zip(STYLES, ("f1", "precision", "recall")):
        ax.plot(scan["threshold"], scan[column], color=color, ls=ls, label=column)
    ax.axvline(p5["threshold"], color="0.6", lw=0.6)
    ax.set_xlabel("probability cutoff"), ax.set_ylabel("validation score")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(), fig.savefig(FIGURES / "fig4.png"), plt.close(fig)


for draw in (fig_phase1, fig_phase3, fig_phase4, fig_phase5):
    draw()

# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def f(x, d=4):
    return f"{x:.{d}f}".replace("-", "−")


S, C = seeded, chrono
deg3, target, deg4 = "hour x workingday x temp", v["p3_target"], "hour x workingday x yr x season"
ridge, lasso, enet = v["p4_sel"]["Ridge"], v["p4_sel"]["Lasso"], v["p4_sel"]["Elastic Net"]
cmp_ = v["bonus"]
mse_col, asym_col = cmp_.columns[0], cmp_.columns[1]
tuned, half = v["p5_tuned"], v["p5_half"]
verd = v["p4_verdicts"]
lc = curves.set_index(["design", "fraction"])
big = "largest extended design"
odds_top = v["p5_odds"].head(6)


def verdict_rows():
    rows = []
    for column, row in verd.iterrows():
        if row["terms_in_target"] > 0:
            test, change = "ablation", row["ablation_val_r2_change"]
        elif row.get("add_back_columns_kept", 0) == 0:
            test, change = "add back", np.nan
        else:
            test, change = "add back", row["add_back_val_r2_change"]
        change_text = "exact combination" if np.isnan(change) else f(change)
        kept = f"{int(row['terms_surviving_lasso'])}/{int(row['terms_in_target'])}" \
            if row["terms_in_target"] > 0 else "–"
        rows.append(f"<tr><td>{column}</td><td>{test}</td><td>{change_text}</td>"
                    f"<td>{f(row['alone_val_r2'], 3)}</td><td>{kept}</td><td>{row['verdict'].split(' (')[0]}</td></tr>")
    return "\n".join(rows)


HTML = f"""
<h1>Predicting hourly bike-sharing demand with a chained regression–classification pipeline</h1>
<p class="sub">Machine Learning, Winter 2026 — Project 1. Team seed {v['seed']}.</p>

<h2>1. Problem and data</h2>
<p>The task is to predict the number of bikes rented per hour (<i>cnt</i>) and, finally, to classify hours as high demand. The labelled file contains {v['n_all']:,} hours over {v['n_days']} days (days 1–19 of each month, 2011–2012); the hidden test file contains {v['n_test']} hours, all on day 20. The team seed, the SHA-256 of the sorted, underscore-joined student IDs modulo 100,000 (the brief's example reproduces 41,698), splits the labelled rows into {v['n_train']:,} training and {v['n_val']:,} validation hours. Every encoder, scaler and centring mean is fitted on the training portion only. Three properties shape the design: <i>workingday</i> is an exact function of <i>weekday</i> and <i>holiday</i>, <i>season</i> is an exact function of <i>mnth</i> in this file, and the year, month and weekday of <i>dteday</i> coincide exactly with <i>yr</i>, <i>mnth</i> and <i>weekday</i>. Phases are chained through saved artifacts. Phases 2–5 were preceded by written expectations; those of Phase 1 and the bonus were reconstructed afterwards (Section 8). Besides R² and RMSE, every regression model is checked on the hours that matter to the operator: the mean shortfall in the top demand decile (actual demand above {v['top_cut']} bikes), grouped both by actual and by predicted demand.</p>

<h2>2. Phase 1 — Gradient descent</h2>
<p><b>Method.</b> A linear model with main effects only was fitted by full-batch gradient descent on L(w) = ‖Xw − y‖²/(2n), with gradient Xᵀ(Xw − y)/n. Hour, weekday, weathersit (level 4, absent from the training portion, merged into 3) and month were one-hot encoded with one reference level each; <i>yr</i>, <i>holiday</i>, <i>temp</i>, <i>hum</i> and <i>windspeed</i> enter as numbers, and <i>workingday</i>, <i>season</i>, <i>atemp</i> and <i>instant</i> were excluded as exact or near copies. Month was preferred to season (validation R² {f(v['p1_month_r2'])} vs {f(v['p1_season_r2'])}); the elapsed-day trend from <i>dteday</i> added only {f(v['p1_trend_gain'])} R² while raising the condition number from {v['p1_cond']:.0f} to {v['p1_cond_trend']:,.0f} ({v['p1_iter_trend']:,} instead of {v['p1_iter_notrend']} updates) and was left out. Because the loss is quadratic, gradient descent is stable only for learning rates below 2/λ<sub>max</sub>, where λ<sub>max</sub> = {f(v['p1_lam'])} is the largest eigenvalue of XᵀX/n. Learning rates of 0.005–2.1/λ<sub>max</sub> were compared (Fig. 1); convergence was declared when max|gradient| &lt; 10⁻⁴ (cap 50,000 updates). The loop uses the equivalent precomputed form Hw − b with H = XᵀX/n and b = Xᵀy/n, and asserts at the start and end of every fit that both forms give the same gradient.</p>
<p><b>Results.</b> 2.1/λ<sub>max</sub> diverged after {v['p1_div_iter']} updates; 0.005/λ<sub>max</sub> reached the cap with max|gradient| {f(v['p1_slow_grad'], 4)}; all rates between converged to the same loss. The chosen rate 1.9/λ<sub>max</sub> = {f(v['p1_lr'], 3)} converged in {v['p1_iter']} updates, within {f(v['p1_gap_exact'], 4)} bikes of the exact least-squares solution. Training R² {f(v['p1_train_r2'])}, validation R² {f(v['p1_val_r2'])} (RMSE {f(v['p1_val_rmse'], 1)}): the model underfits. Hours above {v['top_cut']} bikes were under-predicted by {f(v['p1_top_short'], 1)} bikes on average, and even the hours predicted to be busiest by {f(-v['p1_toppred_err'], 1)}; {v['p1_negative']} validation hours were predicted below zero. <b>Passed on:</b> the weight vector, learning rate, update count, feature list and fitted scaler; {f(float(v['resid_between']), 1)}% of the training residual variance lay between hour × working-day cells.</p>
<img src="fig1.png" width="340" height="163"/>
<p class="cap">Figure 1. Phase 1 learning-rate sweep: max|gradient| against update (log–log); horizontal line: tolerance.</p>

<h2>3. Phase 2 — Polynomial regression</h2>
<p><b>Method.</b> Interaction groups were chosen from Phase 1's training residuals, ranked by the share of residual variance lying between groups and re-ranked after removing the confirmed groups. Degree is interpreted as interaction order: degree 2 adds hour × workingday, hour × yr, hour × season, hour-block × weathersit and temp × hum; degree 3 adds hour × workingday × yr and hour × workingday × temp with their lower-order terms. Terms are explicit lists (no products of two levels of one column, no squares of binary columns); every non-hour factor is centred at its training mean before multiplying, which lowered the condition number of the full design from {v['p2_cond_unc']:,} to {v['p2_cond']:,} without changing what it can express. Each group was added in turn, every fit warm-started from the Phase 1 weights (new weights zero, predictions asserted identical) and run with the same 1.9/λ<sub>max</sub> rule.</p>
<p><b>Results.</b> hour × workingday alone raised validation R² from {f(S.loc['(Phase 1)','val_r2'])} to {f(S.loc['hour x workingday','val_r2'])}; degree 2 reached {f(S.loc['temp x hum','val_r2'])} and degree 3 {f(S.loc[deg3,'val_r2'])} (RMSE {f(S.loc[deg3,'val_rmse'],1)}, {int(S.loc[deg3,'n_features'])} features, {int(S.loc[deg3,'iterations']):,} updates). The train–validation R² gap was {f(S.loc[deg3,'train_val_r2_gap'])}, with near-equal RMSE, i.e. sampling noise. The top-predicted-decile shortfall fell from {f(S.loc['(Phase 1)','top_predicted_decile_shortfall'],1)} to {f(S.loc[deg3,'top_predicted_decile_shortfall'],1)} bikes. temp × hum added only {f(S.loc['temp x hum','val_r2']-S.loc['hour-block x weathersit','val_r2'])} R². <b>Passed on:</b> degree 3, the 242-feature list with lineage, weights and scaler.</p>

<h2>4. Phase 3 — Bias–variance trade-off</h2>
<p><b>Method.</b> The sweep followed Phase 2's order and continued beyond it (hour × holiday, hour × weekday, hour × month, full hour × weathersit and a degree-4 group hour × workingday × yr × season) to reach the high-variance side; columns that were exact combinations of earlier ones were detected by QR factorisation and dropped. Learning curves were drawn for the degree-3 and the largest design. The one chronological split held out the last 20% of labelled days (from {v['p3_first_day']}, whole days), refitting all preprocessing on the earlier days. The target rule, fixed beforehand, took the converged fit with the fewest features within 0.002 of the best seeded validation R².</p>
<p><b>Results.</b> Seeded validation R² rose from {f(S.loc[deg3,'val_r2'])} to a plateau of {f(S.loc[deg4,'val_r2'])} at {int(S.loc[deg4,'n_features'])} features while the gap widened from {f(S.loc[deg3,'train_val_r2_gap'])} to {f(S.loc[deg4,'train_val_r2_gap'])} (Fig. 2). The degree-3 learning curve was flat with train ≈ validation RMSE (bias-limited); the largest design showed a gap of {f(lc.loc[(big,0.25),'val_rmse']-lc.loc[(big,0.25),'train_rmse'],1)} bikes at 25% of the data, closing to {f(lc.loc[(big,1.0),'val_rmse']-lc.loc[(big,1.0),'train_rmse'],1)} (variance-limited). Chronologically, degree 3 scored R² {f(C.loc[deg3,'val_r2'])} (RMSE {f(C.loc[deg3,'val_rmse'],1)}); the extensions up to full hour × weathersit improved it to {f(C.loc['hour x weathersit','val_r2'])}, but degree 4 fell to {f(C.loc[deg4,'val_r2'])}, because no 2012-autumn hour exists in the earlier days. <b>Diagnosis:</b> degree 3 is bias-limited; variance appears only at degree 4. <b>Passed on:</b> the {v['p3_target_n']}-feature design up to {target.replace(' x ', ' × ')} (seeded R² {f(S.loc[target,'val_r2'])}, chronological {f(C.loc[target,'val_r2'])}) and the noise candidate temp × hum.</p>
<img src="fig2.png" width="250" height="174"/>
<p class="cap">Figure 2. Phase 3 complexity sweep; vertical line: the {v['p3_target_n']}-feature target.</p>

<h2>5. Phase 4 — Regularisation</h2>
<p><b>Method.</b> One objective was used for all methods, J(w) = ‖Xw − y‖²/(2n) + λ[ρΣ|w<sub>j</sub>| + (1 − ρ)Σw<sub>j</sub>²/2] with the intercept unpenalised; Ridge (ρ = 0), Lasso (ρ = 1) and Elastic Net (ρ ∈ {{0.1, 0.5, 0.9}}). Each update takes a gradient step on the smooth part (the Ridge term adds λ(1 − ρ) to the diagonal of H) and then soft-thresholds: every penalised weight is moved towards zero by ηλρ and set to exactly zero if it would cross it. The step is 1.9/L for Ridge and 1/L with an L1 part (L = λ<sub>max</sub> + λ(1 − ρ)), so the objective never increases. Each method was fitted along a log-spaced λ path from its largest value downwards, every fit warm-started from the previous one; selection used the seeded validation set alone (no k-fold, as the brief fixes one holdout): Ridge by best R², Lasso and Elastic Net by the largest λ within 0.002 of the path's best.</p>
<p><b>Results.</b> All L1-type paths first peaked at the last grid value, so they were extended to λ = 10⁻⁵ at the same spacing; the first 30 values reproduced exactly and only the Elastic Net choice changed. Ridge was best (λ = {ridge['lambda']:.2g}, R² {f(ridge['val_r2'],6)}, equal to the unpenalised fit); Lasso (λ = {f(lasso['lambda'],3)}) kept {int(lasso['nonzero'])} of 564 weights at R² {f(lasso['val_r2'])}; Elastic Net chose ρ = {enet['l1_ratio']:g} (λ = {f(enet['lambda'],4)}, R² {f(enet['val_r2'])}, {int(enet['nonzero'])} nonzero). With no variance left to remove, the penalty only added bias, and it raised the top-decile shortfall from {f(S.loc[target,'top_decile_shortfall'],1)} to {f(lasso['top_decile_shortfall'],1)} (Lasso) bikes (Fig. 3). Column verdicts (Table 1) combine a grouped ablation — the design rebuilt without the column and every term derived from it, refitted with Ridge — or an add-back test, the R² of the column alone in its full encoding, and Lasso survival: <i>useful</i> if removing it costs at least 0.002 R²; otherwise <i>redundant</i> if alone R² ≥ 0.01; otherwise <i>uninformative</i>. Holiday keeps most of its terms under Lasso yet costs nothing when removed: a surviving coefficient is not evidence of usefulness. <b>Passed on:</b> the {v['p4_survivors']} Lasso survivors (from {len(v['p4_columns'])} original columns) with lineage, and Ridge as the recommended regression model.</p>
<img src="fig3.png" width="500" height="162"/>
<p class="cap">Figure 3. Phase 4 regularisation paths: validation R² and number of nonzero weights against λ.</p>
<table>
<tr><th>column</th><th>test</th><th>ΔR²</th><th>alone R²</th><th>Lasso kept</th><th>verdict</th></tr>
{verdict_rows()}
</table>
<p class="cap">Table 1. Column verdicts (validation; ΔR² relative to the Ridge model).</p>

<h2>6. Phase 5 — Logistic regression</h2>
<p><b>Method.</b> An hour is labelled high demand if its count exceeds the 75th percentile of training hours in the same hour × working-day × year group (96 cutoffs from the training portion, applied unchanged to validation). A global cutoff would label eight hours of the day as never high, and a cutoff without year would label 2011 and 2012 hours very differently. {v['p5_pos']:,} training hours ({f(v['p5_pos_rate'],1)}%) are positive ({f(v['p5_val_pos'],1)}% in validation). The inputs are exactly the {v['p4_survivors']} Lasso survivors with the fitted scaler. The log-loss was minimised by gradient descent with gradient Xᵀ(p − y)/m computed explicitly at each step; since its curvature is at most λ<sub>max</sub>(XᵀX/m)/4 = {f(v['p5_curv'])}, the step 1.9/L = {f(v['p5_lr'],3)} is safe. The probability cutoff maximises validation F1 over 0.05–0.95.</p>
<p><b>Results.</b> The fit converged in {v['p5_iter']:,} updates. ROC-AUC {f(tuned['roc_auc'],3)}; at the tuned cutoff {f(v['p5_cut'],2)}: accuracy {f(tuned['accuracy'],3)} (always-0 baseline {f(tuned['always_0_accuracy'],3)}), precision {f(tuned['precision'],3)}, recall {f(tuned['recall'],3)}, F1 {f(tuned['f1'],3)}; confusion matrix TN {tuned['tn']}, FP {tuned['fp']}, FN {tuned['fn']}, TP {tuned['tp']}. At 0.5 accuracy is higher ({f(half['accuracy'],3)}) but F1 lower ({f(half['f1'],3)}, recall {f(half['recall'],3)}): with a quarter of the hours positive, accuracy rewards missing busy hours (Fig. 4). Temperature has the largest odds ratio (×{odds_top.iloc[0]['odds']:.2f} per standard deviation), followed by month dummies. The label still carries seasonality ({f(v['p5_jan'],1)}% of January but {f(v['p5_jun'],1)}% of June training hours are positive): a calendar-only ranking reaches ROC-AUC {f(v['p5_auc_cal'],3)}, and within a month the classifier reaches {f(v['p5_auc_month'],3)} (within a season {f(v['p5_auc_season_lo'],3)}–{f(v['p5_auc_season_hi'],3)}), which measures its skill beyond the calendar.</p>
<img src="fig4.png" width="250" height="174"/>
<p class="cap">Figure 4. Phase 5 precision, recall and F1 against the probability cutoff; vertical line: tuned cutoff.</p>

<h2>7. Bonus — asymmetric cost</h2>
<p>Under-prediction was taken to cost three times as much as over-prediction of the same size, keeping the squared error: ℓ(r) = ½c(r)r² with r = ŷ − y and c = 3 if r &lt; 0, else 1. Its derivative c(r)r is continuous, so the gradient is Xᵀ(c ⊙ r)/n = Xᵀr/n + 2Xᵀmin(r, 0)/n, and the gradient's Lipschitz constant is 3λ<sub>max</sub>; the minimiser is the 0.75 expectile rather than the mean. (A cost linear in the number of bikes would give the 75th percentile, but it has no gradient at zero, so fixed-step gradient descent could not converge on it.) On Phase 1's features the Phase 1 learning rate diverged within {v['b_div_iter']} updates, while the rate divided by three ({f(v['b_lr'],3)}) converged in {v['b_iter']} updates. Predictions rose by {f(v['b_shift'],1)} bikes per hour on average; R² fell from {f(cmp_.loc['r2',mse_col],3)} to {f(cmp_.loc['r2',asym_col],3)} while the asymmetric cost fell from {cmp_.loc['asymmetric_cost',mse_col]:,.0f} to {cmp_.loc['asymmetric_cost',asym_col]:,.0f}; bikes short per hour fell from {f(cmp_.loc['mean_shortfall_bikes',mse_col],1)} to {f(cmp_.loc['mean_shortfall_bikes',asym_col],1)} and idle bikes rose from {f(cmp_.loc['mean_surplus_bikes',mse_col],1)} to {f(cmp_.loc['mean_surplus_bikes',asym_col],1)}; the top-decile shortfall fell from {f(v['b_top_mse'],1)} to {f(v['b_top_asym'],1)}. Much of the lift went to hours already over-predicted, a consequence of Phase 1's underfitting. On Phase 2's features the same loss shifted predictions by only {f(v['f_shift'],1)} bikes, cost {f(v['f_r2']-v['f_r2_asym'],3)} R² (from {f(v['f_r2'],3)}) and reduced the top-decile shortfall from {f(v['f_top'],1)} to {f(v['f_top_asym'],1)} bikes.</p>

<h2>8. Limitations</h2>
<p>(i) <b>Seasonal label leakage:</b> the Phase 5 label removes slot and year levels but not the season, so a large part of the classifier's ROC-AUC is reproducible from the calendar alone. (ii) <b>Chronological shortfall:</b> on later, busier days the models under-predict their own busiest hours (top-predicted-decile mean error {f(C.loc[target,'top_predicted_decile_mean_error'],1)} bikes for the target design, against {f(S.loc[target,'top_predicted_decile_mean_error'],1)} on the seeded split); growth beyond the training period is not extrapolated. (iii) <b>One validation set:</b> λ, the Phase 3 target and the Phase 5 cutoff were tuned and scored on the same seeded validation set, so those scores are slightly optimistic. (iv) <b>Grid revision:</b> the Phase 4 L1 paths were extended after the first official run because their best values lay at the grid edge; the revision is recorded in the notebook and changed only the Elastic Net choice. (v) The Phase 1 and bonus expectations were reconstructed after their runs and are labelled as such.</p>

<h2>9. Final submission</h2>
<p>The submitted predictions come from Phase 4's Ridge model (λ = {ridge['lambda']:.2g}) on the {v['p3_target_n']}-feature design. With every choice frozen on the seeded split, the model was refitted on all {v['n_all']:,} labelled hours with the same procedure (encoder, scaler and centring means refitted on the combined rows; the same {v['s_dropped']} exact combinations dropped). The final fit converged in {v['s_iter']:,} updates (training R² {f(v['s_r2'])}, RMSE {f(v['s_rmse'],2)}). The {v['n_test']} test hours were predicted in the sample file's order; {v['s_clipped']} negative predictions were set to zero (mean {f(v['s_mean'],1)}, maximum {f(v['s_max'],1)} bikes).</p>
"""

CSS = """
body { font-family: serif; font-size: 9.6pt; line-height: 1.32; color: black; }
h1 { font-size: 14pt; text-align: center; margin: 0 0 2pt 0; }
p.sub { text-align: center; font-size: 9.5pt; margin: 0 0 6pt 0; }
h2 { font-size: 11pt; margin: 7pt 0 2pt 0; }
p { text-align: justify; margin: 0 0 4pt 0; }
p.cap { font-size: 8.3pt; text-align: center; margin: 1pt 0 5pt 0; }
img { display: block; margin: 2pt auto 0 auto; }
table { border-collapse: collapse; margin: 3pt auto 0 auto; font-size: 8.3pt; }
th, td { border: 0.5pt solid black; padding: 1pt 4pt; text-align: center; }
"""


def build_pdf(path: Path) -> int:
    story = fitz.Story(html=HTML, user_css=CSS, archive=fitz.Archive(str(FIGURES)))
    writer = fitz.DocumentWriter(str(path))
    page_rect = fitz.paper_rect("a4")
    body = page_rect + (54, 50, -54, -50)
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(body)
        story.draw(device)
        writer.end_page()
    writer.close()
    doc = fitz.open(str(path))
    for number, page in enumerate(doc, start=1):
        page.insert_text((page_rect.width / 2 - 8, page_rect.height - 28), f"{number}",
                         fontsize=9, fontname="tiro")
    doc.saveIncr()
    count = doc.page_count
    doc.close()
    return count


if __name__ == "__main__":
    pages = build_pdf(OUT / "report.pdf")
    print(f"report/report.pdf: {pages} pages")
    assert pages <= 6, "the report must not exceed 6 pages"
