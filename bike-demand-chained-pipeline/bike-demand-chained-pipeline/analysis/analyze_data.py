"""Reproducible EDA and exploratory chronological feature ablations.

This is an analysis tool, not the five-phase submission pipeline. No default
random split, test-label lookup, test tuning, or test predictions are made.
Run from anywhere with: python analysis/analyze_data.py
"""
from pathlib import Path
from itertools import combinations_with_replacement
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis"


def features(df, design):
    """Deterministic calendar encodings; observed categorical levels fit later.

    Numeric category meanings are provided by the brief, rather than learned
    from test.csv. No date IDs or target-derived features are used.
    """
    dt = pd.to_datetime(df.dteday)
    f = {}
    f["yr"] = df.yr
    for c in ["workingday", "holiday", "temp", "hum", "windspeed"]:
        f[c] = df[c]
    if design == "raw_linear":
        for c in ["hr", "mnth", "weekday", "season", "weathersit", "atemp"]:
            f[c] = df[c]
        return pd.DataFrame(f, index=df.index)
    for c in ["hr", "mnth", "weekday", "season", "weathersit"]:
        # Enumerated meanings are fixed by the assignment; use reference coding.
        levels = {"hr": range(1, 24), "mnth": range(2, 13),
                  "weekday": range(1, 7), "season": range(2, 5),
                  "weathersit": range(2, 5)}[c]
        for level in levels:
            f[f"{c}={level}"] = df[c].eq(level).astype(float)
    if design == "categorical_linear":
        f["atemp"] = df.atemp
        return pd.DataFrame(f, index=df.index)
    for h in range(1, 24):
        f[f"hr={h}:workingday"] = f[f"hr={h}"] * df.workingday
    if design == "hour_workingday":
        f["atemp"] = df.atemp
        return pd.DataFrame(f, index=df.index)
    # Advanced designs retain every main-effect column from the preceding
    # model and introduce targeted products. No blind expansion of all columns.
    f["atemp"] = df.atemp
    f["elapsed_days"] = (dt - pd.Timestamp("2011-01-01")).dt.days / 365.25
    angle = 2 * np.pi * (dt.dt.dayofyear - 1) / 365.25
    for harmonic in [1, 2]:
        f[f"annual_sin{harmonic}"] = np.sin(harmonic * angle)
        f[f"annual_cos{harmonic}"] = np.cos(harmonic * angle)
    for a, b in combinations_with_replacement(["temp", "hum", "windspeed"], 2):
        f[f"{a}:{b}"] = df[a] * df[b]
    for h in range(1, 24):
        hour = f[f"hr={h}"]
        for c in ["temp", "hum", "yr"]:
            f[f"hr={h}:{c}"] = hour * df[c]
        for c in ["temp", "yr"]:
            f[f"hr={h}:workingday:{c}"] = hour * df.workingday * df[c]
    f["workingday:temp"] = df.workingday * df.temp
    if design in ["structured_no_atemp", "structured_extended_no_atemp"]:
        del f["atemp"]
    if design == "structured_extended_no_atemp":
        for h in range(1, 24):
            hour = f[f"hr={h}"]
            f[f"hr={h}:windspeed"] = hour * df.windspeed
            for c in ["season", "weathersit", "weekday"]:
                for level in {"season":range(2,5), "weathersit":range(2,4),
                              "weekday":range(1,7)}[c]:
                    f[f"hr={h}:{c}={level}"] = hour * df[c].eq(level).astype(float)
    if design == "structured_no_trend":
        del f["elapsed_days"]
    return pd.DataFrame(f, index=df.index)


def fit_gd(x, y, init=None, max_iter=100000, tol=1e-5):
    """Full-batch GD for MSE/(2n), evaluated using sufficient statistics.

    Gram multiplication is algebraically identical to X.T @ (Xw-y)/n.
    Step size = 0.9 / largest Hessian eigenvalue; no library optimizer.
    Standardization, active-column choice and centering use training only.
    """
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    gram = z.T @ z / len(z)
    cross = z.T @ y / len(z)
    lr = .9 / np.linalg.eigvalsh(gram)[-1]
    w = np.zeros(z.shape[1]) if init is None else init.copy()
    history = []
    converged = False
    for i in range(max_iter):
        grad = gram @ w - cross
        if np.max(np.abs(grad)) < tol:
            converged = True
            break
        w -= lr * grad
        if i % 100 == 0:
            history.append(float(.5 * (w @ gram @ w - 2 * w @ cross + y @ y / len(y))))
    return dict(weights=w, mean=mean, scale=scale, iterations=i+1,
                learning_rate=float(lr), converged=converged,
                max_abs_gradient=float(np.max(np.abs(gram @ w - cross))),
                loss_history=history)


def predict(model, x):
    return model["weights"][0] + ((x-model["mean"])/model["scale"]) @ model["weights"][1:]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "data/raw/train.csv")
    test = pd.read_csv(ROOT / "data/raw/test.csv")
    sub = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    dt = pd.to_datetime(train.dteday)
    test_dt = pd.to_datetime(test.dteday)
    summary = {
        "train_rows": len(train), "test_rows": len(test),
        "train_missing_cells": int(train.isna().sum().sum()),
        "test_missing_cells": int(test.isna().sum().sum()),
        "train_duplicate_rows": int(train.duplicated().sum()),
        "train_duplicate_timestamps": int(train.duplicated(["dteday", "hr"]).sum()),
        "overlapping_train_test_ids": len(set(train.instant)&set(test.instant)),
        "submission_id_order_matches": bool(sub.instant.equals(test.instant)),
        "train_date_range": [str(dt.min().date()), str(dt.max().date())],
        "test_date_range": [str(test_dt.min().date()), str(test_dt.max().date())],
        "training_days": int(dt.nunique()),
        "test_days": int(test_dt.nunique()),
        "training_day_of_month": sorted(int(v) for v in dt.dt.day.unique()),
        "test_day_of_month": sorted(int(v) for v in test_dt.dt.day.unique()),
        "absent_hours_within_labeled_calendar_days": int(dt.nunique()*24-len(train)),
        "target_summary": train.cnt.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).to_dict(),
        "temp_atemp_correlation": float(train[["temp","atemp"]].corr().iloc[0,1]),
        "workingday_mismatches": int(train.workingday.ne(
            (train.weekday.between(1,5)&train.holiday.eq(0)).astype(int)).sum()),
        "humidity_zero_rows": int(train.hum.eq(0).sum()),
        "windspeed_zero_rows": int(train.windspeed.eq(0).sum()),
        "weathersit_counts": {str(k):int(v) for k,v in train.weathersit.value_counts().items()},
        "constant_atemp_anomaly": "2012-08-17: 24 rows, atemp=0.2424; temp has 11 distinct values",
    }
    tables = {}
    for c in ["hr", "yr", "mnth", "season", "weathersit", "holiday", "workingday"]:
        tables[c] = train.groupby(c).cnt.agg(["size", "mean", "median", "std"])
        tables[c].to_csv(OUT / f"demand_by_{c}.csv")
    hour = train.pivot_table(index="hr", columns="workingday", values="cnt", aggfunc="mean")
    hour.to_csv(OUT / "hour_by_workingday.csv")
    train.select_dtypes("number").corr().to_csv(OUT / "numeric_correlations.csv")

    # A whole-day cut prevents a single date from straddling the time split.
    ordered = train.sort_values(["dteday", "hr"])
    approx_date = ordered.iloc[int(.8*len(ordered))].dteday
    cut_date = (pd.Timestamp(approx_date)+pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    earlier = ordered[ordered.dteday < cut_date]
    later = ordered[ordered.dteday >= cut_date]
    summary["chronological_diagnostic"] = {
        "cut_date":cut_date, "train_rows":len(earlier), "validation_rows":len(later),
        "train_target_mean":float(earlier.cnt.mean()),
        "validation_target_mean":float(later.cnt.mean()),
        "official_team_seeded_results":False,
    }
    rows = []
    last = None
    designs = ["raw_linear", "categorical_linear", "hour_workingday", "structured_degree2_3",
               "structured_no_atemp", "structured_no_trend", "structured_extended_no_atemp"]
    for design in designs:
        print("Fitting chronological diagnostic:", design, flush=True)
        a, b = features(earlier, design), features(later, design)
        # Fit column availability from earlier data; zero-variance categories
        # are omitted even if they occur in later or hidden data.
        active = a.std(ddof=0) > 1e-12
        a, b = a.loc[:,active], b.loc[:,active]
        x = a.to_numpy(dtype=float)
        xv = b.to_numpy(dtype=float)
        y = earlier.cnt.to_numpy(dtype=float)
        init = np.zeros(x.shape[1]+1)
        if last is not None:
            # Match coefficients in ORIGINAL feature units. This also handles
            # dropping a feature without copying incompatible array positions.
            old, old_names = last
            slope = old["weights"][1:] / old["scale"]
            raw_bias = old["weights"][0] - old["mean"] @ slope
            lookup = dict(zip(old_names, slope))
            new_slope = np.array([lookup.get(c,0.) for c in a.columns])
            init[1:] = new_slope * x.std(axis=0)
            init[0] = raw_bias + x.mean(axis=0) @ new_slope
        model = fit_gd(x, y, init)
        pt, pv = predict(model,x), predict(model,xv)
        # Numerical reference only: not used for training, model selection,
        # warm starts, or any phase artifact. Quantify optimization error.
        z = np.column_stack([np.ones(len(x)),(x-model["mean"])/model["scale"]])
        exact_weights = np.linalg.lstsq(z, y, rcond=None)[0]
        exact_pv = np.column_stack([np.ones(len(xv)),
            (xv-model["mean"])/model["scale"]]) @ exact_weights
        rows.append({"design":design, "features":x.shape[1],
            "train_r2":float(r2_score(y,pt)), "chronological_r2":float(r2_score(later.cnt,pv)),
            "chronological_rmse":float(np.sqrt(mean_squared_error(later.cnt,pv))),
            "negative_predictions":int((pv<0).sum()),
            "learning_rate":model["learning_rate"], "iterations":model["iterations"],
            "converged":model["converged"], "max_abs_gradient":model["max_abs_gradient"],
            "least_squares_reference_r2":float(r2_score(later.cnt,exact_pv)),
            "optimization_prediction_rmse":float(np.sqrt(np.mean((pv-exact_pv)**2)))})
        if design in designs[:4]:
            last = model,a.columns.tolist()
        if design == "structured_degree2_3":
            # Residual diagnosis, including counts for context.
            residual = later[["dteday", "hr", "weathersit", "cnt"]].copy()
            residual["prediction"] = pv
            residual["error"] = pv - later.cnt.to_numpy()
            residual["squared_error"] = residual.error**2
            residual.groupby("hr").agg(n=("cnt","size"),bias=("error","mean"),
                mse=("squared_error","mean")).assign(
                    rmse=lambda v:np.sqrt(v.mse)).to_csv(OUT / "chronological_residuals_by_hour.csv")
            pd.DataFrame({"feature":["intercept"]+a.columns.tolist(),
                          "standardized_coefficient":model["weights"]}).to_csv(
                              OUT / "exploratory_structured_coefficients.csv",index=False)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "chronological_feature_ablation.csv", index=False)
    summary["feature_ablation"] = rows

    # Illustrate why a global high-demand threshold mainly picks busy hours.
    q = earlier.cnt.quantile(.75)
    example = train.assign(global_high=train.cnt.ge(q).astype(int))
    example.groupby("hr").global_high.agg(["size", "mean"]).to_csv(
        OUT / "global_high_demand_rate_by_hour.csv")
    summary["illustrative_global_q75_training_threshold"] = float(q)

    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, ax = plt.subplots(2,2,figsize=(13,9),layout="constrained")
    for c,label in [(0,"Non-working day"),(1,"Working day")]:
        ax[0,0].plot(hour.index,hour[c],marker="o",label=label)
    ax[0,0].set(title="Hour matters differently on working days",xlabel="Hour",ylabel="Mean rentals")
    ax[0,0].legend()
    ax[0,1].hist(train.cnt,bins=50,color="#377eb8",edgecolor="white")
    ax[0,1].set(title="Demand is right-skewed",xlabel="Rentals per hour",ylabel="Labeled hours")
    monthly = train.groupby(["yr","mnth"]).cnt.mean().unstack(0)
    for c,label in [(0,"2011"),(1,"2012")]:
        ax[1,0].plot(monthly.index,monthly[c],marker="o",label=label)
    ax[1,0].set(title="Seasonality and growth occur together",xlabel="Month",ylabel="Mean rentals")
    ax[1,0].legend()
    tmp=train[(train.temp.between(.60,.90)) & (train.atemp<.35)]
    ax[1,1].scatter(train.temp,train.atemp,s=5,alpha=.15,color="#377eb8")
    ax[1,1].scatter(tmp.temp,tmp.atemp,s=20,color="#e6550d",label="Flagged low feels-like values")
    ax[1,1].set(title="Temperature duplication and one anomalous day",xlabel="Normalized temperature",ylabel="Normalized feels-like temperature")
    ax[1,1].legend(loc="upper left")
    fig.suptitle("Bike demand: evidence from the supplied training data",fontsize=16)
    fig.savefig(OUT / "data_insights.png",dpi=160)
    plt.close(fig)
    (OUT / "analysis_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(result.round(5).to_string(index=False))
    print("Output:",OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    with threadpool_limits(limits=1):
        run()
