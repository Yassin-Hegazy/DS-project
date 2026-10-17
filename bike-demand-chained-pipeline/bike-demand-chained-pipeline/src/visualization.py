"""Reusable plotting helpers for the report/notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical order (never cycled), plus ink for observed data.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#3d3d3a"
# One hue, light -> dark, for ordered levels such as quantile bins.
SEQUENTIAL_BLUE = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
GRID = dict(alpha=0.2)


def plot_loss_curve(loss_history, title="Gradient Descent Convergence"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(loss_history)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(**GRID)
    return fig, ax


def plot_gd_runs(runs: dict, tol: float, title: str):
    """Loss and largest |gradient| per iteration for several GD runs.

    runs maps a label to a GDResult. Iteration axis is logarithmic so runs
    that stop after a few hundred steps and runs that hit the cap both show.
    """
    fig, (ax_loss, ax_grad) = plt.subplots(1, 2, figsize=(12, 4.5))
    for color, (label, result) in zip(SERIES_COLORS, runs.items()):
        steps = range(1, len(result.loss_history) + 1)
        name = f"{label} ({result.status}, {result.iterations} it.)"
        ax_loss.plot(steps, result.loss_history, color=color, lw=2, label=name)
        ax_grad.plot(steps, result.grad_history, color=color, lw=2, label=name)
    ax_grad.axhline(tol, color=INK, ls="--", lw=1, label=f"tol = {tol:g}")
    for ax, ylabel in ((ax_loss, "Training loss"), (ax_grad, "max |gradient|")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(**GRID)
    ax_grad.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_shift_by_hour(hour_table, title: str):
    """Actual demand and both models' mean predictions by hour, per day type."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (workingday, label) in zip(axes, ((0, "Non-working days"), (1, "Working days"))):
        part = hour_table[hour_table["workingday"] == workingday]
        ax.plot(part["hr"], part["actual"], color=INK, lw=2, ls=":", marker="o",
                ms=4, label="Actual mean")
        ax.plot(part["hr"], part["pred_mse"], color=SERIES_COLORS[0], lw=2,
                marker="s", ms=4, label="MSE model")
        ax.plot(part["hr"], part["pred_asym"], color=SERIES_COLORS[1], lw=2,
                marker="^", ms=4, label="Asymmetric model (k=3)")
        ax.set_title(label)
        ax.set_xlabel("Hour of day")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(**GRID)
    axes[0].set_ylabel("Mean rentals per hour (validation)")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_residual_profiles(panels: list, title: str):
    """Mean Phase 1 residual by hour, one line per level of a second column.

    panels: list of (panel title, cell table with columns hr / level /
    mean_residual, level column name, ordered). Ordered levels (e.g. bins)
    use one hue light -> dark; unordered levels use the categorical order.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (panel_title, table, level, ordered) in zip(axes.flat, panels):
        levels = list(dict.fromkeys(table[level]))
        colors = (SEQUENTIAL_BLUE[-len(levels):] if ordered
                  else SERIES_COLORS[:len(levels)])
        for color, value in zip(colors, levels):
            part = table[table[level] == value]
            ax.plot(part["hr"], part["mean_residual"], color=color, lw=2,
                    marker="o", ms=3, label=f"{level} = {value}")
        ax.axhline(0, color=INK, lw=1)
        ax.set_title(panel_title)
        ax.set_xticks(range(0, 24, 3))
        ax.grid(**GRID)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Hour of day")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean residual, actual - predicted (bikes)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_shortfall_by_hour(tables: dict, title: str):
    """Mean bikes short per hour, per day type, for one or more models.

    tables maps a model label to an errors_by_hour_daytype table.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (workingday, label) in zip(axes, ((0, "Non-working days"), (1, "Working days"))):
        for color, (model, table) in zip(SERIES_COLORS, tables.items()):
            part = table[table["workingday"] == workingday]
            ax.plot(part["hr"], part["mean_shortfall_bikes"], color=color, lw=2,
                    marker="o", ms=4, label=model)
        ax.set_title(label)
        ax.set_xlabel("Hour of day")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(**GRID)
    axes[0].set_ylabel("Mean shortfall (bikes per hour, validation)")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


DEGREE_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}


def _sweep_panel(ax, table, columns: dict, metric: str):
    """Lines for each (label -> column) against feature count; marker shape
    shows the degree; a non-converged fit is drawn as a red-ringed point."""
    for color, (label, column) in zip(SERIES_COLORS, columns.items()):
        ax.plot(table["n_features"], table[column], color=color, lw=2, label=label)
        for degree, marker in DEGREE_MARKERS.items():
            part = table[table["degree"] == degree]
            ax.plot(part["n_features"], part[column], ls="none", marker=marker,
                    ms=7, color=color)
    flagged = table[table["status"] != "converged"]
    for column in columns.values():
        ax.plot(flagged["n_features"], flagged[column], ls="none", marker="o", ms=14,
                mfc="none", mec="#d03b3b", mew=2)
    ax.set_xlabel("Number of features")
    ax.set_ylabel(metric)
    ax.grid(**GRID)


def plot_complexity_sweep(table, title: str, peak_features: int, extension_start: int):
    """Train vs validation R2 and RMSE against feature count."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    panels = ((axes[0], {"train": "train_r2", "validation": "val_r2"}, "R²"),
              (axes[1], {"train": "train_rmse", "validation": "val_rmse"}, "RMSE (bikes)"))
    for ax, columns, metric in panels:
        ax.axvspan(extension_start, table["n_features"].max() + 15, color=INK, alpha=0.06)
        ax.axvline(peak_features, color=INK, ls=":", lw=1.5)
        _sweep_panel(ax, table, columns, metric)
    axes[0].text(peak_features, axes[0].get_ylim()[0], " validation peak",
                 rotation=90, va="bottom", fontsize=8, color=INK)
    axes[0].text(extension_start, axes[0].get_ylim()[1], " beyond Phase 2",
                 va="top", fontsize=8, color=INK)
    handles = [plt.Line2D([], [], color=c, lw=2, label=l)
               for c, l in zip(SERIES_COLORS, ("train", "validation"))]
    handles += [plt.Line2D([], [], color=INK, ls="none", marker=m, label=f"degree {d}")
                for d, m in DEGREE_MARKERS.items()]
    handles.append(plt.Line2D([], [], ls="none", marker="o", ms=10, mfc="none",
                              mec="#d03b3b", mew=2, label="not converged (flagged)"))
    axes[1].legend(handles=handles, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_split_comparison(seeded, chronological, title: str):
    """Validation R2 and RMSE of the same sweep on the seeded and the
    chronological split (plus chronological training R2 for reference)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, column, metric in ((axes[0], "val_r2", "Validation R²"),
                               (axes[1], "val_rmse", "Validation RMSE (bikes)")):
        for color, (label, table) in zip(SERIES_COLORS, (("seeded random split", seeded),
                                                         ("chronological split", chronological))):
            ax.plot(table["n_features"], table[column], color=color, lw=2, marker="o",
                    ms=5, label=label)
        ax.set_xlabel("Number of features")
        ax.set_ylabel(metric)
        ax.grid(**GRID)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_learning_curves(curves: dict, title: str):
    """Train and validation RMSE against training-set size, one panel per model."""
    fig, axes = plt.subplots(1, len(curves), figsize=(6.5 * len(curves), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, table) in zip(axes, curves.items()):
        for color, (name, column) in zip(SERIES_COLORS, (("train", "train_rmse"),
                                                         ("validation", "val_rmse"))):
            ax.plot(table["n_train"], table[column], color=color, lw=2, marker="o",
                    ms=6, label=name)
        flagged = table[table["status"] != "converged"]
        ax.plot(flagged["n_train"], flagged["val_rmse"], ls="none", marker="o", ms=14,
                mfc="none", mec="#d03b3b", mew=2, label="not converged (flagged)")
        ax.set_title(label)
        ax.set_xlabel("Training rows")
        ax.grid(**GRID)
    axes[0].set_ylabel("RMSE (bikes)")
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_regularization_paths(paths: dict, chosen: dict, title: str):
    """Validation R2 and number of nonzero weights against lambda, per method.

    paths maps a method label to its path table; chosen maps the label to the
    selected lambda (drawn as a ringed point).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for color, (label, table) in zip(SERIES_COLORS, paths.items()):
        for ax, column in ((axes[0], "val_r2"), (axes[1], "nonzero")):
            ax.plot(table["lambda"], table[column], color=color, lw=2, marker="o",
                    ms=3, label=label)
            pick = table[table["lambda"] == chosen[label]]
            ax.plot(pick["lambda"], pick[column], ls="none", marker="o", ms=11,
                    mfc="none", mec=color, mew=2)
    for ax, ylabel in ((axes[0], "Validation R²"), (axes[1], "Nonzero weights")):
        ax.set_xscale("log")
        ax.set_xlabel("λ (log scale)")
        ax.set_ylabel(ylabel)
        ax.grid(**GRID)
    axes[0].legend(fontsize=8, title="ringed = selected", title_fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_complexity_curve(degrees, train_scores, val_scores):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(degrees, train_scores, marker="o", label="Train")
    ax.plot(degrees, val_scores, marker="o", label="Validation")
    ax.set_xlabel("Model complexity / polynomial degree")
    ax.set_ylabel("R²")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig, ax
