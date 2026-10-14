"""Reusable plotting helpers for the report/notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt

# Fixed categorical order (never cycled), plus ink for observed data.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#3d3d3a"
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


def plot_complexity_curve(degrees, train_scores, val_scores):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(degrees, train_scores, marker="o", label="Train")
    ax.plot(degrees, val_scores, marker="o", label="Validation")
    ax.set_xlabel("Model complexity / polynomial degree")
    ax.set_ylabel("R²")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig, ax
