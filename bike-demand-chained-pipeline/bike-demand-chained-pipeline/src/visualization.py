"""Reusable plotting helpers for the report/notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_loss_curve(loss_history, title="Gradient Descent Convergence"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(loss_history)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.2)
    return fig, ax


def plot_complexity_curve(degrees, train_scores, val_scores):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(degrees, train_scores, marker="o", label="Train")
    ax.plot(degrees, val_scores, marker="o", label="Validation")
    ax.set_xlabel("Model complexity / polynomial degree")
    ax.set_ylabel("R²")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig, ax
