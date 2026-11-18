"""Model-comparison figures for report §3.3 (owner: P2 - Aditya).

Every figure is drawn ONLY from saved experiment outputs in experiments/results/
(model_comparison.csv, cv_folds.csv, confusion_matrices.json, histgb_loss_history.csv),
so each one can be regenerated and checked against the numbers in the report.
Training curves are drawn only for HistGradientBoosting, the one model here that
records a real per-iteration training history (no artificial curves for the others).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def _bar_labels(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=7, xytext=(0, 1), textcoords="offset points")


def comparison_panel(res_dir, fig_dir):
    """3-panel figure for the report: CV macro-F1 (±std) | student-test Top-1 + macro-F1 | log loss."""
    df = pd.read_csv(Path(res_dir) / "model_comparison.csv")
    names = df["model"].tolist()
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    ax = axes[0]
    bars = ax.bar(x, df["cv_macro_f1_mean"], yerr=df["cv_macro_f1_std"], capsize=4, color=COLORS[:len(x)])
    _bar_labels(ax, bars)
    ax.set_title("5-fold GroupKFold CV macro-F1 (±1 std)", fontsize=10)
    ax.set_ylim(max(0, df["cv_macro_f1_mean"].min() - 0.1), 1.0)

    ax = axes[1]
    w = 0.38
    b1 = ax.bar(x - w / 2, df["test_top1"], w, label="Top-1 accuracy", color="#4C72B0")
    b2 = ax.bar(x + w / 2, df["test_macro_f1"], w, label="Macro-F1", color="#55A868")
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    ax.set_title("Student-test (10 trips, all observations)", fontsize=10)
    ax.set_ylim(max(0, min(df["test_top1"].min(), df["test_macro_f1"].min()) - 0.1), 1.0)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    b1 = ax.bar(x - w / 2, df["cv_log_loss_mean"], w, label="CV", color="#8172B3")
    b2 = ax.bar(x + w / 2, df["test_log_loss"], w, label="Student-test", color="#DD8452")
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    ax.set_title("Log loss (lower is better)", fontsize=10)
    ax.legend(fontsize=8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(fig_dir) / "model_comparison_panel.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def overfit_gap(res_dir, fig_dir):
    """Train-fold vs validation-fold Top-1 per model (evidence for over/underfitting)."""
    df = pd.read_csv(Path(res_dir) / "model_comparison.csv")
    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    b1 = ax.bar(x - w / 2, df["cv_train_top1_mean"], w, label="training folds", color="#C44E52")
    b2 = ax.bar(x + w / 2, df["cv_top1_mean"], w, label="validation folds (unseen trips)", color="#4C72B0")
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, fontsize=8)
    ax.set_ylim(max(0, df[["cv_train_top1_mean", "cv_top1_mean"]].min().min() - 0.1), 1.0)
    ax.set_title("Top-1 accuracy: training vs validation folds", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(fig_dir) / "model_overfit_gap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def confusion_grid(res_dir, fig_dir):
    """Student-test confusion matrices (top-1 decision rule) for every model."""
    cms = json.load(open(Path(res_dir) / "confusion_matrices.json"))
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 2.9))
    axes = np.atleast_1d(axes)
    for ax, (name, cm) in zip(axes, cms.items()):
        cm = np.asarray(cm)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"], fontsize=7)
        ax.set_title(name, fontsize=9)
    fig.suptitle("Student-test confusion matrices (candidate level, top-1 rule)", fontsize=10)
    fig.tight_layout()
    out = Path(fig_dir) / "model_confusion_matrices.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def histgb_loss_curve(res_dir, fig_dir):
    """Per-iteration training / validation log loss recorded by HistGradientBoosting."""
    p = Path(res_dir) / "histgb_loss_history.csv"
    if not p.exists():
        return None
    h = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(h["iteration"], h["train_log_loss"], label="training data")
    ax.plot(h["iteration"], h["validation_log_loss"], label="internal validation split")
    ax.set_xlabel("boosting iteration (trees added)")
    ax.set_ylabel("log loss")
    ax.set_title("HistGradientBoosting training history", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(fig_dir) / "model_histgb_loss_curve.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def make_all(res_dir, fig_dir):
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    outs = [comparison_panel(res_dir, fig_dir), overfit_gap(res_dir, fig_dir),
            confusion_grid(res_dir, fig_dir), histgb_loss_curve(res_dir, fig_dir)]
    return [o for o in outs if o is not None]
