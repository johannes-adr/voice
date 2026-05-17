#!/usr/bin/env python3
"""Generate training plots from exported CSVs.

Usage:
    python3 plot.py <plots_dir>

The plots_dir should contain the CSVs written by `cargo run --release`:
  - training_log.csv  (epoch, train_loss, val_loss, train_acc, val_acc)
  - predictions.csv   (true_label, pred_label, confidence)

Output PNGs are written to the same directory.
"""

import sys
import os
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BLUE   = "#4d8fff"
RED    = "#ff6b6b"
GREEN  = "#44cc88"
GRAY   = "#888888"

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
})


def load_training_log(path):
    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            vl = row["val_loss"]
            val_loss.append(float(vl) if vl != "NaN" else math.nan)
            ta = row["train_acc"]
            train_acc.append(float(ta) if ta != "NaN" else math.nan)
            va = row["val_acc"]
            val_acc.append(float(va) if va != "NaN" else math.nan)
    return epochs, train_loss, val_loss, train_acc, val_acc


def load_predictions(path):
    true_labels, pred_labels, confidences = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_labels.append(row["true_label"])
            pred_labels.append(row["pred_label"])
            confidences.append(float(row["confidence"]))
    return true_labels, pred_labels, confidences


def plot_loss(epochs, train_loss, val_loss, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_loss, color=BLUE, lw=2, label="Train loss")
    val_epochs = [e for e, v in zip(epochs, val_loss) if not math.isnan(v)]
    val_vals   = [v for v in val_loss if not math.isnan(v)]
    if val_vals:
        ax.plot(val_epochs, val_vals, color=RED, lw=2, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_accuracy(epochs, train_acc, val_acc, out_path):
    acc_epochs  = [e for e, v in zip(epochs, train_acc) if not math.isnan(v)]
    train_vals  = [v * 100 for v in train_acc if not math.isnan(v)]
    val_epochs  = [e for e, v in zip(epochs, val_acc) if not math.isnan(v)]
    val_vals    = [v * 100 for v in val_acc if not math.isnan(v)]
    if not train_vals:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(acc_epochs, train_vals, color=BLUE, lw=2, marker="o", ms=4, label="Train worst-class")
    if val_vals:
        ax.plot(val_epochs, val_vals, color=RED, lw=2, marker="o", ms=4, label="Val worst-class")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Worst-class Accuracy over Training")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_confusion_matrix(true_labels, pred_labels, out_path):
    classes = ["Male", "Female"]
    matrix = np.zeros((2, 2), dtype=int)
    idx = {"Male": 0, "Female": 1}
    for t, p in zip(true_labels, pred_labels):
        if t in idx and p in idx:
            matrix[idx[t]][idx[p]] += 1

    fig, ax = plt.subplots(figsize=(5, 5))
    # Normalize per row for color intensity
    row_sums = matrix.sum(axis=1, keepdims=True).clip(min=1)
    norm = matrix / row_sums

    for ri in range(2):
        for ci in range(2):
            frac = norm[ri, ci]
            color = BLUE if ri == ci else RED
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            face = (
                min(1.0, (r * frac + 255 * (1 - frac)) / 255),
                min(1.0, (g * frac + 255 * (1 - frac)) / 255),
                min(1.0, (b * frac + 255 * (1 - frac)) / 255),
            )
            ax.add_patch(plt.Rectangle((ci, 1 - ri), 1, 1, color=face))
            count = matrix[ri, ci]
            pct   = norm[ri, ci] * 100
            ax.text(ci + 0.5, 1 - ri + 0.55, str(count),
                    ha="center", va="center", fontsize=18, fontweight="bold")
            ax.text(ci + 0.5, 1 - ri + 0.35, f"{pct:.1f}%",
                    ha="center", va="center", fontsize=12, color="#333333")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Pred Male", "Pred Female"])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actual Female", "Actual Male"])
    ax.set_title("Confusion Matrix")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_confidence_histogram(true_labels, pred_labels, confidences, out_path):
    correct   = [c for t, p, c in zip(true_labels, pred_labels, confidences) if t == p]
    incorrect = [c for t, p, c in zip(true_labels, pred_labels, confidences) if t != p]

    bins = np.linspace(0.5, 1.0, 21)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(correct,   bins=bins, color=GREEN, alpha=0.65, label=f"Correct ({len(correct)})")
    ax.hist(incorrect, bins=bins, color=RED,   alpha=0.65, label=f"Incorrect ({len(incorrect)})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    plots_dir = sys.argv[1]

    log_path  = os.path.join(plots_dir, "training_log.csv")
    pred_path = os.path.join(plots_dir, "predictions.csv")

    if os.path.exists(log_path):
        epochs, train_loss, val_loss, train_acc, val_acc = load_training_log(log_path)
        plot_loss(epochs, train_loss, val_loss,
                  os.path.join(plots_dir, "loss.png"))
        plot_accuracy(epochs, train_acc, val_acc,
                      os.path.join(plots_dir, "accuracy.png"))
    else:
        print(f"No training_log.csv found in {plots_dir}, skipping loss/accuracy plots.")

    if os.path.exists(pred_path):
        true_labels, pred_labels, confidences = load_predictions(pred_path)
        plot_confusion_matrix(true_labels, pred_labels,
                              os.path.join(plots_dir, "confusion_matrix.png"))
        plot_confidence_histogram(true_labels, pred_labels, confidences,
                                  os.path.join(plots_dir, "confidence_histogram.png"))
    else:
        print(f"No predictions.csv found in {plots_dir}, skipping confusion/histogram plots.")


if __name__ == "__main__":
    main()
