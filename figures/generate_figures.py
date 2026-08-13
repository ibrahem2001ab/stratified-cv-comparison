"""Generate the two vector figures used by the JMLR paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
RESULTS = OUT.parent / "results" / "reported_gap_results.csv"

DATASETS = [
    "Blood Transfusion", "Adult", "QSAR Biodegradation", "Spambase", "Iris",
    "German Credit", "KC1", "PC1", "ILPD", "Diabetes", "Credit Approval",
    "Mammography", "Phoneme",
]
MINORITY = [23.8, 23.9, 33.7, 39.4, 33.3, 30.0, 15.5, 6.9, 28.6, 34.9, 44.5, 2.3, 29.3]

GAP_REGULAR = {
    "Adult": 0.0318,
    "Blood Transfusion": -0.0450,
    "Credit Approval": 0.0572,
    "Diabetes": 0.0542,
    "German Credit": 0.0594,
    "ILPD": -0.0161,
    "Iris": 0.0119,
    "KC1": -0.0275,
    "Mammography": -0.0370,
    "PC1": -0.0896,
    "Phoneme": -0.1615,
    "QSAR Biodegradation": 0.0717,
    "Spambase": 0.0110,
}
GAP_STRATIFIED = {
    "Adult": 0.0319,
    "Blood Transfusion": -0.0424,
    "Credit Approval": 0.0504,
    "Diabetes": 0.0632,
    "German Credit": 0.0508,
    "ILPD": -0.0176,
    "Iris": 0.0075,
    "KC1": -0.0332,
    "Mammography": -0.0265,
    "PC1": -0.0753,
    "Phoneme": -0.1625,
    "QSAR Biodegradation": 0.0579,
    "Spambase": 0.0124,
}


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="both", color="#d6dde7", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def dataset_imbalance():
    order = np.argsort(MINORITY)
    labels = np.array(DATASETS)[order]
    values = np.array(MINORITY)[order]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    colors = ["#1f77b4" if v >= 10 else "#d95f02" for v in values]
    ax.barh(labels, values, color=colors, height=0.68)
    ax.axvline(50, color="#566573", linestyle="--", linewidth=1, label="50% binary balance")
    for y, value in enumerate(values):
        ax.text(value + 0.7, y, f"{value:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, 53)
    ax.set_xlabel("Smallest class share (%)")
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "dataset_imbalance.pdf", bbox_inches="tight")
    plt.close(fig)


def algorithm_gaps():
    if RESULTS.exists():
        import pandas as pd

        reported = pd.read_csv(RESULTS)
        labels = reported["dataset"].tolist()
        x = reported["regular_gap"].to_numpy()
        y = reported["stratified_gap"].to_numpy()
    else:
        labels = list(GAP_REGULAR)
        x = np.array([GAP_REGULAR[k] for k in labels])
        y = np.array([GAP_STRATIFIED[k] for k in labels])
    lo, hi = -0.175, 0.085
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    ax.plot([lo, hi], [lo, hi], color="#9aa5b1", linewidth=1.1, label="unchanged gap")
    ax.axhline(0, color="#566573", linewidth=0.9)
    ax.axvline(0, color="#566573", linewidth=0.9)
    ax.scatter(x, y, s=34, color="#1f77b4", edgecolor="white", linewidth=0.6, zorder=3)
    offsets = {"Phoneme": (5, -2), "PC1": (5, 3), "QSAR Biodegradation": (-76, 2),
               "Mammography": (5, 2), "Adult": (5, -8), "Spambase": (5, 4)}
    for name, xi, yi in zip(labels, x, y):
        if name in offsets:
            ax.annotate(name, (xi, yi), xytext=offsets[name], textcoords="offset points", fontsize=7)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("LR - DT gap under Regular K-Fold")
    ax.set_ylabel("LR - DT gap under Stratified K-Fold")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "algorithm_gaps.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    dataset_imbalance()
    algorithm_gaps()
