"""
F1 overview figures from docs/benchmark_numbers_verified.csv
=============================================================
  docs/fig_f1_comparison.{png,pdf}   F1 per DocTamper split, all methods in the CSV
  docs/fig_params_vs_f1.{png,pdf}    mean F1 over the three splits vs. parameter count

Every value comes from the CSV, which records the source of each row (papers' tables and our
evaluation record). Usage:  python scripts/plot_f1_overview.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "docs", "benchmark_numbers_verified.csv")
OUT = os.path.join(ROOT, "docs")

C_OURS, C_DOC, C_OLD = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DOC_METHODS = {"FFDN", "DTD", "ADCD-Net"}                 # recent document-tampering methods

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                     "axes.edgecolor": INK_2, "xtick.color": INK_2, "ytick.color": INK})


def load():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    for r in rows:
        for k in ("f1_testingset", "f1_fcd", "f1_scd", "mean_f1", "params_M"):
            r[k] = float(r[k])
    return rows


def colour(r):
    return C_OURS if r["method"] == "Ours" else (C_DOC if r["method"] in DOC_METHODS else C_OLD)


def fmt(r, v):
    if r["method"] == "Ours":
        return f"{v:.2f}"
    if r["method"] == "FFDN":          # printed with two decimals (0.86) in its paper
        return f"{v:.0f}"
    return f"{v:.1f}"


def legend_handles():
    return [Patch(color=C_OURS, label="Ours"), Patch(color=C_DOC, label="Recent document-tampering methods"),
            Patch(color=C_OLD, label="Earlier image-forensics methods")]


def f1_comparison(rows):
    splits = [("f1_testingset", "TestingSet (30,000)"), ("f1_fcd", "FCD (2,000)"), ("f1_scd", "SCD (18,000)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), dpi=200, sharey=True)
    y = list(range(len(rows)))[::-1]
    for ax, (key, title) in zip(axes, splits):
        for yi, r in zip(y, rows):
            v = r[key]
            ax.barh(yi, v, height=0.55, color=colour(r), zorder=2)
            ax.text(v + 1.5, yi, fmt(r, v), va="center", fontsize=8.5,
                    color=INK if r["method"] == "Ours" else INK_2,
                    fontweight="bold" if r["method"] == "Ours" else "normal")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.set_xlim(0, 104)
        ax.set_xlabel("F1 (%)", fontsize=9)
        ax.xaxis.grid(True, color=GRID, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r["label"] for r in rows], fontsize=9)
    for lbl, r in zip(axes[0].get_yticklabels(), rows):
        if r["method"] == "Ours":
            lbl.set_fontweight("bold")
    fig.legend(handles=legend_handles(), loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.55, 1.02))
    fig.text(0.01, 0.01, "DTD protocol (grayscale, 1-3 JPEG rounds, official per-image qualities, min. quality 75). "
             "Sources per row in docs/benchmark_numbers_verified.csv. Ours: one evaluation on all 50,000 test images.",
             fontsize=6.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_f1_comparison.{ext}"))
    plt.close(fig)


def params_vs_f1(rows):
    fig, ax = plt.subplots(figsize=(6.75, 4.4), dpi=200)
    offsets = {"FFDN": (-6, 2, "right"), "DTD": (5, 1.5, "left"), "ADCD-Net": (0, 3, "center"),
               "Ours": (5, -4.5, "left"), "CAT-Net+CLTD": (0, -5.5, "center"), "TruFor": (5, -3.5, "left"),
               "MVSS-Net": (-8, 0.5, "right")}
    for r in rows:
        is_ours = r["method"] == "Ours"
        ax.scatter(r["params_M"], r["mean_f1"], s=110 if is_ours else 55, color=colour(r), zorder=3,
                   edgecolors="white", linewidths=1.2)
        dx, dy, ha = offsets.get(r["method"], (5, 1, "left"))
        name = r["label"].replace(" (ours)", " (ours)")
        ax.text(r["params_M"] + dx, r["mean_f1"] + dy,
                f"{name}\n{r['mean_f1']:.1f} F1 · {r['params_M']:.1f}M".replace(".0M", "M"),
                ha=ha, va="center", fontsize=8.2 if not is_ours else 8.6,
                color=INK if is_ours else INK_2, fontweight="bold" if is_ours else "normal")
    ax.set_xlim(0, 160)
    ax.set_ylim(35, 92)
    ax.set_xlabel("Parameters (millions)", fontsize=10)
    ax.set_ylabel("Mean F1 over TestingSet, FCD, SCD (%)", fontsize=10)
    ax.set_title("Accuracy vs. model size (DTD protocol, min. quality 75)", loc="left", fontsize=11,
                 fontweight="bold", color=INK)
    ax.grid(True, color=GRID, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=legend_handles(), loc="lower left", frameon=False, fontsize=8.5)
    fig.text(0.01, 0.01, "PS-Net omitted: no parameter count published. Mean F1 = plain mean of the three F1 values.",
             fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_params_vs_f1.{ext}"))
    plt.close(fig)


if __name__ == "__main__":
    data = load()
    f1_comparison(data)
    params_vs_f1(data)
    print("Saved docs/fig_f1_comparison.{png,pdf} and docs/fig_params_vs_f1.{png,pdf}")
