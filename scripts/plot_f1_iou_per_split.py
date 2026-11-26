"""
DocROI-Former-QDCT vs. published document-tampering methods: F1 and IoU per DocTamper split
===========================================================================================
Three figures (TestingSet, FCD, SCD). Each figure: left panel = F1, right panel = IoU.

Previous model (v3 / pguoctv3): reports/dtd_protocol/20261116_181157_dtd_q75/summary.json
      (same official protocol; shows the improvement from v3 to QDCT)
Ours: reports/dtd_protocol/FINAL_RESULT_ALLTHREELOGGED_RESULT/summary.json
      (run 20261119_113438_dtd_q75, qdct_qkn_pgu epoch 11, official DTD protocol, min Q 75)
      F1  = f1_dtd        (per-image mean P and R, then F = 2PR/(P+R), as in DTD's script)
      IoU = iou_tampered  (tampered-class IoU pooled over all pixels of the split, DTD's iu[1])

Baselines:
  F1  - ADCD-Net (Wong et al., ICCV 2025) Table 1 for DTD, MVSS-Net, ADCD-Net
        (same source as docs/benchmark_numbers_verified.csv);
        DTD paper (Qu et al., CVPR 2023) Table 4 for Swin-Uper and ManTra-Net.
  IoU - DTD paper Table 7, Q75 columns. ADCD-Net publishes P/R/F1 only -> omitted from the IoU panel.

Usage:  python scripts/plot_f1_iou_per_split.py      (writes docs/fig_<split>_f1_iou.{png,pdf})
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "reports", "dtd_protocol", "FINAL_RESULT_ALLTHREELOGGED_RESULT", "summary.json")
# Previous model (pguoctv3) re-evaluated under the same official protocol (README section 10).
PREV_SUMMARY = os.path.join(ROOT, "reports", "dtd_protocol", "20261116_181157_dtd_q75", "summary.json")
OUT_DIR = os.path.join(ROOT, "docs")

# Validated categorical slots 1-2 (blue / orange) on the light surface.
C_F1 = "#2a78d6"
C_IOU = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
HIGHLIGHT = "#f1f0ec"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK_2,
})


# Published numbers in %:  {split: {method: (F1, IoU)}};  None = not published by that paper.
PUBLISHED = {
    "TestingSet": {
        "ManTra-Net": (15.3, 18.0),
        "MVSS-Net":   (43.1, 43.0),
        "Swin-Uper":  (63.8, 70.0),
        "DTD":        (79.2, 83.0),
        "ADCD-Net":   (80.6, None),
    },
    "FCD": {
        "ManTra-Net": (20.9, 17.0),
        "MVSS-Net":   (42.4, 41.0),
        "Swin-Uper":  (54.6, 41.0),
        "DTD":        (81.6, 75.0),
        "ADCD-Net":   (81.5, None),
    },
    "SCD": {
        "ManTra-Net": (15.7, 16.0),
        "MVSS-Net":   (41.4, 40.0),
        "Swin-Uper":  (57.4, 51.0),
        "DTD":        (75.4, 69.0),
        "ADCD-Net":   (74.0, None),
    },
}

METHODS = [
    ("ManTra-Net", "ManTra-Net\nCVPR '19"),
    ("MVSS-Net",   "MVSS-Net\nICCV '21"),
    ("Swin-Uper",  "Swin-Uper\nICCV '21"),
    ("DTD",        "DTD\nCVPR '23"),
    ("ADCD-Net",   "ADCD-Net\nICCV '25"),
    ("Prev",       "Ours v3\n(previous)"),
    ("Ours",       "Ours QDCT\n(latest)"),
]

SPLITS = [
    ("TestingSet", "DocTamper-TestingSet (30,000 images)", "testingset"),
    ("FCD",        "DocTamper-FCD (2,000 images, cross-domain)", "fcd"),
    ("SCD",        "DocTamper-SCD (18,000 images, cross-domain)", "scd"),
]


def load_run(path):
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    return {k: (v["f1_dtd"], v["iou_tampered"]) for k, v in s["splits"].items()}, s["run_id"]


def draw_metric(ax, values, color, ylabel, panel_title):
    """values: list of (label, value-or-None, role, delta-or-None); role in base / prev / ours."""
    w = 0.62
    for x, (label, v, role, delta) in enumerate(values):
        if role == "ours":
            ax.axvspan(x - 0.5, x + 0.5, color=HIGHLIGHT, zorder=0, lw=0)
        if role == "prev":
            # Same hue, outlined + hatched: our earlier model, visibly not a baseline paper.
            ax.bar(x, v, w, facecolor=SURFACE, edgecolor=color, hatch="///", linewidth=1.4, zorder=2)
        else:
            ax.bar(x, v, w, color=color, zorder=2, linewidth=0)
        top = f"{v:.1f}"
        ax.text(x, v + 1.2, top, ha="center", va="bottom", fontsize=10, color=INK,
                fontweight="bold" if role == "ours" else "normal")
        if delta is not None:
            sign = "+" if delta >= 0 else "−"
            ax.text(x, v + 6.2, f"{sign}{abs(delta):.1f} vs v3", ha="center", va="bottom",
                    fontsize=8.5, color=INK_2)

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([v[0] for v in values], fontsize=9.5)
    for lbl, v in zip(ax.get_xticklabels(), values):
        if v[2] == "ours":
            lbl.set_fontweight("bold")
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(panel_title, fontsize=12, fontweight="bold", color=INK, loc="left", pad=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_facecolor(SURFACE)


def plot_split(split, title, slug, ours, prev, run_id, prev_run_id):
    f1_vals, iou_vals = [], []
    for m, label in METHODS:
        if m == "Ours":
            (f1, iou), role = ours[split], "ours"
            d_f1, d_iou = f1 - prev[split][0], iou - prev[split][1]
        elif m == "Prev":
            (f1, iou), role = prev[split], "prev"
            d_f1 = d_iou = None
        else:
            (f1, iou), role = PUBLISHED[split][m], "base"
            d_f1 = d_iou = None
        # A method whose paper does not publish a metric is left out of that panel.
        if f1 is not None:
            f1_vals.append((label, f1, role, d_f1))
        if iou is not None:
            iou_vals.append((label, iou, role, d_iou))

    fig, (ax_f1, ax_iou) = plt.subplots(1, 2, figsize=(17, 6.4), dpi=300, facecolor=SURFACE)
    draw_metric(ax_f1, f1_vals, C_F1, "F1 (%)", "F1  (DTD protocol, min. JPEG quality 75)")
    draw_metric(ax_iou, iou_vals, C_IOU, "IoU (%)", "IoU  (tampered class, DTD protocol, Q75)")
    fig.suptitle(title, fontsize=14, fontweight="bold", color=INK, x=0.01, ha="left", y=0.99)

    notes = ("F1: ADCD-Net (Wong et al., ICCV 2025) Table 1 for DTD, MVSS-Net, ADCD-Net; "
             "DTD paper (Qu et al., CVPR 2023) Table 4 for Swin-Uper, ManTra-Net.   "
             "IoU: DTD paper Table 7, Q75.\n"
             "ADCD-Net publishes P/R/F1 only, so it appears in the F1 panel only.   "
             f"Ours: qdct_qkn_pgu epoch 11, single evaluation under the official DTD protocol (run {run_id}).\n"
             f"v3 (previous): pguoctv3 checkpoint under the same protocol (run {prev_run_id}, inference at 384 px); "
             "its earlier 80.7 F1 was measured without recompression and is withdrawn.")
    fig.text(0.01, 0.01, notes, fontsize=8, color=INK_2, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    png = os.path.join(OUT_DIR, f"fig_{slug}_f1_iou.png")
    pdf = os.path.join(OUT_DIR, f"fig_{slug}_f1_iou.pdf")
    fig.savefig(png, dpi=300, facecolor=SURFACE)
    fig.savefig(pdf, facecolor=SURFACE)
    plt.close(fig)
    print("Saved:", png)


def main():
    ours, run_id = load_run(SUMMARY)
    prev, prev_run_id = load_run(PREV_SUMMARY)
    os.makedirs(OUT_DIR, exist_ok=True)
    for split, title, slug in SPLITS:
        plot_split(split, title, slug, ours, prev, run_id, prev_run_id)


if __name__ == "__main__":
    main()
