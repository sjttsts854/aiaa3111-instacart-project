"""Rebuild paper figures from frozen CSV evidence; never refit or reselect.

Run with the recorded research environment:
    .venv/Scripts/python.exe closeout/build_latex_figures.py
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "final/latex/figures"
S4 = ROOT / "reports/study4/tables"
FOLLOW = ROOT / "reports/study4_followup/tables"
BLUE, TEAL, RUST = "#3F5875", "#087F78", "#B65037"
GRAY, LIGHT = "#687481", "#E1E6E9"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.labelsize": 10.5, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": GRAY,
    "axes.linewidth": .6, "axes.titleweight": "semibold",
    "axes.titlesize": 11,
})
SOURCES = {}


def load(path):
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def save(fig, name):
    fig.savefig(DEST / f"{name}.pdf", bbox_inches="tight", pad_inches=.07)
    fig.savefig(DEST / f"{name}.png", dpi=400, bbox_inches="tight", pad_inches=.07)
    plt.close(fig)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    hist = load(S4 / "history_sensitivity.csv")
    factor = load(S4 / "factorial_metrics.csv")
    contrast = load(S4 / "primary_contrasts.csv").set_index("contrast")
    depth = load(FOLLOW / "depth_contrasts.csv")

    # Only two distinct supports: min 10 and min 20 produce identical cells.
    hist = hist.loc[hist.min_per_group_per_cell.eq(10)].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    fig.subplots_adjust(left=.36, right=.98, top=.83, bottom=.23)
    for offset, value, lower, upper, color, marker, label in [
        (.13, "raw_common_support_long_minus_short", "raw_common_support_ci95_low", "raw_common_support_ci95_high", BLUE, "o", "Same-support raw"),
        (-.13, "standardized_long_minus_short", "standardized_ci95_low", "standardized_ci95_high", TEAL, "s", "Standardized"),
    ]:
        val = 100 * hist[value].to_numpy()
        err = np.vstack([val - 100 * hist[lower], 100 * hist[upper] - val])
        ax.errorbar(val, np.array([1, 0]) + offset, xerr=err,
                    color=color, marker=marker, markersize=5, linestyle="none",
                    elinewidth=1.6, capsize=3, label=label)
    raw = 100 * hist.iloc[0].raw_overall_long_minus_short
    ax.axvline(raw, color=RUST, lw=1.4, linestyle="--", zorder=0)
    ax.text(raw + .7, .50, f"Overall raw: {raw:.2f} pp", rotation=90,
            ha="left", va="center", color=RUST, fontsize=9)
    ax.axvline(0, color=GRAY, lw=.9, zorder=0)
    ax.set_xlim(-21, 18); ax.set_ylim(-.45, 1.45)
    ax.set_xticks([-20, -10, 0, 10])
    ax.set_yticks([1, 0], [
        "Candidate count\n79.90% retained",
        "Count + historical\nbasket size\n45.82% retained",
    ])
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlabel("Long − short history Recall@10 (pp)")
    ax.legend(loc="lower center", bbox_to_anchor=(.32, 1.02), ncol=2,
              frameon=False, columnspacing=1.2, handletextpad=.5)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=LIGHT, lw=.5, zorder=-1)
    save(fig, "history_composition")

    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.65))
    fig.subplots_adjust(left=.15, right=.98, bottom=.24, top=.82, wspace=.40)
    cmap = LinearSegmentedColormap.from_list("paper_teal", ["#F2F5F5", "#A2D0CB"])
    for ax, method, title in zip(axs, ["frequency", "decay"],
                                 ["Purchase frequency", "Order-decayed frequency (0.8)"]):
        subset = factor.loc[(factor.method == method) & (factor.subgroup == "long_history")]
        vals = np.array([[float(subset.loc[(subset.score_window.astype(str) == window) &
                                            (subset.pool.astype(str) == pool), "recall_at_10"].iloc[0]) * 100
                          for pool in ["5", "all"]] for window in ["5", "all"]])
        # Explicit rectangles keep even the heatmap cells vector in PDF.
        ax.set_xlim(-.5, 1.5); ax.set_ylim(1.5, -.5)
        for i in range(2):
            for j in range(2):
                ax.add_patch(Rectangle((j-.5, i-.5), 1, 1,
                                      facecolor=cmap((vals[i,j]-42)/8),
                                      edgecolor="white", linewidth=2))
                ax.text(j, i, f"{vals[i,j]:.2f}%", ha="center", va="center",
                        fontsize=14, fontweight="semibold", color="#16313E")
        ax.set_xticks([0, 1], ["Recent five", "All history"])
        ax.set_yticks([0, 1], ["Recent five", "All history"])
        ax.tick_params(length=0, pad=7)
        ax.set_xticks([.5], minor=True); ax.set_yticks([.5], minor=True)
        ax.grid(which="minor", color="white", lw=2.5)
        ax.tick_params(which="minor", length=0)
        ax.set_title(title, pad=11, fontsize=10.5)
        for spine in ax.spines.values(): spine.set_visible(False)
    axs[0].set_ylabel("Scoring history", labelpad=9)
    # Both panels share row definitions; repeated right-panel labels crowd
    # the inter-panel gap at the intended paper width.
    axs[1].set_yticklabels([])
    fig.text(.55, .04, "Eligible candidate pool", ha="center", fontsize=10.5)
    save(fig, "factorial_memory")

    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    fig.subplots_adjust(left=.34, right=.98, top=.89, bottom=.20)
    labels = ["Selected − full pool", "Selected − same-budget\nfrequency", "Selected − same-budget\nrecency"]
    for i, key in enumerate(["dual_vs_full", "dual_vs_frequency_budget", "dual_vs_recency_budget"]):
        row = contrast.loc[key]
        delta, low, high = 100 * row.delta, 100 * row.sim_low, 100 * row.sim_high
        y = 2 - i
        ax.errorbar(delta, y, xerr=[[delta-low], [high-delta]], color=TEAL,
                    marker="o", markersize=5, elinewidth=1.8, capsize=3)
        ax.text(delta, y-.25, f"{delta:+.3f} [{low:+.3f}, {high:+.3f}]", fontsize=9.0,
                ha="center", va="top", color=TEAL,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.vlines(-.5, -.53, 2.49, color=RUST, linestyle="--", lw=1.4)
    ax.text(-.5, 2.57, "−0.5 pp margin*", ha="center", color=RUST, fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.axvline(0, color=GRAY, lw=.9)
    ax.set_xlim(-.57, .41); ax.set_ylim(-.53, 2.76)
    ax.set_yticks([2, 1, 0], labels); ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xticks([-.5, -.25, 0, .25])
    ax.set_xlabel("Recall@10 difference (pp)")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=LIGHT, lw=.5, zorder=-1)
    fig.text(.34, -.005, "* Noninferiority margin applies to selected versus full.",
             fontsize=8.8, ha="left", color=GRAY)
    save(fig, "policy_contrasts")

    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    fig.subplots_adjust(left=.13, right=.97, top=.86, bottom=.18)
    for comparator, label, color, marker, style in [
        ("full", "Selected − full pool", BLUE, "o", "-"),
        ("recency_budget", "Selected − recency budget", TEAL, "s", "--"),
    ]:
        sub = depth[depth.comparator.eq(comparator)].sort_values("k")
        delta = 100 * sub.delta.to_numpy()
        error = np.vstack([delta - 100 * sub.sim_low, 100 * sub.sim_high - delta])
        ax.errorbar(sub.k, delta, yerr=error, marker=marker, color=color,
                    label=label, capsize=3, lw=1.4, markersize=4.5, linestyle=style)
    ax.axhline(0, color=GRAY, lw=.8, zorder=0)
    ax.axhline(-.5, color=RUST, lw=1.1, ls=":", zorder=0)
    ax.text(30, -.68, "Original −0.5 pp reference", color=RUST, fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.annotate("−1.33 pp", xy=(30, -1.3341), xytext=(33, -1.55), fontsize=9.5, color=BLUE)
    ax.annotate("−4.03 pp", xy=(50, -4.0279), xytext=(43, -4.72), fontsize=9.5, color=BLUE)
    ax.set_xlim(3, 52); ax.set_ylim(-4.9, .8)
    ax.set_xticks([5, 10, 15, 20, 30, 50])
    ax.set_yticks([-4, -3, -2, -1, 0])
    ax.set_xlabel("Recommendation list length K")
    ax.set_ylabel("Recall difference (pp)")
    ax.grid(axis="y", color=LIGHT, lw=.6, zorder=-1)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.015), frameon=False, ncol=2,
              columnspacing=1.2, handletextpad=.5)
    save(fig, "list_depth")

    (DEST / "README.txt").write_text("""Paper figures for the single-column English LaTeX report

Use the PDF versions in LaTeX for vector text and graphics. Matching 400-dpi
PNG versions are supplied for slides, previews and other editors. These are
drawn directly from completed study tables; no model or policy is refitted.
All recall differences use percentage points (pp).

history_composition.pdf / .png
  Source: reports/study4/tables/history_sensitivity.csv
  Rows: candidate_only_min10, candidate_historical_basket_min10.
  Same-support raw and standardized differences with conditional marginal
  95% intervals. Dashed line: raw overall difference. Min-20 rows have the
  same support and estimates as min-10, not additional independent results.
  Inclusion is limited to nonempty-repeat-target users; this is not causal.

factorial_memory.pdf / .png
  Source: reports/study4/tables/factorial_metrics.csv
  Rows: subgroup=long_history; methods frequency and decay; scoring windows
  5/all crossed with candidate pools 5/all. Recall uses 2,526 nonempty users.
  The CSV users column counts all 2,590 long-history users, including 64
  empty-target users; these are not the recall denominator. Color scale is
  shared across panels (42 to 50 percent recall).

policy_contrasts.pdf / .png
  Source: reports/study4/tables/primary_contrasts.csv
  Rows: dual_vs_full, dual_vs_frequency_budget, dual_vs_recency_budget.
  Uses sim_low/sim_high from the ORIGINAL nine-contrast primary family.
  7,472 eligible users; metrics average seeds within user, not model scores.
  The -0.5 pp margin is a noninferiority criterion for selected versus full.

list_depth.pdf / .png
  Source: reports/study4_followup/tables/depth_contrasts.csv
  All 12 rows: selected versus full and versus recency budget at six K values.
  Uses sim_low/sim_high from the separate POSTHOC twelve-contrast family.
  Connected lines interpolate visually between tested K values only.
  The original -0.5 pp reference does not define a newly validated safe K.

Rebuild: .venv/Scripts/python.exe closeout/build_latex_figures.py
CSV SHA256 checksums are recorded in source_hashes.json.
""", encoding="utf-8")
    (DEST / "source_hashes.json").write_text(json.dumps(SOURCES, indent=2), encoding="utf-8")
    print(json.dumps({"figures": 4, "formats": ["pdf", "png"], "output": str(DEST)}, indent=2))


if __name__ == "__main__":
    main()
