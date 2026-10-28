"""Make manuscript figures from saved, verified outputs; no models are refitted."""

from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
NAVY, GREY, GOLD = "#244b68", "#6b7379", "#b68c48"


def save(fig, name):
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    with (ROOT / "analysis/results/original_notebook/final_test_comparisons.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["Contact rate"] == "25%"]
    assert len(rows) == 4
    check = json.loads((ROOT / "analysis/results/targeting_checks.json").read_text())
    primary = next(r for r in rows if r["Rule"] == "Incremental revenue")
    assert np.isclose(float(primary["Advantage over random"]), check["independent_checks"]["primary_advantage_per_1000"])
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for i, row in enumerate(rows):
        mean, low, high = [float(row[k]) for k in ["Advantage over random", "95% CI lower", "95% CI upper"]]
        is_primary = row["Rule"] == "Incremental revenue"
        ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt="o", capsize=4,
                    color=NAVY if is_primary else GREY, markersize=6,
                    markerfacecolor=NAVY if is_primary else "white", elinewidth=1.5)
    ax.axvline(0, color="#555555", linewidth=0.9, linestyle="--", zorder=2)
    ax.set_yticks(range(4), [r["Rule"] + ("\n(original primary)" if r["Rule"] == "Incremental revenue" else "") for r in rows])
    ax.invert_yaxis()
    ax.set_ylim(3.6, -.6)
    ax.set_xlim(-220, 600)
    ax.set_xticks([-200, 0, 200, 400, 600])
    ax.set_xlabel("Revenue advantage over expected random targeting\n($ per 1,000 eligible customers)")
    ax.grid(axis="x", color="#e5e8eb", linewidth=.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "revenue_comparisons")

    with (ROOT / "analysis/results/selection_frequencies.csv").open() as f:
        selected = [float(r["selection_frequency"]) for r in csv.DictReader(f) if r["original_selected"] == "True"]
    selected = np.array(selected)
    count = check["independent_checks"]["selected_per_rule"]
    common = check["independent_checks"]["overlap_incremental_historical"]
    unique = count-common
    below_half = int((selected < .5).sum())
    assert len(selected) == count == 2130
    assert below_half == check["new_exploratory_training_stability"]["original_selected_with_frequency_below_half"] == 524
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.3), gridspec_kw={"height_ratios": [1, 1.55]})
    a, b = axes
    a.set_title("A. Shared and rule-specific selections", loc="left", pad=10)
    for y in [0, 1]:
        a.barh(y, common, color=NAVY, height=.48)
        a.barh(y, unique, left=common, color=GOLD if y == 0 else "#c5cacf", height=.48)
        a.text(common/2, y, f"{common:,} shared", ha="center", va="center", color="white", fontsize=9)
        label = "incremental only" if y == 0 else "historical only"
        a.text(common+unique/2, y, f"{unique:,} {label}", ha="center", va="center", color="#222222", fontsize=7.5)
    a.set_yticks([0, 1], ["Incremental revenue", "Historical spending"])
    a.invert_yaxis()
    a.set_xlim(0, count)
    a.set_xticks([0, 1000, 2130], ["0", "1,000", "2,130"])
    a.set_xlabel("Selected customers")
    a.spines["left"].set_visible(False)
    a.tick_params(axis="y", length=0)
    b.set_title("B. Re-selection across 200 training refits", loc="left", pad=10)
    b.hist(selected, bins=np.linspace(0, 1, 21), color=NAVY, edgecolor="white", linewidth=.6)
    heights, _ = np.histogram(selected, bins=np.linspace(0, 1, 21))
    b.set_ylim(0, heights.max()*1.35)
    b.axvline(.5, color=GOLD, linestyle="--", linewidth=1.3)
    b.text(.025, .92, f"{below_half:,} originally selected customers\nre-selected in fewer than half of refits",
           transform=b.transAxes, ha="left", va="top", fontsize=8.5,
           bbox={"facecolor":"white", "edgecolor":"none", "alpha":.92, "pad":3})
    b.set_xlim(0, 1)
    b.xaxis.set_major_formatter(PercentFormatter(1))
    b.set_xlabel("Share of refits selecting a customer")
    b.set_ylabel("Originally selected customers")
    b.grid(axis="y", color="#e5e8eb", linewidth=.7)
    b.set_axisbelow(True)
    fig.tight_layout(h_pad=2.1)
    save(fig, "audience_and_training_sensitivity")
    print("Created two figures in PNG, PDF, and SVG formats; source values checked.")


if __name__ == "__main__":
    main()
