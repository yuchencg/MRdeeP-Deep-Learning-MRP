"""
Poster figure: the two "where does it come from?" scatters tiled side by side,
styled to match the UC San Diego capstone poster (variable: climate_change_problem).

  LEFT  — DISAGREEMENT: model spread vs sample spread (no ground truth needed).
          How much do the answers move when you swap models vs. add data?
  RIGHT — ACCURACY:     model-choice vs sample-size effect on ERROR, measured
          against the full-sample weighted benchmark. How much does each move
          you closer to / further from the truth?

Both panels share the identical layout, the y = x diagonal, and a SHARED axis
scale so the two views are directly comparable. Reuses the per-state computations
from the two standalone scripts, so there is a single source of truth and the
data-location guardrail (absolute ROOT, never a worktree copy) is inherited.

Output: sources_sidebyside_climate_change_problem.png (this folder)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------- Absolute paths (main folder, not worktree) ----------
PROJECT_ROOT = Path("/Users/kaeleyoshea/Capstone/A.MRdeeP-Deep-Learning--MRP")
MODEL_ROOT = PROJECT_ROOT / "model_run_ces"
OUT_DIR = MODEL_ROOT / "figures" / "Figures Using"
OUT_PNG = OUT_DIR / "sources_sidebyside_climate_change_problem.png"

# Reuse the per-state computations from the two standalone scripts.
sys.path.insert(0, str(OUT_DIR))
from make_poster_uncertainty_sources_figure import (  # noqa: E402
    compute_uncertainty_sources, STATE_ABBR, X_SAMPLE, N_MODELS, OUTCOME_TITLE,
)
from make_poster_error_sources_figure import (  # noqa: E402
    compute_error_sources,
)

# ---------- UC San Diego palette ----------
UCSD_NAVY = "#182B49"
UCSD_BLUE = "#00629B"     # California blue  -> sample size
UCSD_GOLD = "#C69214"     # darker gold      -> model choice
GRID = "#EDEDED"
LABEL_GRAY = "#333333"
SUBTITLE_GRAY = "#555555"

N_LABEL = 8  # highest-total states to label per panel

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


def draw_panel(ax, x, y, state_names, lim, *, xlabel, ylabel, panel_title,
               region_blue, region_gold, summary):
    """Draw one model-vs-sample scatter into `ax` using the shared style."""
    ax.set_facecolor("white")

    # Shaded half-planes
    ax.fill_between([0, lim], [0, lim], [lim, lim],
                    color=UCSD_BLUE, alpha=0.06, zorder=0)
    ax.fill_between([0, lim], [0, 0], [0, lim],
                    color=UCSD_GOLD, alpha=0.07, zorder=0)

    # y = x reference
    ax.plot([0, lim], [0, lim], linestyle="--", color="#9A9A9A", linewidth=1.3,
            zorder=1)

    # Points colored by dominant source, sized by total
    total = x + y
    above = y > x
    ax.scatter(x[~above], y[~above], s=50 + total[~above] * 5,
               color=UCSD_GOLD, alpha=0.85, edgecolor="white", linewidth=0.8,
               zorder=3)
    ax.scatter(x[above], y[above], s=50 + total[above] * 5,
               color=UCSD_BLUE, alpha=0.85, edgecolor="white", linewidth=0.8,
               zorder=3)

    # Label top-N states by total
    for i in np.argsort(total)[::-1][:N_LABEL]:
        abbr = STATE_ABBR.get(state_names[i], state_names[i])
        ax.annotate(abbr, xy=(x[i], y[i]), xytext=(6, 4),
                    textcoords="offset points", fontsize=9.5, color=UCSD_NAVY,
                    fontweight="bold", zorder=5)

    # Half-plane labels
    ax.text(lim * 0.80, lim * 0.93, region_blue, fontsize=11, color=UCSD_BLUE,
            ha="center", va="center", fontweight="bold", linespacing=1.3, zorder=3)
    ax.text(lim * 0.80, lim * 0.30, region_gold, fontsize=11, color=UCSD_GOLD,
            ha="center", va="center", fontweight="bold", linespacing=1.3, zorder=3)

    # Axes
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel, fontsize=11, color=LABEL_GRAY, labelpad=9)
    ax.set_ylabel(ylabel, fontsize=11, color=LABEL_GRAY, labelpad=9)
    ax.set_xticks(range(0, int(lim) + 1, 5))
    ax.set_yticks(range(0, int(lim) + 1, 5))
    ax.tick_params(colors=LABEL_GRAY, labelsize=9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#C9C9C9")
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Panel title (bold navy, left-aligned)
    ax.set_title(panel_title, fontsize=12.5, loc="left", color=UCSD_NAVY,
                 fontweight="bold", pad=10)

    # Summary box (top-left)
    ax.text(0.03, 0.985, summary, transform=ax.transAxes, fontsize=9,
            va="top", ha="left", color=LABEL_GRAY,
            bbox=dict(facecolor="white", edgecolor="#CFCFCF",
                      boxstyle="round,pad=0.45"))


def build_figure():
    # ----- Left panel: disagreement (spread) -----
    df_u, totals_u = compute_uncertainty_sources()
    xu = df_u["model_spread_pp"].values
    yu = df_u["sample_spread_pp"].values
    states_u = df_u["state_name"].tolist()
    x_n = totals_u[X_SAMPLE]
    x_k = round(x_n, -3) // 1000

    # ----- Right panel: accuracy (error vs benchmark) -----
    df_e, totals_e, bench_n = compute_error_sources()
    xe = df_e["model_err_spread_pp"].values
    ye = df_e["sample_err_spread_pp"].values
    states_e = df_e["state_name"].tolist()

    # Shared axis scale so the two views are directly comparable.
    lim = max(xu.max(), yu.max(), xe.max(), ye.max()) * 1.12

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(17, 9))
    fig.patch.set_facecolor("white")

    draw_panel(
        axL, xu, yu, states_u, lim,
        xlabel=(f"Model-choice spread  (pp)\n"
                f"max − min across {N_MODELS} MRP models, at sample {X_SAMPLE} "
                f"(≈{x_k:.0f}k, n = {x_n:,})"),
        ylabel=("Sample-size spread  (pp)\n"
                "max − min of mean-across-models, across samples 1 / 2 / 3"),
        panel_title="A.  Disagreement — how far apart the estimates are",
        region_blue="SAMPLE SIZE\ncreates more spread",
        region_gold="MODEL CHOICE\ncreates more spread",
        summary=(f"Mean spread:  model {xu.mean():.1f} pp, sample {yu.mean():.1f} pp\n"
                 f"Sample size creates more spread in {int((yu > xu).sum())} / {len(df_u)} states"),
    )

    draw_panel(
        axR, xe, ye, states_e, lim,
        xlabel=(f"Model-choice effect on error  (pp)\n"
                f"max − min across {N_MODELS} MRP models, at sample {X_SAMPLE} "
                f"(≈{x_k:.0f}k, n = {x_n:,})"),
        ylabel=("Sample-size effect on error  (pp)\n"
                "max − min of mean-across-models error, across samples 1 / 2 / 3"),
        panel_title="B.  Accuracy — how far the estimates miss the benchmark",
        region_blue="SAMPLE SIZE\nchanges accuracy more",
        region_gold="MODEL CHOICE\nchanges accuracy more",
        summary=(f"Mean effect:  model {xe.mean():.1f} pp, sample {ye.mean():.1f} pp\n"
                 f"Sample size changes accuracy more in {int((ye > xe).sum())} / {len(df_e)} states"),
    )

    # Shared legend (bottom center)
    handles = [
        plt.Line2D([0], [0], linestyle="--", color="#9A9A9A", linewidth=1.3,
                   label="y = x  (model choice & sample size contribute equally)"),
        plt.Line2D([0], [0], marker="o", linestyle="none", markersize=10,
                   markerfacecolor=UCSD_BLUE, markeredgecolor="white",
                   label="sample size is the bigger source (above the line)"),
        plt.Line2D([0], [0], marker="o", linestyle="none", markersize=10,
                   markerfacecolor=UCSD_GOLD, markeredgecolor="white",
                   label="model choice is the bigger source (below the line)"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                     fontsize=10.5, bbox_to_anchor=(0.5, 0.015))
    for t in leg.get_texts():
        t.set_color(LABEL_GRAY)

    # Title + subtitle (poster style)
    fig.text(0.5, 0.965,
             f"Where does each state's uncertainty come from? — {OUTCOME_TITLE}",
             ha="center", va="center", fontsize=16.5, fontweight="bold",
             color=UCSD_NAVY)
    fig.text(0.5, 0.930,
             "Two views of the same question: how far apart the estimates are (left) "
             "and how far they miss the full-sample weighted benchmark "
             f"(n = {bench_n:,}, right)  ·  each dot = one state (51, incl. DC)",
             ha="center", va="center", fontsize=11, color=SUBTITLE_GRAY)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13, wspace=0.18)
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", OUT_PNG)


if __name__ == "__main__":
    main()
