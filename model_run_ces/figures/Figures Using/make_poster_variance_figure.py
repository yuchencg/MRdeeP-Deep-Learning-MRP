"""
Poster figure: per-state model variance across the three sample sizes,
styled to match the UC San Diego capstone poster (variable: climate_change_problem).

Squarer side-by-side layout (3 panels) chosen to fit the poster better than the
original wide letterbox. Standalone — produces exactly one PNG next to this script.
Reads estimates from the MAIN project folder via an absolute ROOT (never a worktree
copy), per the project's data-location guardrail.

Output: per_state_variance_climate_change_problem.png (this folder)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd

# ---------- Absolute paths (main folder, not worktree) ----------
PROJECT_ROOT = Path("/Users/kaeleyoshea/Capstone/A.MRdeeP-Deep-Learning--MRP")
MODEL_ROOT = PROJECT_ROOT / "model_run_ces"
OUT_DIR = MODEL_ROOT / "figures" / "Figures Using"
OUT_PNG = OUT_DIR / "per_state_variance_climate_change_problem.png"

SAMPLES = [1, 2, 3]
OUTCOME = "climate_problem"                  # data-file key (unchanged on disk)
OUTCOME_TITLE = "climate_change_problem"     # display name in title
N_EXTREME = 3

MODEL_COLS = ["glm_mrp", "glmer_mrp", "srp", "glmerstan"]
BASELINE_COL = "baseline"

# ---------- UC San Diego palette ----------
UCSD_NAVY = "#182B49"
UCSD_BLUE = "#00629B"     # California blue
UCSD_GOLD = "#C69214"     # darker gold for contrast on white
UCSD_TEAL = "#006A5C"
GRID = "#EDEDED"
LABEL_GRAY = "#333333"

MODEL_STYLE = {
    "glm_mrp":   {"marker": "o", "color": UCSD_NAVY},
    "glmer_mrp": {"marker": "s", "color": UCSD_BLUE},
    "srp":       {"marker": "^", "color": UCSD_GOLD},
    "glmerstan": {"marker": "P", "color": UCSD_TEAL},
}
BASELINE_STYLE = {"marker": "x", "color": "#8A8A8A"}

X_AXIS_LABEL = "Estimated proportion affirming climate change is a problem"

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


def load_all_samples() -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for s in SAMPLES:
        csv = (MODEL_ROOT / f"sample{s}_state" / "outputs" / "estimates"
               / f"{OUTCOME}_state_estimates.csv")
        if not csv.exists():
            raise FileNotFoundError(f"Missing estimates file: {csv}")
        out[s] = pd.read_csv(csv)
    return out


def state_order(samples) -> list[str]:
    """Rank by sample-3 spread; bottom = 3 lowest variance, top = 3 highest."""
    s3 = samples[3].copy()
    s3["_spread"] = s3[MODEL_COLS].max(axis=1) - s3[MODEL_COLS].min(axis=1)
    ranked = s3.sort_values("_spread")["state_name"].tolist()
    return ranked[:N_EXTREME] + ranked[-N_EXTREME:]


def draw_panel(ax, df, order, sample_n):
    df_ord = df.set_index("state_name").loc[order].reset_index()
    ys = list(range(len(order)))
    ax.set_facecolor("white")

    for y, (_, row) in zip(ys, df_ord.iterrows()):
        vmin, vmax = row[MODEL_COLS].min(), row[MODEL_COLS].max()
        ax.plot([vmin, vmax], [y, y], color="#B9B9B9", linewidth=1.6, zorder=1)
        ax.text((vmin + vmax) / 2, y + 0.30, f"{(vmax - vmin) * 100:.0f} pp",
                ha="center", va="bottom", fontsize=8.5, color=LABEL_GRAY, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.85))

    for model in MODEL_COLS:
        st = MODEL_STYLE[model]
        ax.scatter(df_ord[model], ys, marker=st["marker"], color=st["color"],
                   s=46, zorder=3, label=model, edgecolor="white", linewidth=0.5)
    ax.scatter(df_ord[BASELINE_COL], ys, marker=BASELINE_STYLE["marker"],
               color=BASELINE_STYLE["color"], s=50, zorder=2, label="baseline",
               linewidth=1.6)

    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8.5,
                       color=LABEL_GRAY)
    ax.set_ylim(-0.7, len(order) - 0.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#C9C9C9")
    ax.tick_params(left=False, colors=LABEL_GRAY)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    n_total = int(df["n_respondents"].sum())
    ax.set_title(f"sample {sample_n}  ·  n = {n_total}", fontsize=11,
                 loc="left", color=UCSD_NAVY, fontweight="bold")


def build_figure(samples, order):
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.2), sharey=True, sharex=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.15, right=0.99, top=0.84, bottom=0.16, wspace=0.07)

    for ax, s in zip(axes, SAMPLES):
        draw_panel(ax, samples[s], order, s)

    # State names on leftmost panel
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=9, color=LABEL_GRAY)

    # Grouped side-bar labels: highest near top, lowest near bottom.
    # x sits well left of the (long) state names so the top label clears
    # "District of Columbia"; y in data coords at each group's center.
    trans = mtransforms.blended_transform_factory(axes[0].transAxes,
                                                  axes[0].transData)
    top_center = len(order) - (N_EXTREME / 2) - 0.5      # center of top group
    bot_center = (N_EXTREME / 2) - 0.5                   # center of bottom group
    axes[0].text(-0.46, top_center, f"{N_EXTREME} highest variance", rotation=90,
                 ha="center", va="center", transform=trans, clip_on=False,
                 color=UCSD_NAVY, fontsize=10.5, fontweight="bold")
    axes[0].text(-0.46, bot_center, f"{N_EXTREME} lowest variance", rotation=90,
                 ha="center", va="center", transform=trans, clip_on=False,
                 color=UCSD_NAVY, fontsize=10.5, fontweight="bold")

    # X-axis label under the middle panel only
    axes[1].set_xlabel(X_AXIS_LABEL, fontsize=12, color=LABEL_GRAY, labelpad=10)

    # Centered title + lighter subtitle
    fig.text(0.5, 0.95,
             f"Per-state model variance across sample sizes — {OUTCOME_TITLE}",
             ha="center", va="center", fontsize=15, fontweight="bold",
             color=UCSD_NAVY)
    fig.text(0.5, 0.905,
             "Line spans min/max across models (percentage points); "
             "baseline shown as ×",
             ha="center", va="center", fontsize=10, fontweight="normal",
             color="#555555")

    # Shared legend at the bottom
    handles, labels = axes[-1].get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    leg = fig.legend([h for h, _ in uniq], [l for _, l in uniq],
                     loc="lower center", ncol=len(uniq), frameon=False,
                     fontsize=10.5, bbox_to_anchor=(0.5, 0.0))
    for t in leg.get_texts():
        t.set_color(LABEL_GRAY)
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_all_samples()
    order = state_order(samples)
    fig = build_figure(samples, order)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", OUT_PNG)


if __name__ == "__main__":
    main()
