"""
Poster figure: where does each state's ERROR come from — model choice or sample
size? The real-data companion to make_poster_schematic_figure.py, styled to match
the UC San Diego capstone poster (variable: climate_change_problem).

Uses the SAME data as the bar graph (make_poster_mae_figure.py): the full-sample
weighted benchmark (= "ground truth") and the four MRP models' per-state estimates
at each sample size. Error = |estimate − benchmark| in percentage points.

One dot per state (51, incl. DC). For each state:
  x-axis = model-choice effect on error (pp)
           = max − min, across the 4 MRP models, of each model's error, at sample 2
             (how much accuracy changes if you swap models, holding data fixed)
  y-axis = sample-size effect on error (pp)
           = max − min, across samples 1 / 2 / 3, of the mean-across-models error
             (how much accuracy changes as you add data, holding the ensemble fixed)

The y = x diagonal splits the plane:
  above  -> sample size changes accuracy more
  below  -> model choice changes accuracy more
  near 0 -> already accurate; neither moves the answer much

Drops MRdeeP and keeps the four MRP architectures, matching the rest of the poster.
Standalone — produces exactly one PNG next to this script. Reads estimates from the
MAIN project folder via an absolute ROOT (never a worktree copy), per the project's
data-location guardrail.

Output: error_sources_climate_change_problem.png (this folder)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- Absolute paths (main folder, not worktree) ----------
PROJECT_ROOT = Path("/Users/kaeleyoshea/Capstone/A.MRdeeP-Deep-Learning--MRP")
MODEL_ROOT = PROJECT_ROOT / "model_run_ces"
OUT_DIR = MODEL_ROOT / "figures" / "Figures Using"
OUT_PNG = OUT_DIR / "error_sources_climate_change_problem.png"

SAMPLES = [1, 2, 3]
X_SAMPLE = 2                                 # model-error spread read at sample 2 (≈3k)
OUTCOME = "climate_problem"                  # data-file key (unchanged on disk)
OUTCOME_TITLE = "climate_change_problem"     # display name in title

# MRdeeP intentionally excluded for the poster — four MRP architectures only.
MODEL_COLS = ["glm_mrp", "glmer_mrp", "srp", "glmerstan"]
N_MODELS = len(MODEL_COLS)

N_LABEL = 8  # how many highest-error states to label

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

# ---------- UC San Diego palette ----------
UCSD_NAVY = "#182B49"
UCSD_BLUE = "#00629B"     # California blue  -> "sample size changes accuracy more"
UCSD_GOLD = "#C69214"     # darker gold      -> "model choice changes accuracy more"
GRID = "#EDEDED"
LABEL_GRAY = "#333333"
SUBTITLE_GRAY = "#555555"

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


def load_benchmark() -> pd.Series:
    """Full-sample weighted estimate per state (= ground truth), by state_name."""
    csv = MODEL_ROOT / "all_sample" / "outputs" / f"{OUTCOME}_state_estimates.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Missing benchmark file: {csv}")
    return pd.read_csv(csv).set_index("state_name")["estimate"]


def compute_error_sources() -> tuple[pd.DataFrame, dict[int, int], int]:
    """Per state: model-choice error spread (at X_SAMPLE) and sample-size error
    spread (across s1/s2/s3), both in percentage points.

    Returns (df, sample_totals, benchmark_n)."""
    truth = load_benchmark()

    per_sample: dict[int, pd.DataFrame] = {}
    sample_totals: dict[int, int] = {}
    for s in SAMPLES:
        csv = (MODEL_ROOT / f"sample{s}_state" / "outputs" / "estimates"
               / f"{OUTCOME}_state_estimates.csv")
        if not csv.exists():
            raise FileNotFoundError(f"Missing estimates file: {csv}")
        df = pd.read_csv(csv)
        missing = [c for c in MODEL_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"sample{s}: missing model columns {missing}")
        sample_totals[s] = int(df["n_respondents"].sum())
        per_sample[s] = df.set_index("state_name")

    states = [s for s in per_sample[X_SAMPLE].index if s in truth.index]
    t = truth.loc[states]

    # Per-model error |estimate - truth| (pp) at each sample.
    err_by_sample: dict[int, pd.DataFrame] = {}
    for s in SAMPLES:
        est = per_sample[s].loc[states, MODEL_COLS]
        err_by_sample[s] = (est.sub(t, axis=0)).abs() * 100

    # x: spread of per-model error across the 4 models, at X_SAMPLE.
    ex = err_by_sample[X_SAMPLE]
    model_err_spread_pp = ex.max(axis=1) - ex.min(axis=1)

    # y: spread of the mean-across-models error, across the three samples.
    mean_err_per_sample = pd.DataFrame(
        {s: err_by_sample[s].mean(axis=1) for s in SAMPLES}, index=states
    )
    sample_err_spread_pp = mean_err_per_sample.max(axis=1) - mean_err_per_sample.min(axis=1)

    out = pd.DataFrame({
        "state_name": states,
        "model_err_spread_pp": model_err_spread_pp.values,
        "sample_err_spread_pp": sample_err_spread_pp.values,
    })

    bench_n = int(pd.read_csv(
        MODEL_ROOT / "all_sample" / "outputs" / f"{OUTCOME}_state_estimates.csv"
    )["n_respondents"].sum())
    return out, sample_totals, bench_n


def build_figure(df: pd.DataFrame, totals: dict[int, int], bench_n: int):
    x_n = totals[X_SAMPLE]
    x_k = round(x_n, -3) // 1000

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = df["model_err_spread_pp"].values
    y = df["sample_err_spread_pp"].values

    lim = max(x.max(), y.max()) * 1.12

    # Shade the two half-planes (light UCSD tints)
    ax.fill_between([0, lim], [0, lim], [lim, lim],
                    color=UCSD_BLUE, alpha=0.06, zorder=0)
    ax.fill_between([0, lim], [0, 0], [0, lim],
                    color=UCSD_GOLD, alpha=0.07, zorder=0)

    # y = x reference line
    ax.plot([0, lim], [0, lim], linestyle="--", color="#9A9A9A", linewidth=1.3,
            label="y = x  (model choice & sample size affect accuracy equally)", zorder=1)

    # Color points by which source dominates; size grows with total effect
    total = x + y
    above = y > x
    ax.scatter(x[~above], y[~above], s=55 + total[~above] * 5,
               color=UCSD_GOLD, alpha=0.85, edgecolor="white", linewidth=0.8,
               label="model choice changes accuracy more", zorder=3)
    ax.scatter(x[above], y[above], s=55 + total[above] * 5,
               color=UCSD_BLUE, alpha=0.85, edgecolor="white", linewidth=0.8,
               label="sample size changes accuracy more", zorder=3)

    # Label the top-N states by total effect on error
    top_idx = np.argsort(total)[::-1][:N_LABEL]
    for i in top_idx:
        abbr = STATE_ABBR.get(df["state_name"].iloc[i], df["state_name"].iloc[i])
        ax.annotate(abbr, xy=(x[i], y[i]), xytext=(7, 5),
                    textcoords="offset points", fontsize=10, color=UCSD_NAVY,
                    fontweight="bold", zorder=5)

    # Half-plane labels
    ax.text(lim * 0.80, lim * 0.93, "SAMPLE SIZE\nchanges accuracy more",
            fontsize=12, color=UCSD_BLUE, ha="center", va="center",
            fontweight="bold", linespacing=1.3)
    ax.text(lim * 0.80, lim * 0.31, "MODEL CHOICE\nchanges accuracy more",
            fontsize=12, color=UCSD_GOLD, ha="center", va="center",
            fontweight="bold", linespacing=1.3)

    # Axes
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(
        f"Model-choice effect on error  (pp)\n"
        f"max − min across {N_MODELS} MRP models, at sample {X_SAMPLE} "
        f"(≈{x_k:.0f}k, n = {x_n:,})",
        fontsize=12, color=LABEL_GRAY, labelpad=10)
    ax.set_ylabel(
        "Sample-size effect on error  (pp)\n"
        "max − min of mean-across-models error, across samples 1 / 2 / 3",
        fontsize=12, color=LABEL_GRAY, labelpad=10)
    ax.set_xticks(range(0, int(lim) + 1, 5))
    ax.set_yticks(range(0, int(lim) + 1, 5))
    ax.tick_params(colors=LABEL_GRAY, labelsize=10)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#C9C9C9")
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Summary box (top-left)
    n_sample = int(above.sum())
    summary = (
        f"Mean effect:  model {x.mean():.1f} pp, sample {y.mean():.1f} pp\n"
        f"Sample size changes accuracy more in {n_sample} / {len(df)} states"
    )
    ax.text(0.025, 0.985, summary, transform=ax.transAxes,
            fontsize=9.5, va="top", ha="left", color=LABEL_GRAY,
            bbox=dict(facecolor="white", edgecolor="#CFCFCF",
                      boxstyle="round,pad=0.5"))

    # Legend (lower right), clean frame
    leg = ax.legend(loc="lower right", frameon=True, framealpha=0.95,
                    fontsize=9.5, edgecolor="#CFCFCF")
    for t in leg.get_texts():
        t.set_color(LABEL_GRAY)

    # Title + subtitle (poster style)
    fig.text(0.5, 0.965,
             f"Where does each state's error come from? — {OUTCOME_TITLE}",
             ha="center", va="center", fontsize=15, fontweight="bold",
             color=UCSD_NAVY)
    fig.text(0.5, 0.935,
             f"Error = |estimate − full-sample weighted benchmark (n = {bench_n:,})|  ·  "
             "each dot = one state (51, incl. DC)",
             ha="center", va="center", fontsize=10.5, color=SUBTITLE_GRAY)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.11)
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, totals, bench_n = compute_error_sources()
    fig = build_figure(df, totals, bench_n)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", OUT_PNG)


if __name__ == "__main__":
    main()
