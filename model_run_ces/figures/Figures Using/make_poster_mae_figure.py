"""
Poster figure: mean absolute difference between each sample's per-model state
estimates and the all-sample benchmark, styled to match the UC San Diego capstone
poster (variable: climate_change_problem). This is FIG 1 on the poster.

Drops the Naive baseline and MRdeeP panels — keeps the four MRP architectures
(GLM-MRP, GLMER-MRP, GLMER-Stan, SRP) in a 2x2 grid. Standalone — produces exactly
one PNG next to this script. Reads estimates from the MAIN project folder via an
absolute ROOT (never a worktree copy), per the project's data-location guardrail.

Output: mae_vs_all_sample_climate_change_problem.png (this folder)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------- Absolute paths (main folder, not worktree) ----------
PROJECT_ROOT = Path("/Users/kaeleyoshea/Capstone/A.MRdeeP-Deep-Learning--MRP")
MODEL_ROOT = PROJECT_ROOT / "model_run_ces"
OUT_DIR = MODEL_ROOT / "figures" / "Figures Using"
OUT_PNG = OUT_DIR / "mae_vs_all_sample_climate_change_problem.png"

SAMPLES = [1, 2, 3]
OUTCOME = "climate_problem"                  # data-file key (unchanged on disk)
OUTCOME_TITLE = "climate_change_problem"     # display name in title

# Baseline and MRdeeP intentionally removed for the poster.
MODELS = ["glm_mrp", "glmer_mrp", "glmerstan", "srp"]
MODEL_LABELS = {
    "glm_mrp":   "GLM-MRP",
    "glmer_mrp": "GLMER-MRP",
    "glmerstan": "GLMER-Stan",
    "srp":       "SRP",
}

# ---------- UC San Diego palette ----------
UCSD_NAVY = "#182B49"
UCSD_BLUE = "#00629B"     # California blue
UCSD_GOLD = "#C69214"     # darker gold for contrast on white
GRID = "#EDEDED"
LABEL_GRAY = "#333333"
MUTED_GRAY = "#8A8A8A"    # worst-performer badge

# Light -> dark sample ramp drawn from the UC San Diego blues.
SAMPLE_COLORS = ["#A3C7DE", "#00629B", "#182B49"]

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


def load_benchmark() -> pd.DataFrame:
    path = MODEL_ROOT / "all_sample" / "outputs" / f"{OUTCOME}_state_estimates.csv"
    df = pd.read_csv(path, dtype={"state_fips": str})
    return df[["state_fips", "estimate"]].rename(columns={"estimate": "benchmark"})


def load_sample_estimates(sample_num: int) -> tuple[pd.DataFrame, int]:
    """Build a state x model DataFrame from whichever files exist for this sample."""
    est_dir = MODEL_ROOT / f"sample{sample_num}_state" / "outputs" / "estimates"

    combined: dict[str, pd.Series] = {}
    total_n: int | None = None

    agg_path = est_dir / f"{OUTCOME}_state_estimates.csv"
    if agg_path.exists():
        agg = pd.read_csv(agg_path, dtype={"state_fips": str}).set_index("state_fips")
        if "n_respondents" in agg.columns:
            total_n = int(agg["n_respondents"].sum())
        for m in MODELS:
            if m in agg.columns:
                combined[m] = agg[m]

    for m in MODELS:
        if m in combined:
            continue
        per_path = est_dir / f"{m}_{OUTCOME}_state_estimates.csv"
        if per_path.exists():
            per = pd.read_csv(per_path, dtype={"state_fips": str}).set_index("state_fips")
            if "estimate" in per.columns:
                combined[m] = per["estimate"]
                if total_n is None and "n_respondents" in per.columns:
                    total_n = int(per["n_respondents"].sum())

    est_df = pd.DataFrame(combined)
    return est_df, (total_n if total_n is not None else 0)


def compute_mae_table() -> tuple[pd.DataFrame, dict[int, int]]:
    """MAE table indexed by model, columns = sample numbers; plus sample -> n total."""
    bench = load_benchmark().set_index("state_fips")
    rows = []
    sample_n: dict[int, int] = {}
    for s in SAMPLES:
        est_df, n_total = load_sample_estimates(s)
        sample_n[s] = n_total
        joined = est_df.join(bench, how="inner")
        for m in MODELS:
            mae = (joined[m] - joined["benchmark"]).abs().mean() if m in joined.columns else float("nan")
            rows.append({"model": m, "sample": s, "mae": mae})
    mae_df = pd.DataFrame(rows).pivot(index="model", columns="sample", values="mae").reindex(MODELS)
    return mae_df, sample_n


def draw_panel(ax, mae_pp, model, sample_labels, ymax, badge=None):
    """Values in `mae_pp` are already in percentage points. `badge` is an optional
    (text, color) tuple shown in the panel's top-right corner."""
    ax.set_facecolor("white")
    values = [mae_pp.loc[model, s] for s in SAMPLES]
    plot_vals = [0 if pd.isna(v) else v for v in values]

    bars = ax.bar(range(len(SAMPLES)), plot_vals, color=SAMPLE_COLORS,
                  edgecolor="white", linewidth=0.8, width=0.66, zorder=3)

    for bar, v in zip(bars, values):
        if pd.isna(v):
            ax.text(bar.get_x() + bar.get_width() / 2, ymax * 0.02, "n/a",
                    ha="center", va="bottom", fontsize=9, color="grey", style="italic")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, v + ymax * 0.02, f"{v:.1f} pp",
                    ha="center", va="bottom", fontsize=10, color=LABEL_GRAY,
                    fontweight="bold", zorder=5)

    ax.set_title(MODEL_LABELS[model], fontsize=12.5, loc="left",
                 color=UCSD_NAVY, fontweight="bold", pad=8)

    if badge is not None:
        text, color = badge
        ax.text(1.0, 1.02, text, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, fontweight="bold", color=color, zorder=6)
    ax.set_xticks(range(len(SAMPLES)))
    ax.set_xticklabels(sample_labels, fontsize=9, color=LABEL_GRAY)
    ax.set_ylim(0, ymax)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#C9C9C9")
    ax.tick_params(colors=LABEL_GRAY)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def build_figure(mae_df, sample_n):
    import numpy as np

    # Work in percentage points (proportion x 100) for readability.
    mae_pp = mae_df * 100

    sample_labels = [f"sample {s}\nn = {sample_n[s]:,}" for s in SAMPLES]
    finite = mae_pp.values[np.isfinite(mae_pp.values)]
    ymax = (finite.max() if finite.size else 10) * 1.28

    # Best / worst by mean MAE across the three samples (lower = better).
    avg = mae_pp.mean(axis=1, skipna=True)
    best_model = avg.idxmin()
    worst_model = avg.idxmax()
    badges = {
        best_model:  (r"$\bigstar$ Best overall", UCSD_GOLD),
        worst_model: (r"$\triangle$ Highest difference", MUTED_GRAY),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.83, bottom=0.10,
                        wspace=0.08, hspace=0.32)

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 2, idx % 2]
        draw_panel(ax, mae_pp, model, sample_labels, ymax, badge=badges.get(model))
        if idx % 2 == 0:
            ax.set_ylabel("Mean |estimate − benchmark|  (pp)", fontsize=11,
                          color=LABEL_GRAY)

    bench_n = int(pd.read_csv(
        MODEL_ROOT / "all_sample" / "outputs" / f"{OUTCOME}_state_estimates.csv"
    )["n_respondents"].sum())

    fig.text(0.5, 0.960,
             f"Mean absolute difference from the full-sample benchmark — {OUTCOME_TITLE}",
             ha="center", va="center", fontsize=15, fontweight="bold",
             color=UCSD_NAVY)
    fig.text(0.5, 0.925,
             f"Benchmark = full-sample weighted state estimates (n = {bench_n:,})",
             ha="center", va="center", fontsize=10.5, color="#555555")
    fig.text(0.5, 0.900,
             "Mean |sample estimate − benchmark|, averaged across 51 states (incl. DC)",
             ha="center", va="center", fontsize=10.5, color="#555555")

    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mae_df, sample_n = compute_mae_table()
    fig = build_figure(mae_df, sample_n)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", OUT_PNG)


if __name__ == "__main__":
    main()
