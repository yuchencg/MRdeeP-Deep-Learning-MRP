"""
Mean absolute difference between each sample's per-model state estimates and the
all-sample benchmark (weighted mean across all CES respondents).

For each outcome (climate_problem, renewable_fuel):
  - Loads the all-sample benchmark (one estimate per state)
  - Loads sample 1 / 2 / 3 state estimates (one column per model)
  - For each (model, sample) computes mean_{states} |estimate - benchmark|
  - Plots a 3x2 grid of tiles: one tile per model, three bars per tile

Inputs:
  model_run_ces/all_sample/outputs/{outcome}_state_estimates.csv
  model_run_ces/sample{1,2,3}_state/outputs/estimates/{outcome}_state_estimates.csv

Outputs:
  model_run_ces/figures/{outcome}/mae_vs_all_sample.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------- Configuration ----------

ROOT = Path(__file__).resolve().parents[1]  # model_run_ces/
FIG_DIR = ROOT / "figures"

OUTCOMES = ["climate_problem", "renewable_fuel"]
SAMPLES = [1, 2, 3]

MODELS = ["baseline", "glm_mrp", "glmer_mrp", "glmerstan", "srp", "mrdeep"]
MODEL_LABELS = {
    "baseline":  "1. Naive baseline",
    "glm_mrp":   "2. GLM-MRP",
    "glmer_mrp": "3. GLMER-MRP",
    "glmerstan": "4. GLMER-Stan",
    "srp":       "5. SRP",
    "mrdeep":    "6. MRdeeP",
}

OUTCOME_TITLES = {
    "climate_problem": "Climate change is a serious problem",
    "renewable_fuel":  "Support for renewable fuel",
}

SAMPLE_COLORS = ["#bcd9e8", "#5fa8c7", "#1f6b8a"]  # light -> dark blue

# ---------- Helpers ----------

def load_benchmark(outcome: str) -> pd.DataFrame:
    path = ROOT / "all_sample" / "outputs" / f"{outcome}_state_estimates.csv"
    df = pd.read_csv(path, dtype={"state_fips": str})
    return df[["state_fips", "estimate"]].rename(columns={"estimate": "benchmark"})


def load_sample_estimates(outcome: str, sample_num: int) -> tuple[pd.DataFrame, int]:
    """Build a state x model DataFrame from whichever files exist for this (outcome, sample).

    Two layouts are supported:
      1. Aggregated CSV with one column per model: {outcome}_state_estimates.csv
      2. Per-model CSVs: {model}_{outcome}_state_estimates.csv (column 'estimate')

    Returns (state_estimates_df, total_n_respondents) where state_estimates_df is
    indexed by state_fips with one column per model column we managed to load.
    """
    est_dir = ROOT / f"sample{sample_num}_state" / "outputs" / "estimates"

    combined: dict[str, pd.Series] = {}
    total_n: int | None = None

    # 1. Aggregated file (may contain a subset of models)
    agg_path = est_dir / f"{outcome}_state_estimates.csv"
    if agg_path.exists():
        agg = pd.read_csv(agg_path, dtype={"state_fips": str}).set_index("state_fips")
        if "n_respondents" in agg.columns:
            total_n = int(agg["n_respondents"].sum())
        for m in MODELS:
            if m in agg.columns:
                combined[m] = agg[m]

    # 2. Per-model files fill in any gaps
    for m in MODELS:
        if m in combined:
            continue
        per_path = est_dir / f"{m}_{outcome}_state_estimates.csv"
        if per_path.exists():
            per = pd.read_csv(per_path, dtype={"state_fips": str}).set_index("state_fips")
            if "estimate" in per.columns:
                combined[m] = per["estimate"]
                if total_n is None and "n_respondents" in per.columns:
                    total_n = int(per["n_respondents"].sum())

    est_df = pd.DataFrame(combined)
    return est_df, (total_n if total_n is not None else 0)


def compute_mae_table(outcome: str) -> tuple[pd.DataFrame, dict[int, int]]:
    """Return MAE table indexed by model, columns = sample numbers; also a dict of sample -> n_respondents total."""
    bench = load_benchmark(outcome).set_index("state_fips")
    rows = []
    sample_n: dict[int, int] = {}
    for s in SAMPLES:
        est_df, n_total = load_sample_estimates(outcome, s)
        sample_n[s] = n_total
        joined = est_df.join(bench, how="inner")
        for m in MODELS:
            if m in joined.columns:
                mae = (joined[m] - joined["benchmark"]).abs().mean()
            else:
                mae = float("nan")
            rows.append({"model": m, "sample": s, "mae": mae})
    mae_df = pd.DataFrame(rows).pivot(index="model", columns="sample", values="mae").reindex(MODELS)
    return mae_df, sample_n


# ---------- Plotting ----------

def plot_outcome(outcome: str) -> Path:
    mae_df, sample_n = compute_mae_table(outcome)

    sample_labels = [f"Sample {s}\n(n={sample_n[s]:,})" for s in SAMPLES]

    import numpy as np
    finite_vals = mae_df.values[np.isfinite(mae_df.values)]
    ymax = (finite_vals.max() if finite_vals.size else 0.1) * 1.22

    fig, axes = plt.subplots(3, 2, figsize=(10, 11), sharey=True)
    bench_n = pd.read_csv(ROOT / "all_sample" / "outputs" / f"{outcome}_state_estimates.csv")["n_respondents"].sum()
    fig.suptitle(
        f"Mean absolute difference vs. all-sample benchmark — {OUTCOME_TITLES[outcome]}\n"
        f"(averaged across 51 states; all-sample n={int(bench_n):,})",
        fontsize=13, fontweight="bold", y=0.995,
    )

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 2, idx % 2]
        values = [mae_df.loc[model, s] for s in SAMPLES]

        plot_vals = [0 if (v is None or (isinstance(v, float) and pd.isna(v))) else v for v in values]
        bars = ax.bar(range(len(SAMPLES)), plot_vals, color=SAMPLE_COLORS,
                      edgecolor="black", linewidth=0.6, width=0.65)

        for bar, v in zip(bars, values):
            if isinstance(v, float) and pd.isna(v):
                ax.text(bar.get_x() + bar.get_width() / 2, ymax * 0.02,
                        "n/a", ha="center", va="bottom", fontsize=10, color="grey", style="italic")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, v + ymax * 0.015,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=10)

        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(range(len(SAMPLES)))
        ax.set_xticklabels(sample_labels, fontsize=9)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx % 2 == 0:
            ax.set_ylabel("Mean |estimate − all-sample|", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = FIG_DIR / outcome
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mae_vs_all_sample.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    for outcome in OUTCOMES:
        out = plot_outcome(outcome)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
