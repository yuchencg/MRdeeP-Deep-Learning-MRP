"""
Where does the ERROR come from? — per-state scatter of distance from ground truth.

A ground-truth analog of plot_variance_sources_by_xsample.py. Instead of model
and sample *spread* (which are translation-invariant and so unchanged by a known
truth), this measures each state's *distance from the full-sample estimate*
(the all-sample weighted mean = "ground truth") at two sample sizes:

  x-axis = error at the LARGE sample (s3) = |mean-across-models - truth|  (pp)
           error that survives lots of data -> structural / model-choice error
  y-axis = error at the SMALL sample (s1) = |mean-across-models - truth|  (pp)
           error when data is scarce

Diagonal y = x splits the same way as the spread plot:
  above  -> small-sample error >> large-sample error -> sample size matters more
  below  -> error persists even with data           -> model choice matters more

Inputs:  model_run_ces/sample{N}_state/outputs/estimates/{outcome}_state_estimates.csv
         model_run_ces/all_sample/outputs/{outcome}_state_estimates.csv  (ground truth)
Outputs: model_run_ces/figures/{outcome}/error_sources_scatter{variant}.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- Configuration ----------

ROOT = Path(__file__).resolve().parents[1]  # model_run_ces/
FIG_DIR = ROOT / "figures"

SAMPLES = [1, 2, 3]
SMALL_SAMPLE = 1  # y-axis: scarce data
LARGE_SAMPLE = 3  # x-axis: plentiful data
OUTCOMES = ["climate_problem", "renewable_fuel"]

# Model variants: all 5 models, a glm_mrp-excluded view, and a mrdeep-excluded view
MODEL_VARIANTS: dict[str, list[str]] = {
    "":           ["glm_mrp", "glmer_mrp", "srp", "glmerstan", "mrdeep"],
    "_no_glm":    ["glmer_mrp", "srp", "glmerstan", "mrdeep"],
    "_no_mrdeep": ["glm_mrp", "glmer_mrp", "srp", "glmerstan"],
}

VARIANT_NOTES: dict[str, str] = {
    "":           "",
    "_no_glm":    " (excl. glm_mrp)",
    "_no_mrdeep": " (excl. MRdeeP)",
}

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


# ---------- Data ----------

def load_truth(outcome: str) -> pd.Series:
    """Full-sample estimate per state, indexed by state_name."""
    csv = ROOT / "all_sample" / "outputs" / f"{outcome}_state_estimates.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Missing ground-truth file: {csv}")
    return pd.read_csv(csv).set_index("state_name")["estimate"]


def compute_error_sources(
    outcome: str, model_cols: list[str]
) -> tuple[pd.DataFrame, dict[int, int]]:
    """For each state, compute |mean-across-models - truth| (pp) at the small
    and large sample sizes.

    Returns (df, sample_totals) where sample_totals maps sample -> total n.
    """
    truth = load_truth(outcome)

    per_sample: dict[int, pd.DataFrame] = {}
    sample_totals: dict[int, int] = {}
    for s in SAMPLES:
        csv = (
            ROOT
            / f"sample{s}_state"
            / "outputs"
            / "estimates"
            / f"{outcome}_state_estimates.csv"
        )
        if not csv.exists():
            raise FileNotFoundError(f"Missing estimates file: {csv}")
        df = pd.read_csv(csv)
        missing = [c for c in model_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"sample{s} / {outcome}: missing model columns {missing}. "
                f"Found: {list(df.columns)}"
            )
        sample_totals[s] = int(df["n_respondents"].sum())
        per_sample[s] = df.set_index("state_name")

    states = [s for s in per_sample[LARGE_SAMPLE].index if s in truth.index]
    t = truth.loc[states]

    def err_pp(sample: int) -> pd.Series:
        ens_mean = per_sample[sample].loc[states, model_cols].mean(axis=1)
        return (ens_mean - t).abs() * 100

    out = pd.DataFrame({
        "state_name": states,
        "error_large_pp": err_pp(LARGE_SAMPLE).values,
        "error_small_pp": err_pp(SMALL_SAMPLE).values,
    })
    return out, sample_totals


# ---------- Plot ----------

def plot_one(outcome: str, out_path: Path, model_cols: list[str], variant: str) -> None:
    df, totals = compute_error_sources(outcome, model_cols)
    n_models = len(model_cols)
    variant_note = VARIANT_NOTES.get(variant, "")
    n_lg, n_sm = totals[LARGE_SAMPLE], totals[SMALL_SAMPLE]
    k_lg, k_sm = round(n_lg, -3) // 1000, round(n_sm, -3) // 1000

    fig, ax = plt.subplots(figsize=(9, 9))

    x = df["error_large_pp"].values
    y = df["error_small_pp"].values

    # y=x reference
    lim = max(x.max(), y.max()) * 1.12
    ax.plot([0, lim], [0, lim], linestyle="--", color="#888", linewidth=1.2,
            label="y = x (data does not help)", zorder=1)

    # Shade the two half-planes
    ax.fill_between([0, lim], [0, lim], [lim, lim],
                    color="#9ab4d6", alpha=0.08, zorder=0)
    ax.fill_between([0, lim], [0, 0], [0, lim],
                    color="#e7a08a", alpha=0.08, zorder=0)

    # Color points by which source dominates; size by total error
    total = x + y
    above = y > x
    ax.scatter(x[~above], y[~above], s=40 + total[~above] * 4,
               color="#c0392b", alpha=0.78, edgecolor="white", linewidth=0.6,
               label="error persists with data (model choice)", zorder=3)
    ax.scatter(x[above], y[above], s=40 + total[above] * 4,
               color="#2c5f9e", alpha=0.78, edgecolor="white", linewidth=0.6,
               label="more data shrinks error (sample size)", zorder=3)

    # Label top-N states by total error
    top_idx = np.argsort(total)[::-1][:N_LABEL]
    for i in top_idx:
        abbr = STATE_ABBR.get(df["state_name"].iloc[i], df["state_name"].iloc[i])
        ax.annotate(
            abbr,
            xy=(x[i], y[i]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=9, color="#222", fontweight="bold",
        )

    # Region labels
    ax.text(lim * 0.78, lim * 0.94, "sample size\nmatters more",
            fontsize=10, color="#2c5f9e", ha="center", va="center",
            style="italic", alpha=0.9)
    ax.text(lim * 0.94, lim * 0.18, "model choice\nmatters more",
            fontsize=10, color="#c0392b", ha="center", va="center",
            style="italic", alpha=0.9)

    # Axes
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(
        f"Error at sample {LARGE_SAMPLE} (≈{k_lg:.0f}k, n={n_lg}) (pp)\n"
        f"|mean across {n_models} models{variant_note} − ground truth|"
    )
    ax.set_ylabel(
        f"Error at sample {SMALL_SAMPLE} (≈{k_sm:.0f}k, n={n_sm}) (pp)\n"
        f"|mean across models − ground truth|"
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(color="#f0f0f0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Summary stats in corner
    summary = (
        f"Mean error @ s{LARGE_SAMPLE} (large): {x.mean():.1f} pp\n"
        f"Mean error @ s{SMALL_SAMPLE} (small): {y.mean():.1f} pp\n"
        f"States where more data helped: {int(above.sum())} / {len(df)}"
    )
    ax.text(0.02, 0.98, summary, transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="#ddd", boxstyle="round,pad=0.4"))

    ax.set_title(
        f"How far from ground truth? — '{outcome}'{variant_note}\n"
        f"Error = |mean-across-models − full-sample estimate|. Each dot = one state.",
        fontsize=11, loc="left",
    )
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    failures: list[str] = []
    for outcome in OUTCOMES:
        print(f"{outcome}:")
        for variant, model_cols in MODEL_VARIANTS.items():
            out_path = FIG_DIR / outcome / f"error_sources_scatter{variant}.png"
            try:
                plot_one(outcome, out_path, model_cols, variant)
            except (ValueError, FileNotFoundError) as e:
                msg = f"  SKIP {outcome}{variant}: {e}"
                print(msg)
                failures.append(msg)

    if failures:
        print("\nFinished with errors:")
        for f in failures:
            print(f)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
