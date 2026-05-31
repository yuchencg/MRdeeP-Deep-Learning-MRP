"""
Sample-size variance vs model-choice variance — per-state scatter,
with a selectable sample size on the x-axis.

Same as plot_variance_sources.py, but the model-spread on the x-axis can be
taken from any of the three samples instead of always sample 3. The y-axis
(sample-size spread across s1/s2/s3) is unchanged across variants.

For each (outcome, x-sample, model-variant) it plots one panel, one dot per state:
  x-axis = model spread at the chosen sample (max model - min model, in pp)
  y-axis = sample spread (max - min, in pp, of the per-sample mean-across-models)

Inputs:  model_run_ces/sample{N}_state/outputs/estimates/{outcome}_state_estimates.csv
Outputs: model_run_ces/figures/{outcome}/variance_sources_scatter_x{k}{variant}.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- Configuration ----------

ROOT = Path(__file__).resolve().parents[1]  # model_run_ces/
FIG_DIR = ROOT / "figures"

SAMPLES = [1, 2, 3]
X_SAMPLES = [1, 2, 3]  # which sample's model-spread to put on the x-axis
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

N_LABEL = 8  # how many highest-variance states to label

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

def compute_variance_sources(
    outcome: str, model_cols: list[str], x_sample: int
) -> tuple[pd.DataFrame, dict[int, int]]:
    """For each state, compute model-spread (at x_sample) and sample-spread
    (max-min of per-sample model-mean), both in percentage points.

    Returns (df, sample_totals) where sample_totals maps sample -> total n.
    """
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

    states = per_sample[x_sample].index.tolist()

    # Model spread at the chosen x_sample (in pp)
    sx = per_sample[x_sample]
    model_spread_pp = (sx[model_cols].max(axis=1) - sx[model_cols].min(axis=1)) * 100

    # Sample spread: per-sample model-mean, then max-min across samples (in pp)
    means_per_sample = pd.DataFrame(
        {s: per_sample[s].loc[states, model_cols].mean(axis=1) for s in SAMPLES},
        index=states,
    )
    sample_spread_pp = (means_per_sample.max(axis=1) - means_per_sample.min(axis=1)) * 100

    out = pd.DataFrame({
        "state_name": states,
        "model_spread_pp": model_spread_pp.values,
        "sample_spread_pp": sample_spread_pp.values,
    })
    return out, sample_totals


# ---------- Plot ----------

def plot_one(
    outcome: str, out_path: Path, model_cols: list[str], variant: str, x_sample: int
) -> None:
    df, totals = compute_variance_sources(outcome, model_cols, x_sample)
    n_models = len(model_cols)
    variant_note = VARIANT_NOTES.get(variant, "")
    x_n = totals[x_sample]
    x_k = round(x_n, -3) // 1000

    fig, ax = plt.subplots(figsize=(9, 9))

    x = df["model_spread_pp"].values
    y = df["sample_spread_pp"].values

    # y=x reference
    lim = max(x.max(), y.max()) * 1.12
    ax.plot([0, lim], [0, lim], linestyle="--", color="#888", linewidth=1.2,
            label="y = x (equal contribution)", zorder=1)

    # Shade the two half-planes
    ax.fill_between([0, lim], [0, lim], [lim, lim],
                    color="#9ab4d6", alpha=0.08, zorder=0)
    ax.fill_between([0, lim], [0, 0], [0, lim],
                    color="#e7a08a", alpha=0.08, zorder=0)

    # Color points by which source dominates; size by total
    total = x + y
    above = y > x
    ax.scatter(x[~above], y[~above], s=40 + total[~above] * 4,
               color="#c0392b", alpha=0.78, edgecolor="white", linewidth=0.6,
               label="model choice > sample size", zorder=3)
    ax.scatter(x[above], y[above], s=40 + total[above] * 4,
               color="#2c5f9e", alpha=0.78, edgecolor="white", linewidth=0.6,
               label="sample size > model choice", zorder=3)

    # Label top-N states by total uncertainty
    top_idx = np.argsort(total)[::-1][:N_LABEL]
    for i in top_idx:
        abbr = STATE_ABBR.get(df["state_name"].iloc[i], df["state_name"].iloc[i])
        ax.annotate(
            abbr,
            xy=(x[i], y[i]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=9, color="#222", fontweight="bold",
        )

    # Quadrant labels
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
        f"Model spread at sample {x_sample} (≈{x_k:.0f}k, n={x_n}) (pp)\n"
        f"max − min across {n_models} models{variant_note}"
    )
    ax.set_ylabel("Sample-size spread (pp)\nmax − min of mean-across-models, across s1/s2/s3")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(color="#f0f0f0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Summary stats in corner
    mean_model = x.mean()
    mean_sample = y.mean()
    summary = (
        f"Mean model spread:  {mean_model:.1f} pp\n"
        f"Mean sample spread: {mean_sample:.1f} pp\n"
        f"States where sample > model: {int(above.sum())} / {len(df)}"
    )
    ax.text(0.02, 0.98, summary, transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="#ddd", boxstyle="round,pad=0.4"))

    ax.set_title(
        f"Where does the uncertainty come from? — '{outcome}'{variant_note}\n"
        f"Model spread measured at sample {x_sample} (≈{x_k:.0f}k). "
        f"Each dot = one state.",
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
        for x_sample in X_SAMPLES:
            for variant, model_cols in MODEL_VARIANTS.items():
                # Resolve the ≈k label from the data for the filename
                try:
                    _, totals = compute_variance_sources(outcome, model_cols, x_sample)
                except (ValueError, FileNotFoundError) as e:
                    msg = f"  SKIP {outcome} x-sample{x_sample}{variant}: {e}"
                    print(msg)
                    failures.append(msg)
                    continue
                x_k = round(totals[x_sample], -3) // 1000
                print(f"{outcome} x-sample{x_sample} (≈{x_k:.0f}k){variant}:")
                out_path = (
                    FIG_DIR / outcome
                    / f"variance_sources_scatter_x{x_k:.0f}k{variant}.png"
                )
                try:
                    plot_one(outcome, out_path, model_cols, variant, x_sample)
                except (ValueError, FileNotFoundError) as e:
                    msg = f"  SKIP: {e}"
                    print(msg)
                    failures.append(msg)

    if failures:
        print("\nFinished with errors:")
        for f in failures:
            print(f)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
