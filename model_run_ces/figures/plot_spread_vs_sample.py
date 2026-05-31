"""
How estimator disagreement shrinks with sample size.

For each outcome, plots per-state between-model spread (max - min across
the five non-baseline models) across the three sample sizes. Each state
is a thin grey line; the median state and across-state IQR are overlaid,
and the top-N highest-spread states are labelled in color.

Inputs:  model_run_ces/sample{N}_state/outputs/estimates/{outcome}_state_estimates.csv
Outputs: model_run_ces/figures/{outcome}/spread_vs_sample.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- Configuration ----------

ROOT = Path(__file__).resolve().parents[1]  # model_run_ces/
FIG_DIR = ROOT / "figures"

SAMPLES = [1, 2, 3]
OUTCOMES = ["climate_problem", "renewable_fuel"]
MODEL_COLS = ["glm_mrp", "glmer_mrp", "srp", "glmerstan", "mrdeep"]

N_HIGHLIGHT = 3  # how many top-spread states to label

# Distinct USPS abbreviations for highlights
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

HIGHLIGHT_COLORS = ["#c0392b", "#e67e22", "#8e44ad", "#16a085", "#2980b9"]


# ---------- Data ----------

def load_spread_table(outcome: str) -> tuple[pd.DataFrame, dict[int, int]]:
    """Return (wide_spread_pp, sample_totals).

    wide_spread_pp: index=state_name, columns=sample_n, values=spread in percentage points
    sample_totals: dict {sample_n: total n_respondents}
    """
    per_sample = {}
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
        missing = [c for c in MODEL_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"sample{s} / {outcome}: missing model columns {missing}. "
                f"Found: {list(df.columns)}"
            )
        spread_pp = (df[MODEL_COLS].max(axis=1) - df[MODEL_COLS].min(axis=1)) * 100
        per_sample[s] = pd.Series(spread_pp.values, index=df["state_name"], name=s)
        sample_totals[s] = int(df["n_respondents"].sum())

    wide = pd.concat(per_sample.values(), axis=1)
    wide.columns = list(per_sample.keys())
    return wide, sample_totals


# ---------- Plot ----------

def plot_one(outcome: str, out_path: Path) -> None:
    wide, totals = load_spread_table(outcome)

    fig, ax = plt.subplots(figsize=(10, 7.5))

    x = np.array(SAMPLES, dtype=float)
    x_labels = [f"{totals[s]}\n(s{s}≈{round(totals[s], -3) // 1000:.0f}k)" for s in SAMPLES]

    # Per-state thin grey lines
    for state, row in wide.iterrows():
        ax.plot(x, row.values, color="#bbbbbb", linewidth=0.8, alpha=0.6, zorder=1)

    # Across-state IQR band
    q25 = wide.quantile(0.25, axis=0).values
    q75 = wide.quantile(0.75, axis=0).values
    ax.fill_between(x, q25, q75, color="#9ab4d6", alpha=0.35,
                    label="across-state IQR", zorder=2)

    # Median state line
    med = wide.median(axis=0).values
    ax.plot(x, med, color="#1f3a5f", linewidth=2.6, marker="o", markersize=6,
            label="median state", zorder=4)

    # Highlight top-N states by mean spread across samples
    mean_spread = wide.mean(axis=1).sort_values(ascending=False)
    top_states = mean_spread.head(N_HIGHLIGHT).index.tolist()
    for color, state in zip(HIGHLIGHT_COLORS, top_states):
        vals = wide.loc[state].values
        ax.plot(x, vals, color=color, linewidth=2.0, marker="D", markersize=5,
                zorder=5)
        # Label to the right of last point
        ax.text(
            x[-1] + 0.04, vals[-1],
            STATE_ABBR.get(state, state),
            color=color, fontsize=10, fontweight="bold",
            va="center", ha="left",
        )

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(x[0] - 0.15, x[-1] + 0.45)
    ax.set_ylim(0, max(wide.values.max() * 1.08, 10))
    ax.set_xlabel("Achieved sample size")
    ax.set_ylabel("Between-model spread (max − min), percentage points")

    ax.set_title(
        f"How estimator disagreement shrinks with sample size\n"
        f"'{outcome}' — spread across the five non-baseline models, per state",
        fontsize=11, loc="left",
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color="#f0f0f0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    failures: list[str] = []
    for outcome in OUTCOMES:
        print(f"{outcome}:")
        out_path = FIG_DIR / outcome / "spread_vs_sample.png"
        try:
            plot_one(outcome, out_path)
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
