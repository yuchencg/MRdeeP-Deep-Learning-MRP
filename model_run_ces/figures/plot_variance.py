"""
Per-state model variance plots for MRdeeP outcomes.

For each (sample_size x outcome) combination, produces one figure showing
the spread of estimates across non-baseline models per state, with the
baseline shown separately as an x marker. States are sorted top-to-bottom
by spread magnitude (widest disagreement at top).

Inputs:  model_run_ces/sample{N}_state/outputs/estimates/{outcome}_state_estimates.csv
Outputs: model_run_ces/figures/{outcome}/sample{N}_variance.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------- Configuration ----------

ROOT = Path(__file__).resolve().parents[1]  # model_run_ces/
FIG_DIR = ROOT / "figures"

SAMPLES = [1, 2, 3]
OUTCOMES = ["climate_problem", "renewable_fuel"]

# Non-baseline models that span the variance line
MODEL_COLS = ["glm_mrp", "glmer_mrp", "srp", "glmerstan", "mrdeep"]
BASELINE_COL = "baseline"

# Per-model marker styling
MODEL_STYLE = {
    "glm_mrp":    {"marker": "o", "color": "#1f77b4"},
    "glmer_mrp":  {"marker": "s", "color": "#2ca02c"},
    "srp":        {"marker": "^", "color": "#9467bd"},
    "glmerstan":  {"marker": "P", "color": "#ff7f0e"},
    "mrdeep":     {"marker": "D", "color": "#d62728"},
}
BASELINE_STYLE = {"marker": "x", "color": "#7f7f7f"}

OUTCOME_LABEL = {
    "climate_problem": "climate is a problem",
    "renewable_fuel":  "supports renewable fuel",
}

# ---------- Plotting ----------

def plot_one(df: pd.DataFrame, outcome: str, sample_n: int, out_path: Path) -> None:
    """Render the variance figure for one (sample, outcome) and write to out_path."""
    required = [BASELINE_COL] + MODEL_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"sample{sample_n} / {outcome}: missing required columns {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Sort by spread (widest disagreement among non-baseline models at top)
    df = df.copy()
    df["_spread"] = df[MODEL_COLS].max(axis=1) - df[MODEL_COLS].min(axis=1)
    df = df.sort_values("_spread", ascending=True).reset_index(drop=True)

    n_states = len(df)
    fig, ax = plt.subplots(figsize=(11, max(8, 0.28 * n_states)))

    y_positions = range(n_states)

    # Spread line: min -> max across non-baseline models
    for y, (_, row) in zip(y_positions, df.iterrows()):
        vmin = row[MODEL_COLS].min()
        vmax = row[MODEL_COLS].max()
        ax.plot([vmin, vmax], [y, y], color="#bcbcbc", linewidth=1.5, zorder=1)

    # Non-baseline model markers
    for model in MODEL_COLS:
        style = MODEL_STYLE[model]
        ax.scatter(
            df[model], list(y_positions),
            marker=style["marker"], color=style["color"],
            s=42, zorder=3, label=model, edgecolor="white", linewidth=0.5,
        )

    # Baseline marker (does not extend the spread line)
    ax.scatter(
        df[BASELINE_COL], list(y_positions),
        marker=BASELINE_STYLE["marker"], color=BASELINE_STYLE["color"],
        s=55, zorder=2, label="baseline", linewidth=1.5,
    )

    # Endpoint-only labels: min and max model values per state
    for y, (_, row) in zip(y_positions, df.iterrows()):
        vmin = row[MODEL_COLS].min()
        vmax = row[MODEL_COLS].max()
        ax.text(vmin - 0.012, y, f"{vmin:.0%}",
                ha="right", va="center", fontsize=7, color="#444")
        ax.text(vmax + 0.012, y, f"{vmax:.0%}",
                ha="left", va="center", fontsize=7, color="#444")

    # Axes
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(df["state_name"], fontsize=8)
    ax.set_xlim(-0.08, 1.12)
    ax.set_xlabel(f"Estimated proportion ({OUTCOME_LABEL.get(outcome, outcome)})")
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.axvline(0.5, color="#eeeeee", linewidth=1, zorder=0)

    # Remove clutter
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", color="#f0f0f0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Title
    ax.set_title(
        f"Per-state model variance — '{outcome}' (sample {sample_n}, n={int(df['n_respondents'].sum())})\n"
        f"Line spans min/max across non-baseline models; states sorted by spread (widest at top); "
        f"baseline shown as × (does not extend line)",
        fontsize=10, loc="left",
    )

    ax.legend(loc="lower right", frameon=True, fontsize=8, framealpha=0.95)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    failures: list[str] = []
    for outcome in OUTCOMES:
        out_dir = FIG_DIR / outcome
        for sample_n in SAMPLES:
            csv_path = (
                ROOT
                / f"sample{sample_n}_state"
                / "outputs"
                / "estimates"
                / f"{outcome}_state_estimates.csv"
            )
            print(f"sample{sample_n} / {outcome}:")
            if not csv_path.exists():
                msg = f"  SKIP: missing estimates file: {csv_path}"
                print(msg)
                failures.append(msg)
                continue
            df = pd.read_csv(csv_path)
            out_path = out_dir / f"sample{sample_n}_variance.png"
            try:
                plot_one(df, outcome, sample_n, out_path)
            except ValueError as e:
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
