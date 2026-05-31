"""
Per-state model variance — three sample sizes side by side.

One figure per outcome with three panels (sample 1, 2, 3). States share a
common y-axis sorted by sample-3 spread (widest disagreement at top), so
each horizontal row tracks one state across the three sample sizes and
convergence is visible as the spread line shrinking left-to-right.

Inputs:  model_run_ces/sample{N}_state/outputs/estimates/{outcome}_state_estimates.csv
Outputs: model_run_ces/figures/{outcome}/variance_by_sample.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------- Configuration ----------

ROOT = Path(__file__).resolve().parents[1]  # model_run_ces/
FIG_DIR = ROOT / "figures"

SAMPLES = [1, 2, 3]
OUTCOMES = ["climate_problem", "renewable_fuel"]

MODEL_COLS = ["glm_mrp", "glmer_mrp", "srp", "glmerstan", "mrdeep"]
BASELINE_COL = "baseline"

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


# ---------- Data ----------

def load_all_samples(outcome: str) -> dict[int, pd.DataFrame]:
    """Load all three sample CSVs for an outcome. Raises if any are missing cols."""
    out: dict[int, pd.DataFrame] = {}
    required = [BASELINE_COL] + MODEL_COLS
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
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"sample{s} / {outcome}: missing required columns {missing}. "
                f"Found: {list(df.columns)}"
            )
        out[s] = df
    return out


# ---------- Plot ----------

def draw_panel(ax, df: pd.DataFrame, state_order: list[str], sample_n: int) -> None:
    """Render one sample's panel into ax. df must contain all required cols."""
    df_idx = df.set_index("state_name")
    df_ord = df_idx.loc[state_order].reset_index()
    y_positions = list(range(len(state_order)))

    # Spread line: min -> max across non-baseline models
    for y, (_, row) in zip(y_positions, df_ord.iterrows()):
        vmin = row[MODEL_COLS].min()
        vmax = row[MODEL_COLS].max()
        ax.plot([vmin, vmax], [y, y], color="#bcbcbc", linewidth=1.4, zorder=1)

    # Non-baseline model markers
    for model in MODEL_COLS:
        style = MODEL_STYLE[model]
        ax.scatter(
            df_ord[model], y_positions,
            marker=style["marker"], color=style["color"],
            s=36, zorder=3, label=model, edgecolor="white", linewidth=0.4,
        )

    # Baseline marker (does not extend spread line)
    ax.scatter(
        df_ord[BASELINE_COL], y_positions,
        marker=BASELINE_STYLE["marker"], color=BASELINE_STYLE["color"],
        s=45, zorder=2, label="baseline", linewidth=1.3,
    )

    ax.axvline(0.5, color="#eeeeee", linewidth=1, zorder=0)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_ylim(-0.6, len(state_order) - 0.4)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", color="#f4f4f4", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    n_total = int(df["n_respondents"].sum())
    ax.set_title(f"sample {sample_n}  ·  n = {n_total}", fontsize=10, loc="left")


def plot_one(outcome: str, out_path: Path) -> None:
    samples = load_all_samples(outcome)

    # Sort state order by sample-3 spread (widest at top)
    s3 = samples[3].copy()
    s3["_spread"] = s3[MODEL_COLS].max(axis=1) - s3[MODEL_COLS].min(axis=1)
    state_order = s3.sort_values("_spread", ascending=True)["state_name"].tolist()

    n_states = len(state_order)
    fig, axes = plt.subplots(
        nrows=1, ncols=3,
        figsize=(18, max(8, 0.28 * n_states)),
        sharey=True, sharex=True,
    )

    for ax, s in zip(axes, SAMPLES):
        draw_panel(ax, samples[s], state_order, s)

    # Y-axis labels on leftmost panel only
    axes[0].set_yticks(range(n_states))
    axes[0].set_yticklabels(state_order, fontsize=8)
    axes[0].set_ylabel("State (sorted by sample-3 spread, widest at top)")

    # X-label on middle panel
    axes[1].set_xlabel(f"Estimated proportion ({OUTCOME_LABEL.get(outcome, outcome)})")

    # Suptitle
    fig.suptitle(
        f"Per-state model variance across sample sizes — '{outcome}'\n"
        f"Line spans min/max across non-baseline models; baseline shown as × (does not extend line)",
        fontsize=11, x=0.01, ha="left", y=0.995,
    )

    # Single shared legend at the bottom
    handles, labels = axes[-1].get_legend_handles_labels()
    # Deduplicate while preserving order
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    fig.legend(
        [h for h, _ in uniq], [l for _, l in uniq],
        loc="lower center", ncol=len(uniq), frameon=True, fontsize=9,
        bbox_to_anchor=(0.5, -0.005),
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    failures: list[str] = []
    for outcome in OUTCOMES:
        print(f"{outcome}:")
        out_path = FIG_DIR / outcome / "variance_by_sample.png"
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
