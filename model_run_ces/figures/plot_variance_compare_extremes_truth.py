"""
Extremes variance plot (no mrdeep) with all-sample "ground truth" goalpost.

Same layout as plot_variance_compare_extremes_no_mrdeep.py, but overlays the
full-sample weighted estimate as a vertical "goalpost" tick per row — the
target the small-sample models are trying to recover. Vertical orientation
keeps it visually distinct from the horizontal spread bar and the model
markers, so it adds a reference without crowding the row.

Note: for small states (e.g. DC, Alaska, Vermont) the full-sample estimate is
itself based on few respondents, so the goalpost carries sampling noise there.

Inputs:  model_run_ces/sample{N}_state/outputs/estimates/{outcome}_state_estimates.csv
         model_run_ces/all_sample/outputs/{outcome}_state_estimates.csv  (ground truth)
Outputs: model_run_ces/figures/{outcome}/variance_by_sample_extremes{N}_truth.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------- Configuration ----------

ROOT = Path(__file__).resolve().parents[1]  # model_run_ces/
FIG_DIR = ROOT / "figures"

SAMPLES = [1, 2, 3]
OUTCOMES = ["climate_problem", "renewable_fuel"]

MODEL_COLS = ["glm_mrp", "glmer_mrp", "srp", "glmerstan"]

MODEL_STYLE = {
    "glm_mrp":    {"marker": "o", "color": "#1f77b4"},
    "glmer_mrp":  {"marker": "s", "color": "#2ca02c"},
    "srp":        {"marker": "^", "color": "#9467bd"},
    "glmerstan":  {"marker": "P", "color": "#ff7f0e"},
}
TRUTH_STYLE = {"color": "#000000", "linewidth": 1.8, "half_height": 0.26}

OUTCOME_LABEL = {
    "climate_problem": "climate is a problem",
    "renewable_fuel":  "supports renewable fuel",
}

EXTREME_COUNTS = [3, 5]  # produce one figure per count, from each end of the ranking


# ---------- Data ----------

def load_all_samples(outcome: str) -> dict[int, pd.DataFrame]:
    """Load all three sample CSVs for an outcome. Raises if any are missing cols."""
    out: dict[int, pd.DataFrame] = {}
    required = MODEL_COLS
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


def load_truth(outcome: str) -> pd.Series:
    """Full-sample estimate per state, indexed by state_name."""
    csv = ROOT / "all_sample" / "outputs" / f"{outcome}_state_estimates.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Missing ground-truth file: {csv}")
    df = pd.read_csv(csv)
    return df.set_index("state_name")["estimate"]


# ---------- Plot ----------

def draw_panel(
    ax, df: pd.DataFrame, truth: pd.Series, state_order: list[str], sample_n: int
) -> None:
    """Render one sample's panel into ax. df must contain all required cols."""
    df_idx = df.set_index("state_name")
    df_ord = df_idx.loc[state_order].reset_index()
    y_positions = list(range(len(state_order)))

    # Ground-truth goalpost: vertical tick at the full-sample estimate
    hh = TRUTH_STYLE["half_height"]
    for y, state in zip(y_positions, state_order):
        ax.vlines(
            truth[state], y - hh, y + hh,
            color=TRUTH_STYLE["color"], linewidth=TRUTH_STYLE["linewidth"],
            zorder=2, label="ground truth (full sample)",
        )

    # Spread line: min -> max across non-baseline models, labelled with the
    # spread (percentage points) centered just above the bar
    for y, (_, row) in zip(y_positions, df_ord.iterrows()):
        vmin = row[MODEL_COLS].min()
        vmax = row[MODEL_COLS].max()
        ax.plot([vmin, vmax], [y, y], color="#bcbcbc", linewidth=1.4, zorder=1)
        ax.text(
            (vmin + vmax) / 2, y + 0.33, f"{(vmax - vmin) * 100:.0f} pp",
            ha="center", va="bottom", fontsize=8, color="#444444", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.85),
        )

    # Non-baseline model markers
    for model in MODEL_COLS:
        style = MODEL_STYLE[model]
        ax.scatter(
            df_ord[model], y_positions,
            marker=style["marker"], color=style["color"],
            s=36, zorder=3, label=model, edgecolor="white", linewidth=0.4,
        )

    ax.axvline(0.5, color="#eeeeee", linewidth=1, zorder=0)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_ylim(-0.7, len(state_order) - 0.2)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", color="#f4f4f4", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    n_total = int(df["n_respondents"].sum())
    ax.set_title(f"sample {sample_n}  ·  n = {n_total}", fontsize=10, loc="left")


def plot_one(outcome: str, n_extreme: int, out_path: Path) -> None:
    samples = load_all_samples(outcome)
    truth = load_truth(outcome)

    # Rank states by sample-3 spread, then keep the n_extreme lowest + highest
    s3 = samples[3].copy()
    s3["_spread"] = s3[MODEL_COLS].max(axis=1) - s3[MODEL_COLS].min(axis=1)
    ranked = s3.sort_values("_spread", ascending=True)["state_name"].tolist()
    state_order = ranked[:n_extreme] + ranked[-n_extreme:]  # lowest at bottom, highest at top

    n_states = len(state_order)
    fig, axes = plt.subplots(
        nrows=1, ncols=3,
        figsize=(18, max(4, 0.7 * n_states)),
        sharey=True, sharex=True,
    )

    for ax, s in zip(axes, SAMPLES):
        draw_panel(ax, samples[s], truth, state_order, s)

    # Y-axis labels on leftmost panel only
    axes[0].set_yticks(range(n_states))
    axes[0].set_yticklabels(state_order, fontsize=8)
    axes[0].set_ylabel(f"{n_extreme} highest-variance (top) · {n_extreme} lowest (bottom)", fontsize=9)

    # X-label on middle panel
    axes[1].set_xlabel(f"Estimated proportion ({OUTCOME_LABEL.get(outcome, outcome)})")

    # Suptitle
    fig.suptitle(
        f"Per-state model variance across sample sizes — '{outcome}' "
        f"(top/bottom {n_extreme} by spread, excl. mrdeep)\n"
        f"Line spans min/max across models; labelled in percentage points; "
        f"vertical bar = full-sample estimate (target)",
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
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    failures: list[str] = []
    for outcome in OUTCOMES:
        print(f"{outcome}:")
        for n in EXTREME_COUNTS:
            out_path = FIG_DIR / outcome / f"variance_by_sample_extremes{n}_truth.png"
            try:
                plot_one(outcome, n, out_path)
            except (ValueError, FileNotFoundError) as e:
                msg = f"  SKIP (n={n}): {e}"
                print(msg)
                failures.append(msg)

    if failures:
        print("\nFinished with errors:")
        for f in failures:
            print(f)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
