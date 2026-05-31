"""
Per-model cross-state variance table, split by affirmative-response tercile.

Like plot_variance_table.py but each model row is broken into three
sub-rows — top/mid/bottom third of states by mean affirmative response.

Tercile assignment is shared across all models: each state's per-state
mean across the 5 non-baseline models and 3 sample sizes (15 estimates)
is used to assign it to a tercile (17 states each). The same assignment
is then applied to every model row so cells are comparable.

Inputs:  model_run_ces/sample{N}_state/outputs/estimates/{outcome}_state_estimates.csv
Outputs: model_run_ces/figures/{outcome}/variance_table_by_tercile.png
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

MODEL_ROWS = ["baseline", "glm_mrp", "glmer_mrp", "srp", "glmerstan", "mrdeep"]
REFERENCE_MODELS = ["glm_mrp", "glmer_mrp", "srp", "glmerstan", "mrdeep"]  # for bucketing

TERCILE_LABELS = ["high third", "mid third", "low third"]  # high = top, low = bottom


# ---------- Data ----------

def load_samples(outcome: str) -> tuple[dict[int, pd.DataFrame], dict[int, int]]:
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
        missing = [c for c in MODEL_ROWS if c not in df.columns]
        if missing:
            raise ValueError(
                f"sample{s} / {outcome}: missing required model columns {missing}. "
                f"Found: {list(df.columns)}"
            )
        per_sample[s] = df.set_index("state_name")
        sample_totals[s] = int(df["n_respondents"].sum())
    return per_sample, sample_totals


def assign_terciles(per_sample: dict[int, pd.DataFrame]) -> pd.Series:
    """Return Series indexed by state_name with values in {'high third','mid third','low third'}.

    Each state's reference value is the mean across the 5 non-baseline models
    AND the 3 sample sizes — 15 estimates per state."""
    states = per_sample[3].index.tolist()
    parts = []
    for s in SAMPLES:
        parts.append(per_sample[s].loc[states, REFERENCE_MODELS])
    stacked = pd.concat(parts, axis=0)
    ref = stacked.groupby(level=0).mean().mean(axis=1)  # per-state mean across models x samples

    # Rank-based tercile cut so each bucket has ~17 states even with ties
    ranks = ref.rank(method="first", ascending=False)
    n = len(ranks)
    third = n / 3.0
    bucket = pd.Series(index=ref.index, dtype=object)
    bucket[ranks <= third] = "high third"
    bucket[(ranks > third) & (ranks <= 2 * third)] = "mid third"
    bucket[ranks > 2 * third] = "low third"
    return bucket


def build_table(outcome: str) -> tuple[pd.DataFrame, dict[int, int], pd.Series]:
    per_sample, totals = load_samples(outcome)
    tercile = assign_terciles(per_sample)

    # Rows: (model, tercile)
    idx = pd.MultiIndex.from_product([MODEL_ROWS, TERCILE_LABELS], names=["model", "tercile"])
    cols = [f"ss{s}" for s in SAMPLES]
    table = pd.DataFrame(index=idx, columns=cols, dtype=float)

    for m in MODEL_ROWS:
        for tlabel in TERCILE_LABELS:
            states_in_bucket = tercile[tercile == tlabel].index.tolist()
            for s in SAMPLES:
                vals_pp = per_sample[s].loc[states_in_bucket, m].values * 100
                table.loc[(m, tlabel), f"ss{s}"] = float(np.var(vals_pp, ddof=1))

    table["mean"] = table[cols].mean(axis=1)
    table["max"] = table[cols].max(axis=1)
    return table, totals, tercile


# ---------- Plot ----------

def render_table(outcome: str, out_path: Path) -> None:
    table, totals, tercile = build_table(outcome)

    col_headers = [
        f"Sample size ≈ {round(totals[s], -3) // 1000:.0f}k\n(n = {totals[s]})"
        for s in SAMPLES
    ] + ["Mean across\nsample sizes", "Max across\nsample sizes"]

    # Row labels: include model only on first tercile row
    row_headers = []
    for m in MODEL_ROWS:
        for i, tlabel in enumerate(TERCILE_LABELS):
            if i == 0:
                row_headers.append(f"{m}\n— {tlabel}")
            else:
                row_headers.append(f"— {tlabel}")

    cell_text = [[f"{v:.1f}" for v in row] for row in table.values]

    # Color cells by within-column rank within tercile band (so colors are
    # comparable within a "high third"/"mid third"/"low third" slice)
    n_rows, n_cols = table.shape
    cell_colors = [["#ffffff"] * n_cols for _ in range(n_rows)]
    for ci in range(n_cols):
        col_vals = table.iloc[:, ci].values.astype(float)
        # Color within tercile group (rows 0,3,6,9,12,15 = high; 1,4,...= mid; 2,5,...=low)
        for offset in range(3):
            band_idx = list(range(offset, n_rows, 3))
            band_vals = col_vals[band_idx]
            vmin, vmax = float(np.min(band_vals)), float(np.max(band_vals))
            rng = vmax - vmin if vmax > vmin else 1.0
            for ri in band_idx:
                t = (col_vals[ri] - vmin) / rng
                shade = 1.0 - 0.55 * t
                cell_colors[ri][ci] = (0.85 * shade + 0.15, 0.92 * shade + 0.08, 1.0)

    fig, ax = plt.subplots(figsize=(12, 0.42 * (n_rows + 2)))
    ax.set_axis_off()

    table_artist = ax.table(
        cellText=cell_text,
        rowLabels=row_headers,
        colLabels=col_headers,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(9)
    table_artist.scale(1.0, 1.4)

    # Style: bold header, row label band, separator between models
    for (r, c), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#bbb")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#2c3e50")
            cell.get_text().set_color("white")
            cell.set_height(cell.get_height() * 1.3)
        if c == -1:
            cell.set_text_props(ha="left")
            # First tercile row of each model gets a darker band + bold
            if (r - 1) % 3 == 0 and r > 0:
                cell.set_facecolor("#d5dbdb")
                cell.set_text_props(weight="bold", ha="left")
            else:
                cell.set_facecolor("#ecf0f1")
        # Heavier border between models
        if r > 0 and (r - 1) % 3 == 0:
            cell.set_linewidth(1.5)

    # Tercile range strings for the caption
    states = tercile.index.tolist()
    per_state_mean = (
        pd.concat([load_samples(outcome)[0][s].loc[states, REFERENCE_MODELS]
                   for s in SAMPLES], axis=0)
        .groupby(level=0).mean().mean(axis=1)
    )
    ranges = {}
    for t in TERCILE_LABELS:
        in_t = per_state_mean[tercile == t]
        ranges[t] = (in_t.min() * 100, in_t.max() * 100, len(in_t))

    caption = (
        "Tercile assignment uses each state's mean affirmative response across "
        "the 5 non-baseline models × 3 sample sizes. "
        f"Ranges (pp): "
        f"high {ranges['high third'][0]:.0f}–{ranges['high third'][1]:.0f} "
        f"({ranges['high third'][2]} states); "
        f"mid {ranges['mid third'][0]:.0f}–{ranges['mid third'][1]:.0f} "
        f"({ranges['mid third'][2]} states); "
        f"low {ranges['low third'][0]:.0f}–{ranges['low third'][1]:.0f} "
        f"({ranges['low third'][2]} states)."
    )

    fig.suptitle(
        f"Cross-state variance by tercile of affirmative response — '{outcome}'\n"
        f"Variance (pp²) of state estimates within each tercile, at each sample size.\n"
        f"{caption}",
        fontsize=10, x=0.02, ha="left", y=0.99,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    failures: list[str] = []
    for outcome in OUTCOMES:
        print(f"{outcome}:")
        out_path = FIG_DIR / outcome / "variance_table_by_tercile.png"
        try:
            render_table(outcome, out_path)
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
