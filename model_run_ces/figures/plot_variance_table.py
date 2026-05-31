"""
Per-model cross-state variance table.

For each outcome, produces one table figure:
  rows    = models (baseline + 5 non-baseline)
  columns = variance of the 51 state estimates at each sample size (in pp^2),
            plus mean and max across the three sample-size columns

"Variance" here = variance of a model's 51 state-level estimates at a given
sample size — i.e., how much geographic differentiation the model produces.
Reported in percentage-points-squared (pp^2) to match the other variance
figures' unit of percentage points.

Inputs:  model_run_ces/sample{N}_state/outputs/estimates/{outcome}_state_estimates.csv
Outputs: model_run_ces/figures/{outcome}/variance_table.png
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


# ---------- Data ----------

def build_table(outcome: str) -> tuple[pd.DataFrame, dict[int, int]]:
    """Return (table_df, sample_totals).

    table_df: index = model, columns = [ss1, ss2, ss3, mean, max], values = variance
              of the 51 state estimates at that sample size (in pp^2).
    sample_totals: {sample_n: total n_respondents}
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
        missing = [c for c in MODEL_ROWS if c not in df.columns]
        if missing:
            raise ValueError(
                f"sample{s} / {outcome}: missing required model columns {missing}. "
                f"Found: {list(df.columns)}"
            )
        per_sample[s] = df
        sample_totals[s] = int(df["n_respondents"].sum())

    # Variance in pp^2: scale estimates by 100 (probability -> pp) then var
    table = pd.DataFrame(index=MODEL_ROWS, columns=[f"ss{s}" for s in SAMPLES], dtype=float)
    for s in SAMPLES:
        for m in MODEL_ROWS:
            vals_pp = per_sample[s][m].values * 100
            table.loc[m, f"ss{s}"] = float(np.var(vals_pp, ddof=1))

    table["mean"] = table[[f"ss{s}" for s in SAMPLES]].mean(axis=1)
    table["max"] = table[[f"ss{s}" for s in SAMPLES]].max(axis=1)
    return table, sample_totals


# ---------- Plot ----------

def render_table(outcome: str, out_path: Path) -> None:
    table, totals = build_table(outcome)

    # Column headers using actual n_respondents totals
    col_headers = [
        f"Sample size ≈ {round(totals[s], -3) // 1000:.0f}k\n(n = {totals[s]})"
        for s in SAMPLES
    ] + ["Mean across\nsample sizes", "Max across\nsample sizes"]

    row_headers = MODEL_ROWS

    # Format cell values: 1 decimal pp^2
    cell_text = [[f"{v:.1f}" for v in row] for row in table.values]

    # Color cells by within-column rank (lighter = lower variance, darker = higher)
    n_rows, n_cols = table.shape
    cell_colors = [["#ffffff"] * n_cols for _ in range(n_rows)]
    for ci in range(n_cols):
        col_vals = table.iloc[:, ci].values.astype(float)
        vmin, vmax = float(np.min(col_vals)), float(np.max(col_vals))
        rng = vmax - vmin if vmax > vmin else 1.0
        for ri in range(n_rows):
            t = (col_vals[ri] - vmin) / rng
            # Light blue gradient
            shade = 1.0 - 0.55 * t
            cell_colors[ri][ci] = (0.85 * shade + 0.15, 0.92 * shade + 0.08, 1.0)

    fig, ax = plt.subplots(figsize=(11, 0.55 * (n_rows + 2)))
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
    table_artist.set_fontsize(10)
    table_artist.scale(1.0, 1.6)

    # Bold header row + row labels
    for (r, c), cell in table_artist.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#2c3e50")
            cell.get_text().set_color("white")
            cell.set_height(cell.get_height() * 1.3)
        if c == -1:
            cell.set_text_props(weight="bold", ha="right")
            cell.set_facecolor("#ecf0f1")
        cell.set_edgecolor("#bbb")

    fig.suptitle(
        f"Cross-state variance of state estimates by model — '{outcome}'\n"
        f"Values are variance (pp²) of the 51 state-level estimates at each sample size. "
        f"Higher = more geographic differentiation.",
        fontsize=11, x=0.02, ha="left", y=0.98,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    failures: list[str] = []
    for outcome in OUTCOMES:
        print(f"{outcome}:")
        out_path = FIG_DIR / outcome / "variance_table.png"
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
