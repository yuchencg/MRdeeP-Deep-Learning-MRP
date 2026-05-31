"""
Per-state model-disagreement table.

For each outcome, produces one table:
  rows    = states (51), sorted by sample-3 variance descending
  columns = for each sample size: model min–max range and across-model variance

Variance and min/max are computed across the 5 non-baseline models
(glm_mrp, glmer_mrp, srp, glmerstan, mrdeep). Baseline is excluded.

Inputs:  model_run_ces/sample{N}_state/outputs/estimates/{outcome}_state_estimates.csv
Outputs: model_run_ces/figures/{outcome}/per_state_variance_table.png
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

SORT_BY_SAMPLE = 3  # sort rows by this sample size's variance


# ---------- Data ----------

def build_table(outcome: str) -> tuple[pd.DataFrame, dict[int, int]]:
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
        missing = [c for c in MODEL_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"sample{s} / {outcome}: missing model columns {missing}. "
                f"Found: {list(df.columns)}"
            )
        per_sample[s] = df.set_index("state_name")
        sample_totals[s] = int(df["n_respondents"].sum())

    states = per_sample[SORT_BY_SAMPLE].index.tolist()

    # Build per-state rows
    rows = []
    for state in states:
        row = {"state": state}
        for s in SAMPLES:
            vals = per_sample[s].loc[state, MODEL_COLS].values.astype(float) * 100  # pp
            row[f"ss{s}_range"] = f"{vals.min():.0f}% – {vals.max():.0f}%"
            row[f"ss{s}_var"] = float(np.var(vals, ddof=1))
        rows.append(row)

    table = pd.DataFrame(rows).set_index("state")
    table = table.sort_values(f"ss{SORT_BY_SAMPLE}_var", ascending=False)
    return table, sample_totals


# ---------- Plot ----------

def render_table(outcome: str, out_path: Path) -> None:
    table, totals = build_table(outcome)

    # Column order: ss1_range, ss1_var, ss2_range, ss2_var, ss3_range, ss3_var
    var_cols = [f"ss{s}_var" for s in SAMPLES]
    cols_in_order = []
    for s in SAMPLES:
        cols_in_order += [f"ss{s}_range", f"ss{s}_var"]

    # Headers
    col_headers = []
    for s in SAMPLES:
        col_headers.append(f"~{round(totals[s], -3) // 1000:.0f}k\nmin – max")
        col_headers.append(f"~{round(totals[s], -3) // 1000:.0f}k\nvariance (pp²)")

    # Cell text — format variances to 1 decimal
    cell_text = []
    for _, row in table[cols_in_order].iterrows():
        formatted = []
        for c in cols_in_order:
            if c.endswith("_var"):
                formatted.append(f"{row[c]:.1f}")
            else:
                formatted.append(str(row[c]))
        cell_text.append(formatted)

    # Color the variance columns by within-column rank (white = lowest, deeper blue = highest)
    n_rows, n_cols = len(cell_text), len(cols_in_order)
    cell_colors = [["#ffffff"] * n_cols for _ in range(n_rows)]
    for ci, cname in enumerate(cols_in_order):
        if not cname.endswith("_var"):
            continue
        col_vals = table[cname].values.astype(float)
        vmin, vmax = float(col_vals.min()), float(col_vals.max())
        rng = vmax - vmin if vmax > vmin else 1.0
        for ri in range(n_rows):
            t = (col_vals[ri] - vmin) / rng
            shade = 1.0 - 0.55 * t
            cell_colors[ri][ci] = (0.85 * shade + 0.15, 0.92 * shade + 0.08, 1.0)

    fig_height = max(10, 0.28 * n_rows)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.set_axis_off()

    table_artist = ax.table(
        cellText=cell_text,
        rowLabels=list(table.index),
        colLabels=col_headers,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8)
    table_artist.scale(1.0, 1.15)

    for (r, c), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#bbb")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#2c3e50")
            cell.get_text().set_color("white")
            cell.set_height(cell.get_height() * 1.4)
        if c == -1:
            cell.set_text_props(weight="bold", ha="right")
            cell.set_facecolor("#ecf0f1")

    fig.suptitle(
        f"Per-state model disagreement — '{outcome}'\n"
        f"min–max range and variance across the 5 non-baseline models "
        f"(glm_mrp, glmer_mrp, srp, glmerstan, mrdeep). "
        f"Sorted by sample-{SORT_BY_SAMPLE} variance, descending.",
        fontsize=10, x=0.02, ha="left", y=0.995,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    failures: list[str] = []
    for outcome in OUTCOMES:
        print(f"{outcome}:")
        out_path = FIG_DIR / outcome / "per_state_variance_table.png"
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
