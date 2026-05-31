"""
2x3 grid of US state choropleth maps, one per model.

For each (outcome x sample_size), produces one figure with six US maps —
baseline, glm_mrp, glmer_mrp, srp, glmerstan, mrdeep — colored by the
estimated proportion answering affirmative. Color scale is fixed to
[0, 1] across all six panels so visual differences are directly
comparable.

Inputs:  model_run_ces/sample{N}_state/outputs/estimates/{outcome}_state_estimates.csv
Outputs: model_run_ces/figures/{outcome}/sample{N}_maps.png
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- Configuration ----------

ROOT = Path(__file__).resolve().parents[1]  # model_run_ces/
FIG_DIR = ROOT / "figures"

SAMPLES = [1, 2, 3]
OUTCOMES = ["climate_problem", "renewable_fuel"]

# Panel order — 2 rows x 3 cols
MODEL_PANELS = [
    ["baseline",  "glm_mrp",   "glmer_mrp"],
    ["srp",       "glmerstan", "mrdeep"],
]
ALL_MODELS = [m for row in MODEL_PANELS for m in row]

OUTCOME_LABEL = {
    "climate_problem": "Climate is a problem",
    "renewable_fuel":  "Supports renewable fuel",
}

# Diverging-around-0.5 scale (matches map_app convention)
COLOR_SCALE = "RdBu"
COLOR_RANGE = (0.0, 1.0)

# state_fips -> USPS abbreviation (required by Plotly's locationmode="USA-states")
FIPS_TO_USPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}


# ---------- Plot ----------

def plot_one(df: pd.DataFrame, outcome: str, sample_n: int, out_path: Path) -> None:
    missing = [m for m in ALL_MODELS if m not in df.columns]
    if missing:
        raise ValueError(
            f"sample{sample_n} / {outcome}: missing required model columns {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Ensure state_fips is zero-padded string for the dictionary lookup
    df = df.copy()
    df["state_fips"] = df["state_fips"].astype(str).str.zfill(2)
    df["usps"] = df["state_fips"].map(FIPS_TO_USPS)
    if df["usps"].isna().any():
        bad = df.loc[df["usps"].isna(), "state_fips"].unique().tolist()
        raise ValueError(f"Unknown state_fips values: {bad}")

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{"type": "choropleth"}] * 3] * 2,
        subplot_titles=[m for row in MODEL_PANELS for m in row],
        horizontal_spacing=0.02,
        vertical_spacing=0.06,
    )

    n_total = int(df["n_respondents"].sum())
    customdata = df[["state_name", "n_respondents"]].to_numpy()

    for r, row_models in enumerate(MODEL_PANELS, start=1):
        for c, model in enumerate(row_models, start=1):
            show_colorbar = (r == 1 and c == 3)  # one shared colorbar
            fig.add_trace(
                go.Choropleth(
                    locations=df["usps"],
                    z=df[model],
                    locationmode="USA-states",
                    zmin=COLOR_RANGE[0],
                    zmax=COLOR_RANGE[1],
                    zmid=0.5,
                    colorscale=COLOR_SCALE,
                    reversescale=False,
                    customdata=customdata,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"Model: {model}<br>"
                        "Estimate: %{z:.1%}<br>"
                        "Respondents: %{customdata[1]}"
                        "<extra></extra>"
                    ),
                    showscale=show_colorbar,
                    colorbar=dict(
                        title=dict(text="Pr(affirmative)", side="right"),
                        tickformat=".0%",
                        thickness=14,
                        len=0.85,
                        x=1.02,
                    ) if show_colorbar else None,
                ),
                row=r, col=c,
            )

    # Apply USA scope to every geo subplot
    for i in range(1, 7):
        key = "geo" if i == 1 else f"geo{i}"
        fig.layout[key].update(
            scope="usa",
            showlakes=False,
            bgcolor="rgba(0,0,0,0)",
            lakecolor="rgba(0,0,0,0)",
        )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{OUTCOME_LABEL.get(outcome, outcome)}</b> — "
                f"sample {sample_n} (n = {n_total})<br>"
                "<span style='font-size:12px;color:#666'>"
                "State-level estimates by model. Color scale fixed to [0%, 100%] for cross-panel comparison."
                "</span>"
            ),
            x=0.01, xanchor="left", y=0.98,
        ),
        margin=dict(l=10, r=80, t=90, b=10),
        width=1500,
        height=850,
        paper_bgcolor="white",
    )

    # Per-panel subtitle styling
    for ann in fig.layout.annotations:
        ann.font = dict(size=13, color="#222")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=2)
    print(f"  wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    failures: list[str] = []
    for outcome in OUTCOMES:
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
            out_path = FIG_DIR / outcome / f"sample{sample_n}_maps.png"
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
