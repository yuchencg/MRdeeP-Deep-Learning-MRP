"""Pure Plotly figure construction for the MRP choropleth.

This module is intentionally Dash-free so the figure logic can be unit
tested or reused (e.g., for static export) without spinning up a Dash app.
All visual constants come from `constants.py`.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from constants import (
    COLOR_BAR_TICK_LABELS,
    COLOR_BAR_TICKS,
    COLOR_RANGE,
    COLOR_SCALE,
)


def build_state_choropleth(
    df: pd.DataFrame,
    outcome_label: str,
    model_label: str,
    selected_state_fips: str | None = None,
) -> go.Figure:
    """Build the US state-level choropleth figure.

    Renders all 50 states + DC colored by `estimate` on a fixed [0, 1]
    scale so different model selections stay visually comparable. When
    `selected_state_fips` is provided, the figure is filtered to that
    single state and Plotly auto-fits the geo bounds — this powers the
    click-to-zoom interaction. Hover text shows state name, estimate as a
    percentage, and respondent count. Returns a Plotly Figure; callers
    embed it in a `dcc.Graph`.
    """
    plot_df = df.copy()
    if selected_state_fips is not None:
        plot_df = plot_df[plot_df["state_fips"] == selected_state_fips]

    customdata = plot_df[["state_name", "n_respondents"]].to_numpy()
    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        "Estimate: %{z:.1%}<br>"
        "Respondents: %{customdata[1]}"
        "<extra></extra>"
    )

    trace = go.Choropleth(
        locations=plot_df["state_fips"],
        z=plot_df["estimate"],
        locationmode="USA-states",
        geojson=None,  # USA-states uses Plotly's built-in geometry
        featureidkey=None,
        zmin=COLOR_RANGE[0],
        zmax=COLOR_RANGE[1],
        zmid=0.5,
        colorscale=COLOR_SCALE,
        customdata=customdata,
        hovertemplate=hovertemplate,
        marker_line_color="white",
        marker_line_width=0.5,
        colorbar=dict(
            title=dict(text="P(yes)", side="right"),
            tickmode="array",
            tickvals=list(COLOR_BAR_TICKS),
            ticktext=list(COLOR_BAR_TICK_LABELS),
            thickness=14,
            len=0.75,
        ),
    )

    # USA-states only resolves 2-letter postal codes natively; we have
    # FIPS, so drop into the GeoJSON-free 'choropleth' geo trace by
    # converting locations via Plotly's built-in feature ids. The cleanest
    # path is to use locationmode="USA-states" with state postal codes —
    # so we map FIPS -> postal here.
    fips_to_postal = _fips_to_postal()
    trace.locations = [fips_to_postal.get(f, f) for f in plot_df["state_fips"]]

    fig = go.Figure(data=[trace])

    geo_layout: dict = dict(
        scope="usa",
        projection_type="albers usa",
        showlakes=True,
        lakecolor="rgb(255, 255, 255)",
        bgcolor="rgba(0,0,0,0)",
    )
    if selected_state_fips is not None:
        geo_layout["fitbounds"] = "locations"
        geo_layout["visible"] = True

    fig.update_layout(
        geo=geo_layout,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(
            text=f"<b>{outcome_label}</b> — {model_label}",
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _fips_to_postal() -> dict[str, str]:
    """Return a mapping of 2-digit state FIPS -> 2-letter postal abbreviation.

    Plotly's `USA-states` locationmode draws state polygons keyed by postal
    code, so we translate FIPS (the project's canonical state ID) at the
    figure boundary. Includes DC.
    """
    return {
        "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
        "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
        "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
        "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
        "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
        "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
        "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
        "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
        "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
        "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
        "56": "WY",
    }


def postal_to_fips() -> dict[str, str]:
    """Inverse of `_fips_to_postal` — exposed for callbacks that receive
    postal codes from Plotly click events and need to look up the FIPS row.
    """
    return {v: k for k, v in _fips_to_postal().items()}
