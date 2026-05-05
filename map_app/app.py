"""Dash entrypoint for the MRP method-comparison map.

Wires the discovered estimate registry, the metadata catalogs (outcomes
and models), and the choropleth builder into a small interactive UI:
question + model dropdowns at the top, a US choropleth in the middle, an
info strip below, and click-to-zoom on a state. Run with `python app.py`.
"""

from __future__ import annotations

import logging

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update

from components.controls import make_controls
from components.header import make_header
from components.info_strip import make_info_strip, render_info_contents
from constants import MODEL_CAVEATS
from data_loader import build_registry, load_estimates, load_models, load_outcomes
from map_builder import build_state_choropleth, postal_to_fips

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("map_app")


# --- Bootstrap data -------------------------------------------------------

REGISTRY = build_registry()
OUTCOMES_DF = load_outcomes()
MODELS_DF = load_models()

OUTCOME_LABELS: dict[str, str] = dict(
    zip(OUTCOMES_DF["outcome_id"], OUTCOMES_DF["label"])
)
MODEL_LABELS: dict[str, str] = dict(zip(MODELS_DF["model_id"], MODELS_DF["label"]))

available_outcomes = REGISTRY.outcomes("state")
if not available_outcomes:
    raise RuntimeError(
        "No state-level estimate CSVs found in data/state/. "
        "Drop in files named <model_id>_<outcome_id>_state.csv to get started."
    )

initial_outcome = available_outcomes[0]
outcome_options = [
    {"label": OUTCOME_LABELS.get(o, o), "value": o} for o in available_outcomes
]


# --- App ------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="MRP Method Comparison Map",
)

app.layout = dbc.Container(
    [
        make_header(),
        make_controls(outcome_options, initial_outcome),
        dcc.Loading(
            dcc.Graph(
                id="map-graph",
                config={"displayModeBar": False, "scrollZoom": False},
                style={"height": "70vh"},
            ),
            type="default",
        ),
        make_info_strip(),
        dcc.Store(id="selected-state-store", data=None),
    ],
    fluid=True,
    className="app-shell",
)


# --- Callbacks ------------------------------------------------------------


@app.callback(
    Output("model-dropdown", "options"),
    Output("model-dropdown", "value"),
    Input("outcome-dropdown", "value"),
    State("model-dropdown", "value"),
)
def update_model_options(outcome_id: str, current_model: str | None):
    """Populate the model dropdown from CSVs available for this outcome."""
    models = REGISTRY.models_for(outcome_id, level="state")
    options = [{"label": MODEL_LABELS.get(m, m), "value": m} for m in models]
    if current_model in models:
        return options, current_model
    return options, (models[0] if models else None)


@app.callback(
    Output("selected-state-store", "data"),
    Input("map-graph", "clickData"),
    Input("reset-zoom-btn", "n_clicks"),
    Input("outcome-dropdown", "value"),
    Input("model-dropdown", "value"),
    State("selected-state-store", "data"),
    prevent_initial_call=True,
)
def update_selected_state(
    click_data, reset_clicks, outcome_id, model_id, current
):
    """Track which state (if any) is currently zoomed in.

    Resets to None whenever the outcome or model changes, or the user
    clicks the back button. A click on a state in the US view zooms in;
    a click while already zoomed leaves the selection alone (the back
    button is the explicit unzoom).
    """
    triggered = dash.callback_context.triggered_id
    if triggered in {"reset-zoom-btn", "outcome-dropdown", "model-dropdown"}:
        return None
    if triggered == "map-graph" and click_data:
        if current is not None:
            return no_update
        try:
            postal = click_data["points"][0]["location"]
        except (KeyError, IndexError, TypeError):
            return no_update
        return postal_to_fips().get(postal)
    return no_update


@app.callback(
    Output("map-graph", "figure"),
    Output("info-strip", "children"),
    Output("reset-zoom-btn", "style"),
    Input("outcome-dropdown", "value"),
    Input("model-dropdown", "value"),
    Input("selected-state-store", "data"),
)
def render_map(outcome_id: str, model_id: str | None, selected_fips: str | None):
    """Re-render the map and info strip whenever the selection changes."""
    if model_id is None:
        empty_fig = {
            "data": [],
            "layout": {
                "annotations": [
                    {
                        "text": "No model files found for this outcome.",
                        "showarrow": False,
                        "x": 0.5,
                        "y": 0.5,
                        "xref": "paper",
                        "yref": "paper",
                    }
                ]
            },
        }
        return empty_fig, [], {"display": "none"}

    path = REGISTRY.state[outcome_id][model_id]
    df = load_estimates(path, level="state")
    outcome_label = OUTCOME_LABELS.get(outcome_id, outcome_id)
    model_label = MODEL_LABELS.get(model_id, model_id)

    fig = build_state_choropleth(
        df,
        outcome_label=outcome_label,
        model_label=model_label,
        selected_state_fips=selected_fips,
    )

    info_children = render_info_contents(
        model_label=model_label,
        outcome_label=outcome_label,
        total_respondents=int(df["n_respondents"].sum()),
        caveat=MODEL_CAVEATS.get(model_id, ""),
    )

    btn_style = {} if selected_fips else {"display": "none"}
    return fig, info_children, btn_style


# --- Main -----------------------------------------------------------------

if __name__ == "__main__":
    logger.info(
        "Discovered %d outcome(s), %d state file(s)",
        len(REGISTRY.outcomes("state")),
        sum(len(v) for v in REGISTRY.state.values()),
    )
    app.run(debug=True, host="127.0.0.1", port=8050)
