"""Question and Model dropdowns + the back-to-US-view button."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def make_controls(
    outcome_options: list[dict[str, str]],
    initial_outcome: str,
) -> dbc.Container:
    """Build the filter row.

    The model dropdown is left empty here — `app.py` populates it via a
    callback once the user picks an outcome, since the available models
    depend on which CSVs exist for that outcome.
    """
    return dbc.Container(
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Question", htmlFor="outcome-dropdown"),
                        dcc.Dropdown(
                            id="outcome-dropdown",
                            options=outcome_options,
                            value=initial_outcome,
                            clearable=False,
                        ),
                    ],
                    md=5,
                ),
                dbc.Col(
                    [
                        html.Label("Model", htmlFor="model-dropdown"),
                        dcc.Dropdown(
                            id="model-dropdown",
                            options=[],
                            clearable=False,
                        ),
                    ],
                    md=5,
                ),
                dbc.Col(
                    dbc.Button(
                        "← Back to US view",
                        id="reset-zoom-btn",
                        color="secondary",
                        outline=True,
                        size="sm",
                        style={"display": "none"},
                    ),
                    md=2,
                    className="d-flex align-items-end justify-content-end",
                ),
            ],
            className="g-3",
        ),
        fluid=True,
        className="app-controls",
    )
