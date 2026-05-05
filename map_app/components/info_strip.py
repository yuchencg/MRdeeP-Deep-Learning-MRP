"""Footer-style info strip summarizing the current view."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def make_info_strip() -> dbc.Container:
    """Return an empty info strip; contents are filled by callback in app.py."""
    return dbc.Container(
        html.Div(id="info-strip", className="info-strip"),
        fluid=True,
    )


def render_info_contents(
    model_label: str,
    outcome_label: str,
    total_respondents: int,
    caveat: str,
) -> list:
    """Render the children of the info strip for a given selection.

    Returns Dash children (not a full container) so the parent callback
    can swap them in via `Output("info-strip", "children")`.
    """
    parts = [
        html.Span([html.Strong("Model: "), model_label], className="info-item"),
        html.Span(
            [html.Strong("Outcome: "), outcome_label], className="info-item"
        ),
        html.Span(
            [
                html.Strong("Total respondents: "),
                f"{total_respondents:,}",
            ],
            className="info-item",
        ),
    ]
    if caveat:
        parts.append(html.Span(caveat, className="info-caveat"))
    return parts
