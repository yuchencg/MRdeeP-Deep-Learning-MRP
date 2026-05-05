"""Page header with title, subtitle, and YCOM reference link."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from constants import YCOM_URL


def make_header() -> dbc.Container:
    """Return the top-of-page header. Static; no inputs from app state."""
    return dbc.Container(
        [
            html.H2("MRP Method Comparison Map", className="app-title"),
            html.P(
                "State-level public-opinion estimates from a climate survey, "
                "rendered side-by-side across modeling methods.",
                className="app-subtitle",
            ),
            html.P(
                [
                    "Inspired by the ",
                    html.A(
                        "Yale Climate Opinion Maps (YCOM)",
                        href=YCOM_URL,
                        target="_blank",
                        rel="noopener noreferrer",
                    ),
                    ".",
                ],
                className="app-context-link",
            ),
        ],
        fluid=True,
        className="app-header",
    )
