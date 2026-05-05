"""Project-wide constants for the MRP map app.

Centralizing these values keeps file-naming, color choices, and validation
rules consistent across the data loader, map builder, and Dash callbacks.
"""

from __future__ import annotations

import re
from pathlib import Path

# --- Paths ----------------------------------------------------------------

APP_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = APP_DIR / "data"
STATE_DIR: Path = DATA_DIR / "state"
COUNTY_DIR: Path = DATA_DIR / "county"
OUTCOMES_FILE: Path = DATA_DIR / "outcomes.csv"
MODELS_FILE: Path = DATA_DIR / "models.csv"

# --- File-naming convention ----------------------------------------------
# Filenames look like: <model_id>_<outcome_id>_state.csv
#                       (or)  <model_id>_<outcome_id>_county.csv
# Both model_id and outcome_id are lowercase snake_case. Because both can
# contain underscores (e.g. "glm_mrp" + "happening_bin"), regex alone can't
# disambiguate them — the loader splits the stem against the known
# model_id catalog from models.csv.

STATE_SUFFIX: str = "_state.csv"
COUNTY_SUFFIX: str = "_county.csv"
FILENAME_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")

# --- Schema ---------------------------------------------------------------

REQUIRED_STATE_COLUMNS: tuple[str, ...] = (
    "state_fips",
    "state_name",
    "estimate",
    "n_respondents",
)
REQUIRED_COUNTY_COLUMNS: tuple[str, ...] = (
    "county_fips",
    "county_name",
    "estimate",
    "n_respondents",
)

STATE_FIPS_WIDTH: int = 2
COUNTY_FIPS_WIDTH: int = 5

# --- Visual ---------------------------------------------------------------

COLOR_SCALE: str = "RdBu_r"  # diverging blue (low) ↔ red (high)
COLOR_RANGE: tuple[float, float] = (0.0, 1.0)
COLOR_MIDPOINT: float = 0.5
COLOR_BAR_TICKS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
COLOR_BAR_TICK_LABELS: tuple[str, ...] = ("0%", "25%", "50%", "75%", "100%")

# --- Per-model caveats (shown in info strip) -----------------------------
# Empty string means "no caveat". Keys must match model_id values in
# models.csv. Adding a new model with no caveat requires no change here.

MODEL_CAVEATS: dict[str, str] = {
    "baseline": (
        "Raw survey means; no modeling — small-state estimates are noisy."
    ),
    "glm_mrp": "",
    "glmer_mrp": "",
    "stan_mrp": "",
    "srp": "",
    "mrdeep": "",
}

# --- External links ------------------------------------------------------

YCOM_URL: str = "https://climatecommunication.yale.edu/visualizations-data/ycom-us/"
