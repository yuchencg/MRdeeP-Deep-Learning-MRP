"""
Weighted state-level means on all CES data.
Uses survey weights (commonweight) for population-representative estimates.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

SURVEY_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "CES" / "data_processed" / "filtered_responses_preprocessed.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# ── State FIPS → Name ────────────────────────────────────────────────────────

STATE_FIPS_TO_NAME = {
    "01": "Alabama",            "02": "Alaska",         "04": "Arizona",
    "05": "Arkansas",           "06": "California",     "08": "Colorado",
    "09": "Connecticut",        "10": "Delaware",       "11": "District of Columbia",
    "12": "Florida",            "13": "Georgia",        "15": "Hawaii",
    "16": "Idaho",              "17": "Illinois",       "18": "Indiana",
    "19": "Iowa",               "20": "Kansas",         "21": "Kentucky",
    "22": "Louisiana",          "23": "Maine",          "24": "Maryland",
    "25": "Massachusetts",      "26": "Michigan",       "27": "Minnesota",
    "28": "Mississippi",        "29": "Missouri",       "30": "Montana",
    "31": "Nebraska",           "32": "Nevada",         "33": "New Hampshire",
    "34": "New Jersey",         "35": "New Mexico",     "36": "New York",
    "37": "North Carolina",     "38": "North Dakota",   "39": "Ohio",
    "40": "Oklahoma",           "41": "Oregon",         "42": "Pennsylvania",
    "44": "Rhode Island",       "45": "South Carolina", "46": "South Dakota",
    "47": "Tennessee",          "48": "Texas",          "49": "Utah",
    "50": "Vermont",            "51": "Virginia",       "53": "Washington",
    "54": "West Virginia",      "55": "Wisconsin",      "56": "Wyoming",
}

# ── Configuration ────────────────────────────────────────────────────────────

OUTCOMES = ["climate_problem", "renewable_fuel"]
MODEL_NAME = "weighted_mean"

# ── Run for each outcome ─────────────────────────────────────────────────────

survey_full = pd.read_csv(SURVEY_PATH, dtype={"state_fips": str})

for outcome_var in OUTCOMES:
    survey = survey_full.dropna(subset=[outcome_var, "commonweight"])

    # Weighted mean per state
    state_stats = (
        survey.groupby("state_fips")
        .apply(lambda g: pd.Series({
            "estimate": np.average(g[outcome_var], weights=g["commonweight"]),
            "n_respondents": len(g),
        }), include_groups=False)
        .reset_index()
    )
    state_stats["n_respondents"] = state_stats["n_respondents"].astype(int)

    # Join with full state list
    all_states = pd.DataFrame({
        "state_fips": list(STATE_FIPS_TO_NAME.keys()),
        "state_name": list(STATE_FIPS_TO_NAME.values()),
    })
    estimates = all_states.merge(state_stats, on="state_fips", how="left")
    estimates["n_respondents"] = estimates["n_respondents"].fillna(0).astype(int)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{outcome_var}_state_estimates.csv"
    estimates.to_csv(out_path, index=False)

    n_valid = estimates["estimate"].notna().sum()
    n_nan = estimates["estimate"].isna().sum()
    valid = estimates.loc[estimates["estimate"].notna()]

    print(f"\n{'=' * 60}")
    print(f"  {MODEL_NAME} ({outcome_var}) — State-Level Estimates")
    print(f"{'=' * 60}")
    print(f"  States with estimates: {n_valid}")
    print(f"  States with NaN:       {n_nan}")
    if n_valid > 0:
        print(f"  Mean estimate:         {valid['estimate'].mean():.4f}")
        print(f"  Min estimate:          {valid['estimate'].min():.4f}")
        print(f"  Max estimate:          {valid['estimate'].max():.4f}")
    print(f"  Saved → {out_path}")
    print(f"{'=' * 60}")
