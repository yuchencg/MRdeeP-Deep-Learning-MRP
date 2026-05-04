"""
Shared utilities for MRP model comparison — data loading, poststratification, and output.
Author: KO
Created: 2026-05-03
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "test_data"
OUTPUT_DIR = BASE_DIR / "outputs"

SURVEY_PATH = DATA_DIR / "processed" / "climate_survey_responses_recoded.csv"
POSTSTRAT_STATE_PATH = DATA_DIR / "processed" / "poststrat_state.csv"
CARBON_PATH = DATA_DIR / "raw" / "carbon_state.csv"
PRES_PATH = DATA_DIR / "raw" / "pres_state.csv"
DRIVE_PATH = DATA_DIR / "raw" / "drive_state.csv"
SAMESEX_PATH = DATA_DIR / "raw" / "samesex_state.csv"

# ── State FIPS → Census Division Mapping ─────────────────────────────────────

STATE_FIPS_TO_REGION = {
    "01": "E. South Central",
    "02": "Pacific",
    "04": "Mountain",
    "05": "W. South Central",
    "06": "Pacific",
    "08": "Mountain",
    "09": "New England",
    "10": "South Atlantic",
    "11": "South Atlantic",
    "12": "South Atlantic",
    "13": "South Atlantic",
    "15": "Pacific",
    "16": "Mountain",
    "17": "E. North Central",
    "18": "E. North Central",
    "19": "W. North Central",
    "20": "W. North Central",
    "21": "E. South Central",
    "22": "W. South Central",
    "23": "New England",
    "24": "South Atlantic",
    "25": "New England",
    "26": "E. North Central",
    "27": "W. North Central",
    "28": "E. South Central",
    "29": "W. North Central",
    "30": "Mountain",
    "31": "W. North Central",
    "32": "Mountain",
    "33": "New England",
    "34": "Mid-Atlantic",
    "35": "Mountain",
    "36": "Mid-Atlantic",
    "37": "South Atlantic",
    "38": "W. North Central",
    "39": "E. North Central",
    "40": "W. South Central",
    "41": "Pacific",
    "42": "Mid-Atlantic",
    "44": "New England",
    "45": "South Atlantic",
    "46": "W. North Central",
    "47": "E. South Central",
    "48": "W. South Central",
    "49": "Mountain",
    "50": "New England",
    "51": "South Atlantic",
    "53": "Pacific",
    "54": "South Atlantic",
    "55": "E. North Central",
    "56": "Mountain",
}

STATE_FIPS_TO_NAME = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut",
    "10": "Delaware", "11": "District of Columbia", "12": "Florida",
    "13": "Georgia", "15": "Hawaii", "16": "Idaho", "17": "Illinois",
    "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky",
    "22": "Louisiana", "23": "Maine", "24": "Maryland",
    "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana",
    "31": "Nebraska", "32": "Nevada", "33": "New Hampshire",
    "34": "New Jersey", "35": "New Mexico", "36": "New York",
    "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania",
    "44": "Rhode Island", "45": "South Carolina", "46": "South Dakota",
    "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont",
    "51": "Virginia", "53": "Washington", "54": "West Virginia",
    "55": "Wisconsin", "56": "Wyoming",
}

# ── State-Level Covariates ───────────────────────────────────────────────────

def _load_state_covariates():
    carbon = pd.read_csv(CARBON_PATH, dtype={"state_fips": str})
    carbon = carbon[["state_fips", "co2_per_capita"]]

    pres = pd.read_csv(PRES_PATH, dtype={"state_fips": str})
    pres = pres[["state_fips", "dem_share_two_party"]]

    drive = pd.read_csv(DRIVE_PATH, dtype={"state_fips": str})
    drive = drive[["state_fips", "drive_alone_share"]]

    samesex = pd.read_csv(SAMESEX_PATH, dtype={"state_fips": str})
    samesex = samesex[["state_fips", "samesex_share"]]

    covariates = (
        carbon
        .merge(pres, on="state_fips", how="outer")
        .merge(drive, on="state_fips", how="outer")
        .merge(samesex, on="state_fips", how="outer")
    )
    return covariates

# ── Public API ───────────────────────────────────────────────────────────────

def load_survey(outcome_var):
    """Load recoded survey, drop NaN on outcome_var, merge state covariates and region9."""
    df = pd.read_csv(SURVEY_PATH, dtype={"state_fips": str})
    df = df.dropna(subset=[outcome_var])

    covariates = _load_state_covariates()
    df = df.merge(covariates, on="state_fips", how="left")

    if "region9" not in df.columns:
        df["region9"] = df["state_fips"].map(STATE_FIPS_TO_REGION)

    return df


def load_poststrat_state():
    """Load state poststrat frame with covariates and region9."""
    df = pd.read_csv(POSTSTRAT_STATE_PATH, dtype={"state_fips": str})
    df = df[["state_fips", "gender", "race4", "educ_category", "N", "region9"]]

    covariates = _load_state_covariates()
    df = df.merge(covariates, on="state_fips", how="left")

    return df


def poststratify(poststrat_df, prob_col):
    """Compute population-weighted state-level estimates from cell-level probabilities."""
    grouped = poststrat_df.groupby("state_fips").apply(
        lambda g: np.average(g[prob_col], weights=g["N"])
    ).reset_index(name="estimate")

    grouped["state_name"] = grouped["state_fips"].map(STATE_FIPS_TO_NAME)
    return grouped[["state_fips", "state_name", "estimate"]]


def save_estimates(estimates_df, model_name):
    """Save state-level estimates CSV and print console summary."""
    out_dir = OUTPUT_DIR / "estimates"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_state_estimates.csv"

    estimates_df.to_csv(out_path, index=False)

    n_states = len(estimates_df)
    n_valid = estimates_df["estimate"].notna().sum()
    n_nan = estimates_df["estimate"].isna().sum()

    print(f"\n{'=' * 60}")
    print(f"  {model_name} — State-Level Estimates")
    print(f"{'=' * 60}")
    print(f"  States with estimates: {n_valid}")
    print(f"  States with NaN:       {n_nan}")

    if n_valid > 0:
        valid = estimates_df["estimate"].dropna()
        print(f"  Mean estimate:         {valid.mean():.4f}")
        print(f"  Median estimate:       {valid.median():.4f}")
        print(f"  Min estimate:          {valid.min():.4f}")
        print(f"  Max estimate:          {valid.max():.4f}")

    if "n_respondents" in estimates_df.columns:
        resp = estimates_df["n_respondents"].dropna()
        print(f"  Min n_respondents:     {resp.min():.0f}")
        print(f"  Max n_respondents:     {resp.max():.0f}")

    print(f"\n  Saved → {out_path}")
    print(f"{'=' * 60}\n")
