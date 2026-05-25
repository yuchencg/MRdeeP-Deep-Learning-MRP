"""
Baseline (naive disaggregation) — raw state-level means with no modeling.
CES run: sample_ces_1000, state-level estimates.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    OUTPUT_DIR,
    STATE_FIPS_TO_NAME,
    load_survey,
    save_estimates,
)

# ── Configuration ─────────────────────────────────────────────────────────────

OUTCOME_VARS = ["climate_problem", "renewable_fuel"]
MODEL_BASE   = "baseline"

# ── Run for each outcome ──────────────────────────────────────────────────────

for OUTCOME_VAR in OUTCOME_VARS:
    MODEL_NAME = f"{MODEL_BASE}_{OUTCOME_VAR}"

    survey = load_survey(OUTCOME_VAR)

    state_stats = survey.groupby("state_fips").agg(
        estimate=(OUTCOME_VAR, "mean"),
        n_respondents=(OUTCOME_VAR, "count"),
    ).reset_index()

    all_states = pd.DataFrame({
        "state_fips": list(STATE_FIPS_TO_NAME.keys()),
        "state_name": list(STATE_FIPS_TO_NAME.values()),
    })
    estimates = all_states.merge(state_stats, on="state_fips", how="left")
    estimates["n_respondents"] = estimates["n_respondents"].fillna(0).astype(int)

    save_estimates(estimates, MODEL_NAME)

    n_with    = estimates["estimate"].notna().sum()
    n_without = estimates["estimate"].isna().sum()
    valid     = estimates.loc[estimates["estimate"].notna()]

    diag_dir  = OUTPUT_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_path = diag_dir / f"{MODEL_NAME}_summary.txt"

    with open(diag_path, "w") as f:
        f.write(f"Baseline (Naive Disaggregation) — Diagnostic Summary\n")
        f.write(f"{'=' * 55}\n\n")
        f.write(f"Outcome variable: {OUTCOME_VAR}\n")
        f.write(f"Total survey respondents (non-NaN): {len(survey)}\n")
        f.write(f"States with estimates: {n_with}\n")
        f.write(f"States with NaN (no respondents): {n_without}\n\n")
        f.write(f"Estimate range: [{valid['estimate'].min():.4f}, {valid['estimate'].max():.4f}]\n")
        f.write(f"Mean estimate: {valid['estimate'].mean():.4f}\n")
        f.write(f"Median estimate: {valid['estimate'].median():.4f}\n\n")
        f.write(f"Respondents per state:\n")
        f.write(f"  Min: {valid['n_respondents'].min()}\n")
        f.write(f"  Max: {valid['n_respondents'].max()}\n")
        f.write(f"  Mean: {valid['n_respondents'].mean():.1f}\n\n")
        f.write(f"State-level estimates:\n")
        f.write(estimates.to_string(index=False))
        f.write("\n")

    print(f"Diagnostic summary saved → {diag_path}")
