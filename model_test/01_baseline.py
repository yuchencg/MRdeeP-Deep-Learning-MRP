"""
Baseline (naive disaggregation) — raw state-level means with no modeling.
Author: KO
Created: 2026-05-03
"""

import pandas as pd

from models.utils import (
    load_survey,
    save_estimates,
    STATE_FIPS_TO_NAME,
    OUTPUT_DIR,
)

# ── Configuration ────────────────────────────────────────────────────────────

OUTCOME_VAR = "happening_bin"
MODEL_NAME = "baseline"

# ── Load Data ────────────────────────────────────────────────────────────────

survey = load_survey(OUTCOME_VAR)

# ── Compute Raw State Means ──────────────────────────────────────────────────

state_stats = survey.groupby("state_fips").agg(
    estimate=(OUTCOME_VAR, "mean"),
    n_respondents=(OUTCOME_VAR, "count"),
).reset_index()

# ── Ensure All 51 States Present ─────────────────────────────────────────────

all_states = pd.DataFrame({
    "state_fips": list(STATE_FIPS_TO_NAME.keys()),
    "state_name": list(STATE_FIPS_TO_NAME.values()),
})

estimates = all_states.merge(state_stats, on="state_fips", how="left")
estimates["n_respondents"] = estimates["n_respondents"].fillna(0).astype(int)

# ── Save Output ──────────────────────────────────────────────────────────────

save_estimates(estimates, MODEL_NAME)

# ── Diagnostics ──────────────────────────────────────────────────────────────

n_with = estimates["estimate"].notna().sum()
n_without = estimates["estimate"].isna().sum()

print(f"States with estimates: {n_with}")
print(f"States with no respondents (NaN): {n_without}")

if n_without > 0:
    missing = estimates.loc[estimates["estimate"].isna(), "state_name"].tolist()
    print(f"Missing states: {', '.join(missing)}")

valid = estimates.loc[estimates["estimate"].notna()]
print(f"\nEstimate range: [{valid['estimate'].min():.4f}, {valid['estimate'].max():.4f}]")
print(f"Respondents range: [{valid['n_respondents'].min()}, {valid['n_respondents'].max()}]")

# ── Save Diagnostic Summary ──────────────────────────────────────────────────

diag_dir = OUTPUT_DIR / "diagnostics"
diag_dir.mkdir(parents=True, exist_ok=True)
diag_path = diag_dir / f"{MODEL_NAME}_summary.txt"

with open(diag_path, "w") as f:
    f.write(f"Baseline (Naive Disaggregation) — Diagnostic Summary\n")
    f.write(f"{'=' * 55}\n\n")
    f.write(f"Outcome variable: {OUTCOME_VAR}\n")
    f.write(f"Total survey respondents (non-NaN): {len(survey)}\n")
    f.write(f"States with estimates: {n_with}\n")
    f.write(f"States with NaN (no respondents): {n_without}\n\n")

    if n_without > 0:
        f.write(f"Missing states: {', '.join(missing)}\n\n")

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

print(f"\nDiagnostic summary saved → {diag_path}")
