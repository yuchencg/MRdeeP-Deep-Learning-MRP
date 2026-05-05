"""
GLM-MRP: frequentist logistic regression with all fixed effects, poststratified to state-level estimates.
Author: KO
Created: 2026-05-03
"""

import warnings

import statsmodels.api as sm
import statsmodels.formula.api as smf

from models.utils import (
    OUTPUT_DIR,
    load_poststrat_state,
    load_survey,
    poststratify,
    save_estimates,
)

# ── Configuration ─────────────────────────────────────────────────────────────

OUTCOME_VAR = "happening_bin"
MODEL_NAME = "glm_mrp"
FORMULA = (
    "happening_bin ~ C(gender) + C(race4) + C(educ_category) + C(region9)"
    " + co2_per_capita + dem_share_two_party + drive_alone_share + samesex_share"
)
CATEGORICALS = ["gender", "race4", "educ_category", "region9"]

# ── Load Data ─────────────────────────────────────────────────────────────────

survey = load_survey(OUTCOME_VAR)
poststrat = load_poststrat_state()

# ── Cast Categoricals to String ───────────────────────────────────────────────

for col in CATEGORICALS:
    survey[col] = survey[col].astype(str)
    poststrat[col] = poststrat[col].astype(str)

# ── Fit GLM ───────────────────────────────────────────────────────────────────

caught_warnings = []
with warnings.catch_warnings(record=True) as _w:
    warnings.simplefilter("always")
    model = smf.glm(
        formula=FORMULA,
        data=survey,
        family=sm.families.Binomial(link=sm.families.links.Logit()),
    ).fit()
    caught_warnings = list(_w)

# ── Predict on Poststrat Frame ────────────────────────────────────────────────

poststrat["predicted_prob"] = model.predict(poststrat)

# ── Poststratify ──────────────────────────────────────────────────────────────

estimates = poststratify(poststrat, "predicted_prob")

# ── Compute n_respondents Per State ──────────────────────────────────────────

n_resp = (
    survey.groupby("state_fips")[OUTCOME_VAR]
    .count()
    .reset_index(name="n_respondents")
)
estimates = estimates.merge(n_resp, on="state_fips", how="left")
estimates["n_respondents"] = estimates["n_respondents"].fillna(0).astype(int)

# ── Save Estimates ────────────────────────────────────────────────────────────

save_estimates(estimates, MODEL_NAME)

# ── Save Diagnostics ──────────────────────────────────────────────────────────

diag_dir = OUTPUT_DIR / "diagnostics"
diag_dir.mkdir(parents=True, exist_ok=True)
diag_path = diag_dir / f"{MODEL_NAME}_summary.txt"

valid_estimates = estimates["estimate"].dropna()

with open(diag_path, "w") as f:
    f.write("GLM-MRP — Diagnostic Summary\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Outcome variable: {OUTCOME_VAR}\n")
    f.write(f"Formula: {FORMULA}\n")
    f.write(f"Observations: {int(model.nobs)}\n")
    pseudo_r2 = 1 - model.llf / model.llnull
    f.write(f"Pseudo R-squared (McFadden): {pseudo_r2:.4f}\n\n")

    if caught_warnings:
        f.write("Convergence warnings:\n")
        for w in caught_warnings:
            f.write(f"  {w.category.__name__}: {w.message}\n")
        f.write("\n")
    else:
        f.write("Convergence warnings: none\n\n")

    f.write("Model Summary:\n")
    f.write(str(model.summary()))
    f.write("\n\nState Estimate Summary:\n")
    f.write(f"  Mean:   {valid_estimates.mean():.4f}\n")
    f.write(f"  Median: {valid_estimates.median():.4f}\n")
    f.write(f"  Min:    {valid_estimates.min():.4f}\n")
    f.write(f"  Max:    {valid_estimates.max():.4f}\n")
    f.write(f"  Std:    {valid_estimates.std():.4f}\n")

print(f"\nDiagnostic summary saved → {diag_path}")
