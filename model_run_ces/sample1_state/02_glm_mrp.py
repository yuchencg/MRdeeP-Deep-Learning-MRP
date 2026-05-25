"""
GLM-MRP: frequentist logistic regression with all fixed effects, poststratified
to state-level estimates.
CES run: sample_ces_1000, state-level estimates.
"""

import sys
import warnings
from pathlib import Path

import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    OUTPUT_DIR,
    load_poststrat_state,
    load_survey,
    poststratify,
    save_estimates,
)

# ── Configuration ─────────────────────────────────────────────────────────────

OUTCOME_VARS  = ["climate_problem", "renewable_fuel"]
MODEL_BASE    = "glm_mrp"
CATEGORICALS  = ["gender", "race4", "educ_category", "region9"]
COVARIATES    = "co2_per_capita + dem_share_two_party + drive_alone_share + samesex_share"

# ── Load poststrat frame once (shared across outcomes) ────────────────────────

poststrat = load_poststrat_state()
for col in CATEGORICALS:
    poststrat[col] = poststrat[col].astype(str)

# ── Run for each outcome ──────────────────────────────────────────────────────

for OUTCOME_VAR in OUTCOME_VARS:
    MODEL_NAME = f"{MODEL_BASE}_{OUTCOME_VAR}"
    FORMULA    = (
        f"{OUTCOME_VAR} ~ C(gender) + C(race4) + C(educ_category) + C(region9)"
        f" + {COVARIATES}"
    )

    survey = load_survey(OUTCOME_VAR)
    for col in CATEGORICALS:
        survey[col] = survey[col].astype(str)

    caught_warnings = []
    with warnings.catch_warnings(record=True) as _w:
        warnings.simplefilter("always")
        model = smf.glm(
            formula=FORMULA,
            data=survey,
            family=sm.families.Binomial(link=sm.families.links.Logit()),
        ).fit()
        caught_warnings = list(_w)

    poststrat["predicted_prob"] = model.predict(poststrat)
    estimates = poststratify(poststrat, "predicted_prob")

    n_resp = (
        survey.groupby("state_fips")[OUTCOME_VAR]
        .count()
        .reset_index(name="n_respondents")
    )
    estimates = estimates.merge(n_resp, on="state_fips", how="left")
    estimates["n_respondents"] = estimates["n_respondents"].fillna(0).astype(int)

    save_estimates(estimates, MODEL_NAME)

    diag_dir  = OUTPUT_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_path = diag_dir / f"{MODEL_NAME}_summary.txt"
    valid_estimates = estimates["estimate"].dropna()

    with open(diag_path, "w") as f:
        f.write(f"GLM-MRP — Diagnostic Summary\n")
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

    print(f"Diagnostic summary saved → {diag_path}")
