"""
GLMER-MRP: frequentist GLMM via gpboost (Laplace approximation, bernoulli_logit),
with random intercepts on demographics + geography and poststratification to states.
CES run: sample_ces_1000, state-level estimates.
"""

import sys
import warnings
from pathlib import Path

import gpboost as gpb
import numpy as np

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
MODEL_BASE    = "glmer_mrp"
GROUPING_VARS = ["race4", "educ_category", "gender", "region9", "state_fips"]
COVARIATES    = [
    "co2_per_capita",
    "dem_share_two_party",
    "drive_alone_share",
    "samesex_share",
]

# ── Load poststrat frame once ─────────────────────────────────────────────────

poststrat = load_poststrat_state()
for col in GROUPING_VARS:
    poststrat[col] = poststrat[col].astype(str)

group_data_pred = poststrat[GROUPING_VARS].to_numpy()
X_pred = np.column_stack([
    np.ones(len(poststrat)),
    *[poststrat[c].to_numpy(dtype=np.float64) for c in COVARIATES],
])

# ── Run for each outcome ──────────────────────────────────────────────────────

for OUTCOME_VAR in OUTCOME_VARS:
    MODEL_NAME = f"{MODEL_BASE}_{OUTCOME_VAR}"

    survey = load_survey(OUTCOME_VAR)
    for col in GROUPING_VARS:
        survey[col] = survey[col].astype(str)

    y          = survey[OUTCOME_VAR].to_numpy(dtype=np.float64)
    group_data = survey[GROUPING_VARS].to_numpy()
    X = np.column_stack([
        np.ones(len(survey)),
        *[survey[c].to_numpy(dtype=np.float64) for c in COVARIATES],
    ])

    caught_warnings = []
    with warnings.catch_warnings(record=True) as _w:
        warnings.simplefilter("always")
        gp_model = gpb.GPModel(group_data=group_data, likelihood="bernoulli_logit")
        gp_model.set_optim_params(params={"maxit": 1000})
        gp_model.fit(y=y, X=X)
        caught_warnings = list(_w)

    cov_pars = gp_model.get_cov_pars()
    coef     = gp_model.get_coef()

    pred = gp_model.predict(
        X_pred=X_pred,
        group_data_pred=group_data_pred,
        predict_var=False,
        predict_response=True,
    )
    poststrat["predicted_prob"] = pred["mu"]

    estimates = poststratify(poststrat, "predicted_prob")

    n_resp = (
        survey.groupby("state_fips")[OUTCOME_VAR]
        .count()
        .reset_index(name="n_respondents")
    )
    estimates = estimates.merge(n_resp, on="state_fips", how="left")
    estimates["n_respondents"] = estimates["n_respondents"].fillna(0).astype(int)

    save_estimates(estimates, MODEL_NAME)

    # compare against baseline if available
    import pandas as pd
    baseline_path = OUTPUT_DIR / "estimates" / f"baseline_{OUTCOME_VAR}_state_estimates.csv"
    diff_count = None
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path, dtype={"state_fips": str})
        merged = estimates.merge(
            baseline[["state_fips", "estimate"]].rename(columns={"estimate": "baseline_estimate"}),
            on="state_fips", how="left",
        )
        merged["abs_diff"] = (merged["estimate"] - merged["baseline_estimate"]).abs()
        diff_count = int((merged["abs_diff"] > 0.10).sum())

    diag_dir  = OUTPUT_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_path = diag_dir / f"{MODEL_NAME}_summary.txt"
    valid_estimates = estimates["estimate"].dropna()

    with open(diag_path, "w") as f:
        f.write("GLMER-MRP — Diagnostic Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Outcome variable: {OUTCOME_VAR}\n")
        f.write(f"Likelihood: bernoulli_logit (Laplace approximation)\n")
        f.write(f"Grouping variables: {GROUPING_VARS}\n")
        f.write(f"Fixed effects: intercept + {COVARIATES}\n")
        f.write(f"Observations: {len(y)}\n\n")
        if caught_warnings:
            f.write("Convergence warnings:\n")
            for w in caught_warnings:
                f.write(f"  {w.category.__name__}: {w.message}\n")
            f.write("\n")
        else:
            f.write("Convergence warnings: none\n\n")
        f.write("Random Effect Variance Components:\n")
        try:
            cov_arr = np.asarray(cov_pars).flatten()
            for name, val in zip(GROUPING_VARS, cov_arr):
                f.write(f"  {name:20s}  variance = {float(val):.6f}\n")
        except Exception as e:
            f.write(f"  (raw) {cov_pars}\n  [parse note: {e}]\n")
        f.write("\nFixed Effect Coefficients:\n")
        try:
            coef_arr = np.asarray(coef).flatten()
            names    = ["intercept"] + COVARIATES
            for name, val in zip(names, coef_arr):
                f.write(f"  {name:20s}  estimate = {float(val):.6f}\n")
        except Exception as e:
            f.write(f"  (raw) {coef}\n  [parse note: {e}]\n")
        f.write("\n")
        import io, contextlib
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                gp_model.summary()
            f.write("Full gpboost summary:\n")
            f.write(buf.getvalue())
        except Exception as e:
            f.write(f"  [summary capture failed: {e}]\n")
        f.write("\nState Estimate Summary:\n")
        f.write(f"  Mean:   {valid_estimates.mean():.4f}\n")
        f.write(f"  Median: {valid_estimates.median():.4f}\n")
        f.write(f"  Min:    {valid_estimates.min():.4f}\n")
        f.write(f"  Max:    {valid_estimates.max():.4f}\n")
        f.write(f"  Std:    {valid_estimates.std():.4f}\n\n")
        if diff_count is not None:
            f.write(f"Comparison vs. Baseline:\n")
            f.write(f"  States where |GLMER − Baseline| > 0.10: {diff_count} / 51\n")

    print(f"Diagnostic summary saved → {diag_path}")
