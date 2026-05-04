# Prompt 3 — GLMER-MRP (Mixed-Effects Logistic Regression + Poststratification)

Read `CLAUDE.md` for full project context. Read `models/utils.py` before writing — import and use its functions.

## Task

Create `models/03_glmer_mrp.py` — a frequentist generalized linear mixed model (GLMM) via `gpboost`, with random intercepts on all demographic and geographic grouping variables, Howe-style state-level covariates as fixed effects, and poststratification to state-level estimates.

### What This Script Does

1. **Load data** via `utils.load_survey("happening_bin")` and `utils.load_poststrat_state()`

2. **Prepare the random effects grouping matrix** (`group_data`):
   - A numpy array or DataFrame with 5 columns, one per random-effect grouping variable
   - Columns in order: `race4`, `educ_category`, `gender`, `region9`, `state_fips`
   - Each column contains the categorical group label for each respondent
   - These are the `(1|race4) + (1|educ_category) + (1|gender) + (1|region9) + (1|state_fips)` terms

3. **Prepare the fixed effects design matrix** (`X`):
   - Column 0: intercept (vector of ones)
   - Column 1: `co2_per_capita` (continuous)
   - Column 2: `dem_share_two_party` (continuous)
   - Use `np.column_stack`

4. **Fit the GLMM** using `gpboost`:
   ```python
   import gpboost as gpb

   gp_model = gpb.GPModel(group_data=group_data, likelihood="bernoulli_logit")
   gp_model.fit(y=y, X=X)
   ```
   - `y`: numpy array of `happening_bin` values (0/1, dtype float64)
   - `likelihood="bernoulli_logit"`: Bernoulli with logit link (NOT the default probit)
   - This estimates random intercept variances for each grouping variable and fixed-effect coefficients for the intercept + covariates, via Laplace-approximated maximum likelihood

5. **Extract model summary**:
   - `gp_model.summary()` — prints variance components and fixed effects
   - `gp_model.get_cov_pars()` — random effect variance estimates
   - `gp_model.get_coef()` — fixed effect coefficients

6. **Predict** P(happening=1) for every row in the poststrat frame:
   ```python
   # Prepare poststrat group_data and X in the same format as training
   pred = gp_model.predict(
       X_pred=X_poststrat,
       group_data_pred=group_data_poststrat,
       predict_var=False,
       predict_response=True  # returns probabilities, not latent scale
   )
   poststrat_df["predicted_prob"] = pred["mu"]
   ```
   - `group_data_pred`: same 5-column structure as training, using poststrat frame values
   - `X_pred`: intercept + covariates for each poststrat cell
   - `predict_response=True` is critical — it applies the inverse logit to get probabilities

7. **Poststratify** via `utils.poststratify(poststrat_df, "predicted_prob")`

8. **Compute n_respondents** per state from the survey data

9. **Save estimates** via `utils.save_estimates()` with model_name `"glmer_mrp"`

10. **Save diagnostics** — capture and save:
    - Variance components for each random effect (race4, educ_category, gender, region9, state_fips)
    - Fixed effect coefficients (intercept, co2_per_capita, dem_share_two_party) with standard errors
    - Number of observations
    - Convergence status
    - State estimate summary stats (mean, median, min, max, std)
    - Comparison note: number of states where GLMER estimate differs from baseline by >10pp

### Key Implementation Notes

- **Data types matter for gpboost**: `group_data` columns should be string or integer labels. `y` must be float64 (not int). `X` must be float64 numpy array.
- **New levels in prediction**: The poststrat frame contains all 51 states, but the survey may not. gpboost handles unseen group levels by predicting with the random effect set to zero (the prior mean) — this is the partial pooling behavior we want. States with no survey respondents get estimates driven entirely by covariates + region.
- **Do NOT use `likelihood="binary"`** — that defaults to probit link. Use `"bernoulli_logit"` explicitly for logit link to match R's `glmer(..., family=binomial(link="logit"))`.
- **Convergence**: If the optimizer doesn't converge, try `gp_model.set_optim_params(params={"maxit": 1000})` before fitting. Log convergence info in diagnostics.

## Also Create

- `docs/03_glmer_mrp_explained.md` — Full explanation following the format in CLAUDE.md § "Explanation File Format". Include all 5 sections.
  - In **Package Selection & Design Choices**: explain in detail:
    - Why `gpboost` was chosen: pure Python (C++ backend), Laplace approximation matching `lme4::glmer()`, supports `bernoulli_logit` likelihood, handles crossed/nested random effects, benchmarked >100x faster than `lme4` in some cases
    - Alternatives considered and rejected: `pymer4` (requires R), `statsmodels.BinomialBayesMixedGLM` (variational Bayes, not true frequentist, known accuracy issues), `bambi`/`pymc` (Bayesian MCMC — reserved for the Stan-MRP model to preserve the frequentist vs. Bayesian comparison)
    - How gpboost's Laplace approximation works at a high level: approximates the marginal likelihood by a second-order Taylor expansion around the mode of the random effects, then optimizes variance components via gradient-based methods
    - The `bernoulli_logit` vs `binary` (probit) distinction and why logit was chosen (matches R's `glmer` default and Howe et al.)
  - In **Connection to Howe et al. (2015)**: explain that this is the partial pooling model — the core of MRP. State estimates for sparse-data states are shrunk toward the covariate-predicted value. Compare to Baseline (no pooling on geography) and GLM-MRP (no pooling on demographics). Note departures from the full Howe spec: we use crossed random effects rather than nested `(1|region/state)` — explain why this is acceptable (gpboost estimates separate variance components; the covariates provide the informative shrinkage target that nesting would otherwise give).

## Conventions

- Author docstring (Author: KO, today's date)
- Minimal inline comments
- `state_fips` always zero-padded string
