# 02 GLM-MRP (Logistic Regression + Poststratification) — Explained

## 1. Big Picture: What This Script Accomplishes

The GLM-MRP model produces state-level estimates of public opinion by fitting a frequentist logistic regression on the national survey and then poststratifying the predictions to each state's demographic composition. Unlike the Baseline, which simply tabulates raw survey means, this model adjusts for demographic and geographic composition: it learns the relationship between respondent characteristics and climate opinion nationally, then applies those learned relationships to each state's known population structure. The result is a set of 51 state-level estimates that reflect what each state's population *would* say if it had the same opinion-demographic relationship as the national survey, weighted by that state's actual demographics.

1. Load the recoded survey data (dropping don't-know responses on `happening_bin`) and the ACS-based poststratification frame with state-level covariates.
2. Cast all categorical predictors to string type so `statsmodels` treats them as discrete factors, not continuous integers.
3. Fit a logistic GLM (Binomial family, logit link) with demographic and geographic variables as categorical dummies and all four Howe et al. (2015) state-level covariates as continuous fixed effects.
4. Predict P(happening = 1) for every cell in the 1,632-row poststratification frame.
5. Compute the population-weighted average of cell-level predictions within each state (poststratification).
6. Count the number of actual survey respondents per state and merge onto the estimates.
7. Save the 51-row state estimates CSV and a diagnostic summary with full model output.

## 2. Package Selection & Design Choices

**Why `statsmodels` — not `sklearn.LogisticRegression`.** This model requires statistical inference: coefficient estimates, standard errors, p-values, and model fit statistics (pseudo R-squared, deviance, log-likelihood). `statsmodels` provides all of these through `model.summary()`. `sklearn.LogisticRegression` is a prediction-focused API — it returns fitted coefficients but no standard errors, no significance tests, and no deviance-based fit statistics. For a model that will be compared against mixed-effects and Bayesian alternatives, interpretable inference output is essential. Additionally, `statsmodels` uses unpenalized maximum likelihood estimation by default, matching the classical GLM framework, whereas `sklearn` applies L2 regularization by default (`C=1.0`), which would shrink coefficients and make the model subtly different from the frequentist baseline it is intended to be.

**Why the formula API (`smf.glm`).** The `statsmodels.formula.api` module accepts R-style formulas with `C()` notation for categorical variables. This handles dummy encoding internally and ensures that `model.predict(poststrat_df)` applies the same encoding to the poststratification frame — no need to manually construct matching design matrices with `patsy` or `pd.get_dummies()`. The formula `C(gender) + C(race4) + C(educ_category) + C(region9) + co2_per_capita + dem_share_two_party + drive_alone_share + samesex_share` produces 20 parameters (1 intercept + 1 gender dummy + 3 race dummies + 3 education dummies + 8 region dummies + 4 continuous covariates) estimated on 778 observations.

**Why this is a "no pooling" model.** On the pooling spectrum described by Howe et al. (2015), this model sits at the no-pooling extreme for demographic coefficients. Every coefficient — each race category, each education level, each Census division — is estimated freely as a fixed effect with no shrinkage toward a common mean. If the "Hispanic" coefficient is estimated from only 80 respondents, the model trusts that estimate entirely; there is no mechanism to pull it toward the overall intercept. This contrasts with the GLMER-MRP (Model 3), where demographic variables enter as random effects and are partially pooled. The GLM-MRP's fixed-effect estimates will be more variable (higher variance) but unbiased, while the GLMER-MRP trades a small amount of bias for substantially lower variance — the classical bias-variance tradeoff that motivates partial pooling.

**Why `state_fips` is excluded from the model.** State enters the estimates only through poststratification and the four Howe state-level covariates (`co2_per_capita`, `dem_share_two_party`, `drive_alone_share`, `samesex_share`), not as a predictor in the regression. Including 51 state dummies as fixed effects would consume 50 degrees of freedom from 778 observations and risk overfitting. More importantly, excluding state fixed effects isolates what demographic composition alone (mediated through region and the four state-level covariates) can explain about state-level opinion — the gap between this model and the GLMER-MRP (which includes state random intercepts) reveals the value of state-specific partial pooling.

**Estimation method: IRLS.** `statsmodels` fits the binomial GLM using Iteratively Reweighted Least Squares (IRLS), the standard algorithm for maximum likelihood estimation of GLMs. IRLS iterates a weighted least-squares problem where the weights and working response are updated at each step based on the current parameter estimates and the variance function of the Binomial family. Convergence is typically fast — this model converges in 4 iterations with no warnings.

**McFadden pseudo R-squared.** The script computes pseudo R² as `1 − (log L_model / log L_null)`, where `log L_null` is the log-likelihood of an intercept-only model. This is the McFadden pseudo R-squared, the standard goodness-of-fit measure for logistic regression. A value of 0.0109 indicates that the predictors explain very little of the variation in individual-level climate opinion — but this is expected. Individual survey responses are noisy binary outcomes; even a well-specified logistic regression on a 1,200-person survey will have a low pseudo R². The model's value lies not in predicting individual responses but in recovering systematic demographic patterns that, when aggregated through poststratification, produce better state-level estimates than raw means.

## 3. Line-by-Line Technical Breakdown

### Configuration

| Code | Technical Explanation |
|------|----------------------|
| `OUTCOME_VAR = "happening_bin"` | Selects the binary outcome column from the survey. Value is 1 (yes), 0 (no), or NaN (don't-know, dropped by `load_survey()`). |
| `MODEL_NAME = "glm_mrp"` | String identifier used for output file naming: `glm_mrp_state_estimates.csv` and `glm_mrp_summary.txt`. |
| `FORMULA = "happening_bin ~ C(gender) + C(race4) + C(educ_category) + C(region9) + co2_per_capita + dem_share_two_party + drive_alone_share + samesex_share"` | R-style formula parsed by `patsy`. `C()` forces categorical (dummy) encoding. Reference categories are chosen alphabetically by default: gender=Female, race4=Black, educ_category=1, region9=E. North Central. The four Howe state-level covariates enter linearly on the logit scale. |
| `CATEGORICALS = ["gender", "race4", "educ_category", "region9"]` | List of columns that must be cast to string before fitting, ensuring `statsmodels` treats integer-coded variables (gender: 1/2, educ_category: 1–4) as discrete categories, not continuous. |

### Data Loading

| Code | Technical Explanation |
|------|----------------------|
| `survey = load_survey(OUTCOME_VAR)` | Reads the recoded survey CSV, drops rows where `happening_bin` is NaN (don't-know responses), merges all four Howe state-level covariates (`co2_per_capita`, `dem_share_two_party`, `drive_alone_share`, `samesex_share`) from raw state files, and maps `region9` from the FIPS lookup dictionary. Returns 778 rows. |
| `poststrat = load_poststrat_state()` | Reads the 1,632-row state poststratification frame (51 states × 2 genders × 4 races × 4 education levels), each row with an ACS population weight `N`. Merges the same four state covariates and retains `region9`. |

### Categorical Casting

| Code | Technical Explanation |
|------|----------------------|
| `for col in CATEGORICALS: survey[col] = survey[col].astype(str)` | Converts integer-coded categoricals to strings in both DataFrames. Without this, `C(gender)` with values `1, 2` would be treated as a single continuous predictor (or produce unexpected reference levels). String values like `"1"`, `"2"` are unambiguously categorical. Both DataFrames must use the same types so that `model.predict(poststrat)` maps columns correctly. |

### Model Fitting

| Code | Technical Explanation |
|------|----------------------|
| `warnings.catch_warnings(record=True)` | Context manager that captures any warnings raised during fitting as `Warning` objects rather than printing them. This allows the script to log convergence or perfect-separation warnings to the diagnostic file. |
| `smf.glm(formula=FORMULA, data=survey, family=sm.families.Binomial(link=sm.families.links.Logit()))` | Constructs a Generalized Linear Model object. `Binomial()` specifies the response distribution (binary 0/1). `Logit()` is the canonical link function for the Binomial family, mapping the linear predictor to the [0, 1] probability scale via the inverse logit (sigmoid). The formula API internally constructs the design matrix using `patsy`, with one-hot encoding for `C()` terms and a column for each continuous term. |
| `.fit()` | Estimates coefficients via IRLS (Iteratively Reweighted Least Squares). Returns a `GLMResultsWrapper` with attributes: `.params` (coefficients), `.bse` (standard errors), `.llf` (log-likelihood), `.llnull` (null log-likelihood), `.nobs` (observation count), and `.summary()` (formatted results table). |

### Prediction on Poststratification Frame

| Code | Technical Explanation |
|------|----------------------|
| `poststrat["predicted_prob"] = model.predict(poststrat)` | Applies the fitted model to the 1,632 poststrat rows. Internally, `predict()` builds the design matrix from the poststrat DataFrame using the same `patsy` formula, multiplies by the fitted coefficients to get the linear predictor (η = Xβ), and applies the inverse logit transform: P(y=1) = 1 / (1 + exp(−η)). Each row gets a predicted probability conditional on its demographic cell and state covariates. |

### Poststratification

| Code | Technical Explanation |
|------|----------------------|
| `estimates = poststratify(poststrat, "predicted_prob")` | Groups the 1,632 poststrat rows by `state_fips` (32 cells per state) and computes `np.average(predicted_prob, weights=N)` within each group. This is the "P" in MRP: the weighted average of cell-level predictions, where weights are ACS population counts, produces a population-representative state-level estimate. Returns a 51-row DataFrame with `state_fips`, `state_name`, `estimate`. |

### Respondent Counts

| Code | Technical Explanation |
|------|----------------------|
| `n_resp = survey.groupby("state_fips")[OUTCOME_VAR].count().reset_index(name="n_respondents")` | Counts the number of valid (non-NaN) survey responses per state. This is an informational column — it does not affect the estimate but indicates how much direct evidence each state contributed to the model. |
| `estimates = estimates.merge(n_resp, on="state_fips", how="left")` | Left join preserves all 51 states. States not present in the survey (if any) would get NaN for `n_respondents`. |
| `estimates["n_respondents"] = estimates["n_respondents"].fillna(0).astype(int)` | Converts NaN counts to 0 and casts to integer for clean CSV output. |

### Save and Diagnostics

| Code | Technical Explanation |
|------|----------------------|
| `save_estimates(estimates, MODEL_NAME)` | Writes `outputs/estimates/glm_mrp_state_estimates.csv` and prints a console summary (count of valid states, mean/median/min/max of estimates, respondent count range). |
| `diag_dir = OUTPUT_DIR / "diagnostics"` | Creates the diagnostics output directory if it doesn't exist. |
| `pseudo_r2 = 1 - model.llf / model.llnull` | McFadden's pseudo R-squared: ratio of the fitted model's log-likelihood to the null (intercept-only) model's log-likelihood, subtracted from 1. Ranges from 0 (no improvement over null) to 1 (perfect fit). Values above 0.2 are considered a good fit for logistic regression; 0.01 indicates minimal individual-level explanatory power. |
| `f.write(str(model.summary()))` | Writes the full coefficient table including estimates, standard errors, z-statistics, p-values, and 95% confidence intervals for all 18 parameters. |
| State estimate summary stats | Writes mean, median, min, max, and standard deviation of the 51 poststratified state-level estimates. These describe the distribution of model output, not the fit quality. |

## 4. Model Specification

The GLM-MRP fits the following logistic regression:

```
logit(P(y_i = 1)) = β₀ + β₁·gender_i + β₂·race4_i + β₃·educ_i + β₄·region9_i
                  + β₅·co2_s + β₆·dem_share_s + β₇·drive_s + β₈·samesex_s
```

where:
- *y_i* ∈ {0, 1} is respondent *i*'s answer to `happening_bin`
- *gender_i*, *race4_i*, *educ_i*, *region9_i* are categorical predictors encoded as dummy variables (1 + 3 + 3 + 8 = 15 dummy columns plus the intercept)
- *co2_s*, *dem_share_s*, *drive_s*, *samesex_s* are the four Howe et al. (2015) continuous state-level covariates for respondent *i*'s state *s*
- logit(p) = log(p / (1 − p)) is the canonical link function

The model produces 20 parameters estimated by maximum likelihood via IRLS:

| Parameter | Count | Type |
|-----------|-------|------|
| Intercept | 1 | Fixed effect (baseline: Female, Black, educ=1, E. North Central) |
| Gender dummies | 1 | Fixed effect (Male vs. Female) |
| Race dummies | 3 | Fixed effects (Hispanic, Other, White vs. Black) |
| Education dummies | 3 | Fixed effects (educ 2, 3, 4 vs. educ 1) |
| Region dummies | 8 | Fixed effects (8 regions vs. E. North Central) |
| co2_per_capita | 1 | Continuous fixed effect |
| dem_share_two_party | 1 | Continuous fixed effect |
| drive_alone_share | 1 | Continuous fixed effect |
| samesex_share | 1 | Continuous fixed effect |

**Prediction.** For each of the 1,632 cells in the poststratification frame (defined by the cross of state × gender × race × education), the model computes:

```
P̂_cell = logit⁻¹(Xβ̂) = 1 / (1 + exp(−Xβ̂))
```

**Poststratification.** The state-level estimate for state *s* is:

```
θ̂_s = Σ_j (N_j × P̂_j) / Σ_j N_j     for all cells j ∈ state s
```

where *N_j* is the ACS population count for cell *j*. This weights each cell's predicted probability by its share of the state population, producing a demographically adjusted estimate.

## 5. Connection to Howe et al. (2015)

The GLM-MRP implements the "R" (regression) and "P" (poststratification) components of MRP but without the partial pooling that defines the "M" (multilevel). In the Howe framework, this model corresponds to fitting a single-level logistic regression — all predictors enter as fixed effects, and no information is shared across groups through random effects or shrinkage.

**Where this model sits on the pooling spectrum.** The Baseline (Model 1) is the no-pooling extreme at the *state level*: each state's estimate comes only from respondents in that state. The GLM-MRP is the no-pooling extreme at the *coefficient level*: each demographic coefficient is estimated freely from the national data with no shrinkage. The GLMER-MRP (Model 3) introduces partial pooling by replacing fixed demographic effects with random intercepts, shrinking group-specific estimates toward the grand mean proportionally to group sample size.

| Property | Baseline | GLM-MRP (this model) | GLMER-MRP |
|----------|----------|----------------------|-----------|
| Pooling | None (state-level) | None (coefficient-level) | Partial (random intercepts) |
| Demographic adjustment | None | Yes — fixed effects on gender, race, education, region | Yes — random intercepts on the same variables |
| State-level covariates | Not used | Yes — all four Howe covariates (`co2_per_capita`, `dem_share_two_party`, `drive_alone_share`, `samesex_share`) as fixed effects | Yes — same four covariates as fixed effects |
| State-specific parameters | None | None — state variation comes only through region dummies + covariates | Yes — state random intercept (`1|state_fips`) |
| Shrinkage | None | None | Yes — small-sample groups shrink toward grand mean |
| States with zero respondents | NaN | Estimated (via demographic model + poststratification) | Estimated (via model + poststratification) |

**What this model captures vs. Howe.** The GLM-MRP captures the demographic structure of opinion (how gender, race, and education relate to climate beliefs) and adjusts state estimates for compositional differences using poststratification. It incorporates **all four** Howe et al. (2015) state-level covariates: per-capita CO₂ emissions, Democratic two-party vote share, share of workers driving alone to work, and share of households that are same-sex coupled. What it still misses is the partial pooling that makes MRP powerful: (a) demographic group estimates are not regularized, so small groups like "Other" race (n ≈ 60) have high-variance coefficients; (b) state variation is captured only through 9 region dummies and 4 covariates, not through 51 state-level random intercepts that would allow each state to deviate from its region mean based on local data.

**Why this matters for the comparison.** The GLM-MRP is the bridge between the Baseline and the full MRP specification. Comparing Model 1 to Model 2 isolates the value of demographic regression + poststratification (the RP). Comparing Model 2 to Model 3 isolates the value of multilevel partial pooling (the M). This sequential decomposition follows the logic of Howe et al.'s framework, where each layer of the model addresses a specific limitation of simpler approaches.
