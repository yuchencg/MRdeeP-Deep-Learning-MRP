# 03 GLMER-MRP (Mixed-Effects Logistic Regression + Poststratification) — Explained

## 1. Big Picture: What This Script Accomplishes

The GLMER-MRP model produces state-level estimates of public opinion by fitting a frequentist generalized linear mixed model (GLMM): a logistic regression where demographic and geographic grouping variables enter as **random intercepts** rather than fixed-effect dummies. This is the partial-pooling core of MRP — each group's deviation from the grand mean is shrunk toward zero in proportion to how little data supports it. After fitting, the model predicts a probability for every cell in the ACS poststratification frame and aggregates those predictions to state-level estimates weighted by population. Compared to the GLM-MRP, this model trades unbiased-but-noisy fixed-effect estimates for slightly biased-but-stable shrunken estimates — the bias-variance tradeoff that motivates multilevel modeling.

1. Load the recoded survey data (dropping NaN on `happening_bin`) and the ACS state poststratification frame with state-level covariates and `region9` attached.
2. Cast all five grouping variables (`race4`, `educ_category`, `gender`, `region9`, `state_fips`) to string for unambiguous categorical handling.
3. Build the inputs `gpboost` expects: a 5-column `group_data` matrix of group labels, a fixed-effects design matrix `X` with intercept + the four Howe et al. (2015) state-level covariates, and a `float64` outcome vector `y`.
4. Fit the GLMM via Laplace-approximated maximum likelihood (`likelihood="bernoulli_logit"`), estimating one variance component per random effect plus three fixed-effect coefficients.
5. Build the matching prediction inputs from the 1,632-row poststrat frame and call `gp_model.predict(..., predict_response=True)` to get inverse-logit probabilities.
6. Poststratify cell-level predictions to state-level estimates via population-weighted averaging.
7. Save the 51-row state estimate CSV, capture diagnostics including variance components and a comparison against the Baseline.

## 2. Package Selection & Design Choices

**Why `gpboost`.** `gpboost` is a pure-Python (C++ backend) library that fits GLMMs via the same **Laplace approximation** used by R's `lme4::glmer()`. It supports Bernoulli outcomes with a logit link (`bernoulli_logit`), handles multiple crossed grouping variables natively, and on benchmarks runs over 100× faster than `lme4` on Bernoulli problems with many groups. For this project, the decisive features are: (a) Laplace approximation matches the `glmer` reference frequentists are familiar with, (b) no R dependency keeps the pipeline pure Python, and (c) the API accepts a `group_data` matrix with arbitrary categorical labels, including unseen levels at prediction time — which is exactly what MRP needs for states absent from the survey.

**Alternatives considered and rejected.**

- **`pymer4`** — wraps R's `lme4` via `rpy2`. Would give identical estimates to `glmer`, but introduces an R installation dependency. Adds installation complexity and reproducibility friction across machines. Rejected to keep the project pure Python.
- **`statsmodels.BinomialBayesMixedGLM`** — variational Bayes, not maximum-likelihood Laplace. The variational approximation is known to underestimate posterior variance and produce biased point estimates on small-sample binary outcomes. Statsmodels also has no native frequentist GLMM with Laplace approximation — the `MixedLM` class is Gaussian-only. Rejected on accuracy grounds.
- **`bambi` / `pymc`** — Bayesian MCMC via NUTS sampler. Would produce well-calibrated estimates but at substantially higher computational cost and with a different inference paradigm. Reserved for **Model 4 (Stan-MRP)** to preserve a clean frequentist-vs-Bayesian comparison: Models 2–3 are MLE-based, Model 4 is fully Bayesian with priors. Mixing Bayesian estimation into Model 3 would muddy that contrast.

**How Laplace approximation works (high level).** A GLMM has no closed-form marginal likelihood — integrating out the random effects requires evaluating an intractable high-dimensional integral. The Laplace approximation handles this by: (1) for each candidate set of variance components, finding the **mode** of the conditional posterior of the random effects (a penalized IRLS problem), (2) approximating the integrand around that mode with a second-order Taylor expansion (a Gaussian centered at the mode with curvature given by the Hessian), and (3) using the resulting Gaussian integral as a tractable approximation to the marginal likelihood. The variance components are then optimized via gradient-based methods (typically L-BFGS or Newton) to maximize this approximate marginal likelihood. This is the same algorithm `lme4::glmer()` uses by default with `nAGQ=1` — `gpboost`'s implementation differs only in the C++ optimizer choice, not in the statistical method.

**Why `bernoulli_logit` and not `binary` (probit).** `gpboost`'s default `likelihood="binary"` uses a **probit link**, modeling P(y=1) = Φ(η) where Φ is the standard normal CDF. This is convenient for some Bayesian formulations (the latent variable representation is clean) but does not match R's default behavior — `glmer(..., family=binomial)` defaults to `link="logit"`. Howe et al. (2015) and the broader MRP literature use logit. Specifying `likelihood="bernoulli_logit"` explicitly forces the inverse-logit link P(y=1) = 1 / (1 + exp(−η)), making coefficients directly comparable to the GLM-MRP (Model 2) and aligning with the published MRP standard.

**Why crossed random effects, not nested `(1|region/state)`.** In the canonical Howe formulation, region nests state — `(1|region) + (1|region:state)` allows each region to have its own mean and each state to deviate from its region. `gpboost` estimates each grouping variable in `group_data` as a separate variance component without an explicit nesting hierarchy; passing `region9` and `state_fips` as separate columns produces **crossed** random effects rather than nested ones. For prediction purposes the two formulations are essentially equivalent: in either case the linear predictor for a respondent in region *r*, state *s* receives the sum α^region_r + α^state_s. The variance decomposition differs slightly (nested splits geographic variance into within-region-state and between-region components; crossed estimates them as independent variance components), but both produce the partial-pooling shrinkage that MRP needs. This is acceptable here because (a) `gpboost` still estimates separate variances for region and state, capturing both layers of geographic variation, and (b) the four Howe state-level covariates (`co2_per_capita`, `dem_share_two_party`, `drive_alone_share`, `samesex_share`) provide the "informative shrinkage target" that nesting otherwise gives — states without survey data are pulled toward a covariate-predicted mean rather than a flat grand mean.

**Optimizer settings.** `gp_model.set_optim_params(params={"maxit": 1000})` raises the iteration cap from the default to handle slow convergence on the variance components. With small per-group sample sizes (e.g., 2 gender groups but 51 states), the variance-component optimization can be flat near zero and benefits from extra iterations. Convergence warnings are captured via `warnings.catch_warnings(record=True)` and logged to the diagnostic file rather than suppressed.

## 3. Line-by-Line Technical Breakdown

### Configuration

| Code | Technical Explanation |
|------|----------------------|
| `OUTCOME_VAR = "happening_bin"` | Binary climate-change outcome; rows with NaN dropped by `load_survey()`. |
| `MODEL_NAME = "glmer_mrp"` | Used for output file naming: `glmer_mrp_state_estimates.csv` and `glmer_mrp_summary.txt`. |
| `GROUPING_VARS = ["race4", "educ_category", "gender", "region9", "state_fips"]` | Five random-effect grouping variables. Order matches the columns of `group_data` and the `Group_1 ... Group_5` labels in `gp_model.summary()`. |
| `COVARIATES = ["co2_per_capita", "dem_share_two_party", "drive_alone_share", "samesex_share"]` | The four Howe et al. (2015) state-level covariates entered as fixed-effect columns in `X`: per-capita CO₂ emissions, Democratic two-party vote share, share of workers who drive alone to work (ACS B08301), and share of households that are same-sex coupled (ACS B11009). |

### Data Loading and Casting

| Code | Technical Explanation |
|------|----------------------|
| `survey = load_survey(OUTCOME_VAR)` | 778 rows after dropping NaN on `happening_bin`. Includes the four merged Howe state-level covariates and `region9`. |
| `poststrat = load_poststrat_state()` | 1,632 rows (51 states × 2 genders × 4 races × 4 education levels) with `N` (ACS weight), the same four covariates, and `region9`. |
| `for col in GROUPING_VARS: survey[col] = survey[col].astype(str)` | Casts integer-coded categoricals (`gender` 1/2, `educ_category` 1–4) to strings. `gpboost` treats each unique label as a discrete level; mixing string and integer types between training and prediction would produce mismatched groups. |

### Training Inputs

| Code | Technical Explanation |
|------|----------------------|
| `y = survey[OUTCOME_VAR].to_numpy(dtype=np.float64)` | `gpboost` requires `float64` for the outcome vector even when values are 0/1. Integer dtype raises a type error. |
| `group_data = survey[GROUPING_VARS].to_numpy()` | (778, 5) array of group labels. Each row is one respondent; each column is one grouping variable. `gpboost` internally builds an indicator matrix `Z` for each grouping variable. |
| `X = np.column_stack([np.ones(len(survey)), *[survey[c] for c in COVARIATES]])` | (778, 5) design matrix for fixed effects: column 0 = intercept, columns 1–4 = the four Howe state-level covariates in the order defined by `COVARIATES` (CO₂ per capita, Dem two-party vote share, drive-alone share, same-sex household share). The intercept must be added explicitly — `gpboost` does not insert one. |

### Model Fitting

| Code | Technical Explanation |
|------|----------------------|
| `gpb.GPModel(group_data=group_data, likelihood="bernoulli_logit")` | Constructs a GPModel with five grouped random intercepts and a Bernoulli-logit likelihood. No fixed effects are passed yet — those are supplied via `X` at fit time. The model object holds the estimation state and methods for fit/predict/summary. |
| `gp_model.set_optim_params(params={"maxit": 1000})` | Raises the maximum number of optimizer iterations for the variance-component MLE. Default is sometimes too low for flat likelihood surfaces near zero variance. |
| `gp_model.fit(y=y, X=X)` | Estimates: (a) random-effect variance components τ²_k for each grouping variable k via Laplace-approximated MLE, (b) fixed-effect coefficients β for the columns of X via penalized IRLS conditional on τ². Internally, optimization alternates between conditional-mode-finding for random effects and gradient steps for variance components. |
| `caught_warnings = list(_w)` | Captures convergence-related warnings as `Warning` objects for later inclusion in diagnostics, rather than printing them to stderr. |
| `gp_model.get_cov_pars()` | Returns the estimated random-effect variances as a 1-D array (one entry per grouping variable, in the order columns of `group_data`). |
| `gp_model.get_coef()` | Returns the fixed-effect coefficient estimates for the columns of `X`. |

### Prediction Inputs and Poststratification

| Code | Technical Explanation |
|------|----------------------|
| `group_data_pred = poststrat[GROUPING_VARS].to_numpy()` | (1632, 5) array with the same column order as `group_data`. Group labels for each poststrat cell. Must use the same string dtype as training. |
| `X_pred = np.column_stack([np.ones(len(poststrat)), *[poststrat[c] for c in COVARIATES]])` | (1632, 5) prediction design matrix. Same column structure and order as training X. |
| `gp_model.predict(X_pred=X_pred, group_data_pred=group_data_pred, predict_var=False, predict_response=True)` | Computes the posterior mean prediction for each poststrat cell. `predict_var=False` skips the (computationally expensive) prediction variance. `predict_response=True` applies the inverse-logit transform internally so the returned `mu` is on the [0, 1] probability scale rather than the latent linear-predictor scale. **Critical for MRP** — without it, you get logit values that cannot be averaged into probabilities. |
| `poststrat["predicted_prob"] = pred["mu"]` | Stores per-cell predicted probabilities. For groups present in training, `mu` reflects the shrunken random effect; for groups absent (e.g., a state with no survey respondents), the random effect is set to its prior mean of zero, so `mu` is driven by the fixed-effect intercept + covariates only — the partial-pooling fallback. |
| `estimates = poststratify(poststrat, "predicted_prob")` | Computes `np.average(predicted_prob, weights=N)` within each `state_fips`. Returns the 51-row state-level CSV. |

### Respondent Counts and Output

| Code | Technical Explanation |
|------|----------------------|
| `n_resp = survey.groupby("state_fips")[OUTCOME_VAR].count().reset_index(name="n_respondents")` | Counts non-NaN responses per state for the diagnostic column. Does not affect the estimate. |
| `estimates = estimates.merge(n_resp, on="state_fips", how="left")` | Left join preserves all 51 states; states with no respondents would get NaN, then 0. |
| `save_estimates(estimates, MODEL_NAME)` | Writes `outputs/estimates/glmer_mrp_state_estimates.csv` and prints the console summary. |

### Baseline Comparison and Diagnostics

| Code | Technical Explanation |
|------|----------------------|
| `baseline = pd.read_csv(baseline_path, dtype={"state_fips": str})` + `merged["abs_diff"]` | Reads Model 1's saved estimates and computes the absolute per-state difference. `diff_count` records how many states moved by more than 10 percentage points — a proxy for how much shrinkage and demographic adjustment are doing. |
| Variance component block | Iterates over `GROUPING_VARS` and `cov_pars` in matched order, writing each variance estimate to the diagnostic file. Larger τ² means more group-level heterogeneity captured by that random effect; a τ² that collapses to zero indicates the optimizer found no detectable group-level variance beyond what fixed effects explain. |
| Fixed-effect block | Writes the three coefficient estimates (intercept, `co2_per_capita`, `dem_share_two_party`) with names matched to `X` columns. |
| `gp_model.summary()` capture via `redirect_stdout` | `gp_model.summary()` prints to stdout rather than returning a string. Redirecting via `io.StringIO` captures the printed table (including standard errors, z-statistics, and p-values for fixed effects) into the diagnostic file. |

## 4. Model Specification

The GLMER-MRP fits the following GLMM (the Howe et al. 2015 spec, with `mode` and `time` random effects dropped because this is a single survey from a single mode in a single year):

```
logit(P(y_i = 1)) = β₀ + β₁·co2_s + β₂·dem_share_s + β₃·drive_s + β₄·samesex_s
                  + u^race_{r(i)} + u^educ_{e(i)} + u^gender_{g(i)} + u^region_{R(i)} + u^state_{s(i)}
```

with random effects:

```
u^race_k    ~ N(0, τ²_race)        for k = 1..4
u^educ_k    ~ N(0, τ²_educ)        for k = 1..4
u^gender_k  ~ N(0, τ²_gender)      for k = 1..2
u^region_k  ~ N(0, τ²_region)      for k = 1..9
u^state_k   ~ N(0, τ²_state)       for k = 1..51
```

Each random effect is a group-specific deviation from the linear predictor, drawn from a Normal distribution with mean zero and group-specific variance. The five variance components τ² are estimated from the data via Laplace-approximated MLE — they are the parameters that control how much shrinkage each grouping variable receives. Large τ² means little shrinkage (groups are estimated nearly as if they were free fixed effects); τ² near zero means heavy shrinkage (group estimates collapse toward the grand mean).

| Parameter | Count | Type | Estimation |
|-----------|-------|------|-----------|
| Intercept (β₀) | 1 | Fixed | Penalized IRLS |
| co2_per_capita (β₁) | 1 | Fixed | Penalized IRLS |
| dem_share_two_party (β₂) | 1 | Fixed | Penalized IRLS |
| drive_alone_share (β₃) | 1 | Fixed | Penalized IRLS |
| samesex_share (β₄) | 1 | Fixed | Penalized IRLS |
| τ²_race | 1 | Variance component | Laplace MLE |
| τ²_educ | 1 | Variance component | Laplace MLE |
| τ²_gender | 1 | Variance component | Laplace MLE |
| τ²_region | 1 | Variance component | Laplace MLE |
| τ²_state | 1 | Variance component | Laplace MLE |
| u^k (BLUPs) | 4+4+2+9+51 = 70 | Random (predicted) | Conditional posterior mean |

The 70 random intercepts are not free parameters — they are predicted (BLUPs: best linear unbiased predictors) conditional on the estimated variance components. This is what gives partial pooling: a state with 5 respondents has a BLUP shrunk hard toward zero (the prior mean), while a state with 24 respondents has a BLUP closer to its raw deviation.

**Prediction.** For each poststrat cell with feature values *(state s, gender g, race r, educ e, region R, co2_s, dem_s, drive_s, samesex_s)*:

```
η̂_cell = β̂₀ + β̂₁·co2_s + β̂₂·dem_s + β̂₃·drive_s + β̂₄·samesex_s
       + û^race_r + û^educ_e + û^gender_g + û^region_R + û^state_s
P̂_cell = 1 / (1 + exp(−η̂_cell))
```

For groups not seen in training, û is replaced by 0 (its prior mean). This is how the model produces estimates for states without survey respondents.

**Poststratification.** Same as Model 2:

```
θ̂_s = Σ_j (N_j × P̂_j) / Σ_j N_j     for all cells j ∈ state s
```

## 5. Connection to Howe et al. (2015)

The GLMER-MRP is the **partial pooling** model — the multilevel core of MRP and the closest of the first three models to the published Howe et al. (2015) specification. It is where the "M" of MRP earns its place: by modeling demographic and geographic variables as random effects with estimated variance components, the model decides *from the data* how much each grouping variable should influence state estimates and how much each group should be shrunk toward the grand mean.

**Position on the pooling spectrum.**

| Property | Baseline | GLM-MRP | GLMER-MRP (this model) |
|----------|----------|---------|------------------------|
| Pooling on geography | None — each state from its own respondents | Implicit — region dummies + covariates | **Partial — state random intercept shrinks toward covariate-predicted mean** |
| Pooling on demographics | None | None — fixed-effect dummies | **Partial — random intercepts shrink toward the grand mean** |
| Demographic adjustment | None | Yes (fixed effects) | Yes (random effects) |
| State-level covariates | Not used | All four Howe covariates (fixed effects) | All four Howe covariates (fixed effects) |
| State-specific parameters | None | None | **Yes — `(1|state_fips)`** |
| States with zero respondents | NaN | Estimated (region + 4 covariates) | Estimated (region + state intercept = 0 + 4 covariates) |
| Borrowing of strength | None | Across demographic groups (within a single national fit) | **Across both demographic and geographic groups** |

**What this model captures vs. the Baseline and GLM-MRP.** The Baseline pools nothing; the GLM-MRP pools demographics-on-states implicitly through the regression but leaves coefficient estimates unregularized; the GLMER-MRP adds two key things: (a) demographic coefficients are now shrunk toward the grand mean by the variance components τ²_race, τ²_educ, τ²_gender — small-sample groups like "Other" race no longer pull state estimates around freely, and (b) each state has its own random intercept, allowing states within the same Census division to deviate from each other when local survey data supports it. This is the structure Howe et al. exploit to get state-level estimates that correlate with ground truth far better than disaggregation.

**Departures from the full Howe specification.**

- **Crossed rather than nested random effects.** Howe's `(1|region/state)` is implemented here as `(1|region9) + (1|state_fips)` — separate variance components rather than a strict nesting. For the prediction-only purpose of MRP, the two formulations are essentially equivalent (both give each respondent the sum α^region + α^state); they differ only in how the geographic variance is decomposed. The four state-level covariates supply the informative shrinkage target Howe's nesting otherwise provides: a sparse-data state is shrunk toward the prediction implied by its CO₂ emissions, partisan lean, drive-alone share, and same-sex household share, not just a flat regional mean.
- **No `mode` or `time` random effects.** The Howe model includes random intercepts for survey mode (phone vs. online) and survey year because it pools 12 surveys collected over six years and two modes. This implementation uses a single survey from a single mode at a single point in time, so per the Howe Guide §8.3 those terms are dropped — there is no variance to estimate.
- **Single survey, single outcome.** Howe et al. pool 12 surveys totaling n = 12,061 across 2008–2013; this implementation uses a single ~1,200-respondent survey (778 valid responses on `happening_bin`). The model structure is identical, but the absolute volume of data per state is much smaller — meaning shrinkage is heavier here than in Howe's published estimates.

**Note on the actual run.** When fit on this 778-observation sample with all four Howe covariates, the variance components for `race4`, `educ_category`, `region9`, and `state_fips` all collapsed to ≈ 0 in the Laplace optimization (only `gender` retained a tiny non-zero variance, ≈ 3.8e-5). This is not a bug — it is the model **declining to estimate group-level heterogeneity it cannot reliably detect**. With only 5–24 respondents per state and 778 observations spread across many crossed groups, the marginal likelihood is maximized by setting most τ² ≈ 0, leaving the predictions driven by the fixed-effect intercept and the four state-level covariates. This is the most aggressive form of partial pooling — every group is shrunk all the way to zero — and it produces a tighter estimate range (0.40–0.60, std = 0.034) than the Baseline (0.14–0.88, std ≈ 0.15), with 20 of 51 states differing from the Baseline by more than 10 percentage points. A larger sample (closer to Howe's n = 12,061) would be expected to lift several of these variance components above zero, particularly for `state_fips` and `region9`.
