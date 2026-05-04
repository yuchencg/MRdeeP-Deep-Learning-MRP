# Prompt 2 — GLM-MRP (Logistic Regression + Poststratification)

Read `CLAUDE.md` for full project context. Read `models/utils.py` before writing — import and use its functions.

## Task

Create `models/02_glm_mrp.py` — a frequentist logistic regression with all fixed effects, poststratified to state-level estimates.

### What This Script Does

1. **Load data** via `utils.load_survey("happening_bin")` and `utils.load_poststrat_state()`

2. **Fit a logistic GLM** using `statsmodels.api` or `statsmodels.formula.api`:
   ```
   happening_bin ~ C(gender) + C(race4) + C(educ_category) + C(region9) + co2_per_capita + dem_share_two_party
   ```
   - Family: `Binomial()`, Link: `Logit()`
   - All demographic/geographic variables as categorical dummies
   - `co2_per_capita` and `dem_share_two_party` as continuous fixed effects
   - `state_fips` does NOT enter the model

3. **Predict** P(happening=1) for every row in the poststrat frame
   - The poststrat frame must have the same columns the model expects: `gender`, `race4`, `educ_category`, `region9`, `co2_per_capita`, `dem_share_two_party`
   - Use `model.predict(poststrat_df)` to get predicted probabilities
   - Store as a new column `predicted_prob` on the poststrat frame

4. **Poststratify** via `utils.poststratify(poststrat_df, "predicted_prob")` — weighted average of cell-level predictions by ACS population count `N`, grouped by state

5. **Compute n_respondents** per state from the survey data

6. **Save estimates** via `utils.save_estimates()` with model_name `"glm_mrp"`

7. **Save diagnostics** — capture and save:
   - Full model summary (`model.summary()` text)
   - Number of observations used
   - Pseudo R-squared
   - Any convergence warnings
   - State estimate summary stats (mean, median, min, max, std)

8. **Print** summary to console

### Key Implementation Notes

- Use the formula API (`smf.glm(formula, data, family)`) for clean categorical handling, OR use `sm.GLM` with `patsy` dummies — either works
- Make sure categorical variables are explicitly cast: `df['gender'] = df['gender'].astype(str)` before fitting, so statsmodels treats them as categories not continuous
- The poststrat frame must go through the same encoding as the training data. If using the formula API, use `model.predict(poststrat_df)` which handles this automatically. If using design matrices, ensure dummy columns match exactly.
- Watch for perfect separation warnings — with ~1,200 respondents and many categorical levels, some cells may be empty. Log any warnings but don't suppress them.

## Also Create

- `docs/02_glm_mrp_explained.md` — Full explanation following the format in CLAUDE.md § "Explanation File Format". Include all 5 sections.
  - In **Package Selection & Design Choices**: explain why `statsmodels` is appropriate for this fixed-effects model, how it differs from `sklearn.LogisticRegression` (statsmodels gives inference/p-values, sklearn is prediction-focused), and why this is a "no pooling" model on the pooling spectrum.
  - In **Connection to Howe et al. (2015)**: explain that this model corresponds to the "no pooling" extreme — each demographic coefficient is estimated freely with no shrinkage. State-level variation is captured only through region dummies and covariates, not through state-specific parameters.

## Conventions

- Author docstring (Author: KO, today's date)
- Minimal inline comments
- `state_fips` always zero-padded string
