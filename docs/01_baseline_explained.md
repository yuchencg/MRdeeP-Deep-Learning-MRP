# 01 Baseline (Naive Disaggregation) — Explained

## 1. Big Picture: What This Script Accomplishes

The Baseline model produces state-level estimates of public opinion on climate change using the simplest possible approach: raw disaggregation. For each state, it computes the unweighted mean of `happening_bin` from whatever survey respondents happen to reside in that state. There is no statistical model, no demographic adjustment, and no borrowing of information across states. The result is a 51-row CSV of state-level proportions that serves as the lower-bound benchmark against which all subsequent MRP models are compared.

1. Load the recoded national survey data and drop respondents who did not answer the climate question (`happening_bin` is NaN).
2. Group respondents by state FIPS code.
3. Within each state, compute the arithmetic mean of the binary outcome (proportion who say climate change is happening) and count the number of respondents.
4. Ensure all 51 states (including DC) appear in the output — states with zero respondents receive `NaN`.
5. Save the estimates CSV and a plain-text diagnostic summary.

## 2. Package Selection & Design Choices

**Why pandas only — no modeling library.** The Baseline deliberately avoids any statistical modeling. It uses only `pandas` for data manipulation and `numpy` (via `utils.py`) for potential weighted-average operations in downstream models. This is not a limitation but a design choice: the Baseline establishes what you get with zero statistical sophistication. It answers the question "how well can you estimate state opinion if you just tabulate your survey responses by geography?"

**Why this matters as a benchmark.** Ornstein (2020) demonstrates that with a national sample of ~1,000 respondents spread across 50 states, naive disaggregation produces estimates that correlate at roughly r ≈ 0.05 with ground truth. This survey sits squarely in that regime: 1,200 raw responses (1,175 after dropping 25 Puerto Rico respondents who fall outside the 51-state poststrat frame), and just 778 valid responses for `happening_bin` after dropping don't-know answers. Spread across 51 states, the median state contributes only **15 respondents** — far too few for a state-level proportion to be statistically reliable on its own. The fundamental problems Ornstein identifies are amplified here: states with few respondents produce noisy estimates, and the demographic composition of respondents in each state cannot be expected to match the state population. These are exactly the problems MRP is designed to solve — the Baseline quantifies the gap.

**No poststratification.** The Baseline computes raw survey means, not population-weighted means. Poststratification is the "P" in MRP and is applied starting with Model 2 (GLM-MRP). Excluding it here isolates the contribution of poststratification from the contribution of the regression model.

## 3. Line-by-Line Technical Breakdown

### `models/utils.py` — Functions called by Baseline

| Code | Technical Explanation |
|------|----------------------|
| `BASE_DIR = Path(__file__).resolve().parent.parent` | Resolves the repo root relative to this module's location (`models/` → repo root). All paths are built from this anchor so the script runs correctly regardless of working directory. |
| `STATE_FIPS_TO_REGION` | Dictionary mapping all 51 two-digit FIPS codes to their Census division name (9 divisions). Hardcoded rather than derived from data to guarantee completeness. Used by `load_survey()` and `load_poststrat_state()` to attach `region9`. |
| `STATE_FIPS_TO_NAME` | Dictionary mapping FIPS codes to full state names. Used by `poststratify()` and `01_baseline.py` to label output rows. |
| `_load_state_covariates()` | Reads `carbon_state.csv` and `pres_state.csv`, selects only the key columns (`state_fips`, `co2_per_capita`, `dem_share_two_party`), and outer-joins them on `state_fips`. Returns a 51-row DataFrame of state-level covariates. |
| `load_survey(outcome_var)` | Reads the full recoded survey CSV with `state_fips` forced to string dtype. Drops rows where `outcome_var` is NaN (don't-know responses). Merges state covariates and maps `region9` from the FIPS dictionary if not already present. Returns the cleaned DataFrame ready for modeling. |
| `save_estimates(estimates_df, model_name)` | Writes the estimates DataFrame to `outputs/estimates/{model_name}_state_estimates.csv`. Prints a formatted console summary: count of valid/NaN states, mean/median/min/max of estimates, and respondent count range if available. |

### `models/01_baseline.py`

| Code | Technical Explanation |
|------|----------------------|
| `survey = load_survey(OUTCOME_VAR)` | Loads the recoded survey, drops NaN rows for `happening_bin`, merges state covariates and `region9`. For the Baseline these extra columns are unused — they exist because `load_survey()` is shared across all models. |
| `state_stats = survey.groupby("state_fips").agg(estimate=(OUTCOME_VAR, "mean"), n_respondents=(OUTCOME_VAR, "count")).reset_index()` | Groups by state and computes two named aggregations: `estimate` is the arithmetic mean of the binary outcome (i.e., proportion answering 1), and `n_respondents` is the count of non-NaN responses in that state. The `.agg()` named-tuple syntax produces clean column names directly. |
| `all_states = pd.DataFrame({"state_fips": ..., "state_name": ...})` | Creates a 51-row scaffold from the `STATE_FIPS_TO_NAME` dictionary, guaranteeing every state appears in the output even if it has zero survey respondents. |
| `estimates = all_states.merge(state_stats, on="state_fips", how="left")` | Left join ensures all 51 states are retained. States absent from `state_stats` (no respondents) get `NaN` for `estimate` and `NaN` for `n_respondents`. |
| `estimates["n_respondents"] = estimates["n_respondents"].fillna(0).astype(int)` | Converts NaN respondent counts to 0 and casts to integer for clean output. The `estimate` column retains `NaN` for these states — a deliberate signal that no estimate is possible. |
| `save_estimates(estimates, MODEL_NAME)` | Writes `outputs/estimates/baseline_state_estimates.csv` and prints the console summary. |
| Diagnostics block | Prints counts of states with/without estimates, lists any missing state names, and shows the range of estimates and respondent counts. Also writes a full diagnostic text file to `outputs/diagnostics/baseline_summary.txt` including the complete state-level table. |

## 4. Model Specification

There is no statistical model. The estimator for state *s* is:

```
θ̂_s = (1 / n_s) × Σ_{i ∈ s} y_i
```

where *y_i* ∈ {0, 1} is respondent *i*'s answer to `happening_bin` and *n_s* is the number of respondents in state *s*. If *n_s* = 0, then θ̂_s is undefined (NaN).

This is the sample proportion — the maximum likelihood estimator of the population proportion under a Binomial(n_s, θ_s) model with no covariates, no pooling, and no prior.

The variance of this estimator is θ_s(1 − θ_s) / n_s, which is large when n_s is small. In this survey, after dropping NaN responses on `happening_bin`, only 778 respondents remain across 51 states. The median state has 15 respondents (range 5–24). At n_s = 15 with θ_s = 0.5, the standard error of the sample proportion is roughly √(0.25 / 15) ≈ 0.13 — meaning a 95% confidence interval spans about ±0.25 around each estimate. This is the small-sample noise regime where Ornstein's r ≈ 0.05 result was demonstrated, and it is the central problem MRP is designed to solve.

## 5. Connection to Howe et al. (2015)

The Baseline represents the **no-pooling** extreme on the pooling spectrum that Howe et al. describe. Each state's estimate depends entirely on the respondents within that state, with no information shared across states:

| Property | Baseline | Howe MRP |
|----------|----------|----------|
| Pooling | None — each state estimated independently | Partial — random effects shrink state estimates toward the grand mean |
| Demographic adjustment | None — raw sample proportions | Yes — regression on gender, race, education, region; poststratified to census demographics |
| State-level covariates | Not used | CO₂ per capita, Democratic vote share shift state means |
| Borrowing of strength | None | States with few respondents borrow from demographically similar states |
| States with zero respondents | NaN | Still estimated via demographic model + poststratification |

The Baseline estimate will correlate well with truth when (a) the sample is large enough per state and (b) each state's respondents happen to be demographically representative. Condition (b) is almost never met in a national survey — some states will be overrepresented among college-educated whites, others among younger respondents, etc. MRP corrects for this; the Baseline does not.

## 6. Output Interpretation

### `outputs/estimates/baseline_state_estimates.csv`

| Column | Meaning |
|--------|---------|
| `state_fips` | Two-digit FIPS code (zero-padded string) |
| `state_name` | Full state name |
| `estimate` | Proportion of respondents in this state who answered 1 to `happening_bin`. Range [0, 1]. NaN if no respondents. |
| `n_respondents` | Number of survey respondents in this state with a valid (non-NaN) answer. 0 if no respondents. |

### Interpretation

Each `estimate` is the raw fraction of respondents in a state who believe climate change is happening. A value of 0.72 means 72% of survey respondents in that state answered affirmatively. This is **not** an estimate of what the state's full population believes — it is an estimate of what *people who look like the survey sample in that state* believe.

### Actual Run Results (1,200-respondent survey)

| Metric | Value |
|--------|-------|
| Raw survey responses | 1,200 |
| After dropping Puerto Rico (no FIPS in 51-state frame) | 1,175 |
| After dropping NaN on `happening_bin` (don't-know) | 778 |
| States with non-NaN estimate | 51 / 51 |
| States with NaN estimate | 0 |
| Respondents per state — min / median / max | 5 / 15 / 24 |
| Estimate — min / median / max | 0.143 / 0.524 / 0.875 |
| Estimate — mean | 0.521 |
| States with fewer than 15 respondents | 22 |
| States with fewer than 10 respondents | 1 |

The estimate range from 0.14 to 0.88 across states should be read with skepticism: a state with 7 respondents where 1 says yes produces an estimate of 0.143, but that point estimate carries a 95% confidence interval spanning roughly [0.00, 0.58]. The **spread of estimates is dominated by sampling noise, not real geographic variation in opinion**. This is precisely the failure mode MRP is built to correct.

### Caveats

- **No demographic correction.** If a state's survey respondents skew more educated or more urban than the state population, the estimate will be biased toward the opinions of those groups.
- **Small-sample noise.** With a median of 15 respondents per state (and one state with only 5), the standard error of each state proportion is large. A state with 7 respondents where 5 say yes gets an estimate of 0.71, but the 95% confidence interval spans roughly [0.36, 0.92]. Most of the apparent state-to-state variation in this survey is noise, not signal.
- **Non-response bias.** Respondents who answered "don't know" are dropped (NaN on `happening_bin`). 33.8% of valid respondents answered don't-know on this item — a substantial share. If don't-know respondents are systematically different from yes/no respondents, the Baseline estimate is biased.
- **No borrowing of strength.** Unlike MRP, the Baseline cannot produce estimates for states with zero respondents. In this dataset every state happened to receive at least 5 respondents, so no estimate is missing — but this is a feature of the survey design, not a property of the Baseline method.
