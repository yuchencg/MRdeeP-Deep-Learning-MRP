# Prompt 1 — Baseline (Naive Disaggregation)

Read CLAUDE.md for full project context. Then build the following files:

## Files to Create

### 1. `models/utils.py`
Shared utility module imported by all model scripts. Must include:

- `load_survey(outcome_var)` — loads `test_data/processed/climate_survey_responses_recoded.csv`, drops rows where `outcome_var` is NaN, returns clean DataFrame. Must also merge state-level covariates (`test_data/raw/carbon_state.csv` and `test_data/raw/pres_state.csv`) onto the survey data by `state_fips`. Must also ensure `region9` is present on the survey data (build a state_fips → region9 lookup from the survey data itself, or hardcode the 9 Census division mapping).
- `load_poststrat_state()` — loads `test_data/processed/poststrat_state.csv`, merges on state-level covariates and `region9`, returns DataFrame with columns: `state_fips`, `gender`, `race4`, `educ_category`, `region9`, `co2_per_capita`, `dem_share_two_party`, `N`.
- `poststratify(poststrat_df, prob_col)` — given a poststrat frame with predicted probabilities in `prob_col`, computes `weighted_mean = sum(prob * N) / sum(N)` grouped by `state_fips`. Returns DataFrame with `state_fips`, `state_name`, `estimate`.
- `save_estimates(estimates_df, model_name)` — saves to `outputs/estimates/{model_name}_state_estimates.csv` and prints a console summary (number of states, mean/median/min/max estimate, any NaN states).
- A `STATE_FIPS_TO_REGION` dictionary mapping all 51 state FIPS codes to their Census division name.

### 2. `models/__init__.py`
Empty init file.

### 3. `models/01_baseline.py`
The Baseline model script. This is the simplest possible estimator — no model at all. For each state, compute the raw mean of `happening_bin` from whatever survey respondents are in that state.

Steps:
1. Call `load_survey("happening_bin")`
2. Group by `state_fips`, compute mean of `happening_bin` and count of respondents per state
3. For states with zero respondents in the survey, the estimate should be NaN
4. Call `save_estimates()` to write output
5. Print a diagnostic summary: how many states have estimates, how many are NaN, range of estimates, smallest/largest n_respondents per state

No poststratification frame needed for this model — it uses raw survey means only.

### 4. `docs/01_baseline_explained.md`
Follow the explanation format in CLAUDE.md exactly (all 6 sections). Key points for this model:

- **Package Selection:** No modeling library needed — just pandas. Explain why this is a deliberate choice: the Baseline establishes the lower bound of what you get with zero statistical sophistication. Reference Ornstein's demonstration that disaggregation yields correlation ≈ 0.05 with truth at n=1,000.
- **Line-by-Line:** Cover every meaningful line in `01_baseline.py` AND the relevant `utils.py` functions it calls.
- **Howe Connection:** This is pure disaggregation — what Howe et al. compare MRP against. No pooling, no borrowing of strength, no demographic adjustment. States with few respondents will have noisy/missing estimates.

### 5. `outputs/estimates/` and `outputs/diagnostics/`
Create these directories (can be empty with a `.gitkeep`).

### 6. `requirements.txt`
List all pip dependencies for models 1–3: pandas, numpy, statsmodels, gpboost.

## Important

- Every `.py` file starts with the author docstring header (Author: KO, with today's date)
- Minimal inline comments — section headers only
- The script should be runnable from the repo root: `python -m models.01_baseline`
