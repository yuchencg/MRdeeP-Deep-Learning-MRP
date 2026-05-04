# CLAUDE.md — MRP Model Comparison Project

## Project Goal

Compare 6 methods of small-area estimation on a national climate public opinion survey (~1,200 respondents). Each method produces **state-level estimates** of public opinion on climate questions. Since there is no ground truth, we compare the variance and distribution of predictions across models.

The modeling approach follows **Howe et al. (2015)** — "Geographic variation in opinions on climate change at state and local scales in the USA" (*Nature Climate Change*). The full reference guide is at `docs/Howe_2015_MRP_Guide.md`.

---

## Models (Build Order)

Scripts are built one at a time. Current phase: models 1–3.

| # | Model | Type | Python Library | Estimation |
|---|-------|------|----------------|------------|
| 1 | **Baseline (Naive)** | Simple disaggregation | `pandas` | Raw state-level means — no modeling |
| 2 | **GLM-MRP** | Frequentist GLM | `statsmodels` | Logistic regression, MLE, logit link |
| 3 | **GLMER-MRP** | Frequentist GLMM | `gpboost` | Logistic mixed-effects, Laplace approximation, logit link |
| 4 | Stan-MRP | Bayesian GLMM | `bambi` / `pymc` | NUTS sampler (future) |
| 5 | SRP | Ensemble | TBD | Super Learner stack (future) |
| 6 | MRdeeP | Deep Learning | `MRdeeP` package | Autoencoder + WGAN-GP (future) |

---

## Repository Structure

All new code goes inside `A.MRdeeP-Deep-Learning--MRP/`. Items marked ✓ already exist. Items marked ★ are new.

```
A.MRdeeP-Deep-Learning--MRP/
├── post_stratification_frame/          # ✓ ACS census frame builders
│   ├── extract_acs.ipynb
│   └── acs_post_stratification.ipynb
│
├── test_data/                          # ✓ Survey data + covariates
│   ├── raw/
│   │   ├── climate_survey_responses.csv
│   │   ├── carbon_state.csv            #   51 rows: state_fips, co2_per_capita
│   │   ├── pres_state.csv              #   51 rows: state_fips, dem_share_two_party
│   │   └── ...
│   ├── processed/
│   │   ├── climate_survey_responses_recoded.csv   # ~1,200 rows × 38 cols
│   │   ├── poststrat_state.csv                    # 1,632 rows (51 × 32 demo cells)
│   │   └── poststrat_county.csv                   # 99,940 rows
│   ├── recode_survey.py
│   └── 4_GLMERStan_climate.ipynb
│
├── models/                             # ★ One script per model
│   ├── __init__.py
│   ├── utils.py                        # ★ Shared data loading + poststratification
│   ├── 01_baseline.py                  # ★ Naive disaggregation
│   ├── 02_glm_mrp.py                  # ★ Logistic GLM + poststratification
│   └── 03_glmer_mrp.py                # ★ GLMM via gpboost + poststratification
│
├── outputs/                            # ★ All model outputs
│   ├── estimates/                      # ★ State-level estimate CSVs
│   └── diagnostics/                    # ★ Model summaries, logs
│
├── docs/                               # ★ Model explanation files
│   ├── Howe_2015_MRP_Guide.md         # ★ Reference guide (copy from project root)
│   ├── 01_baseline_explained.md
│   ├── 02_glm_mrp_explained.md
│   └── 03_glmer_mrp_explained.md
│
├── run_all.py                          # ★ Master runner (runs models 1–3 sequentially)
└── requirements.txt                    # ★ pip dependencies
```

---

## Data Schema

### Survey Data: `test_data/processed/climate_survey_responses_recoded.csv`

~1,200 rows. Each row is one respondent.

| Column | Type | Values | Role |
|--------|------|--------|------|
| `state_fips` | string | 2-digit zero-padded FIPS ("01"–"56") | Geographic grouping |
| `state_name` | string | Full state name | Label |
| `gender` | int | 1=Male, 2=Female | Demographic predictor |
| `race4` | string | "White", "Black", "Hispanic", "Other" | Demographic predictor |
| `educ_category` | int | 1=Less than HS, 2=HS, 3=Some college, 4=BA+ | Demographic predictor |
| `age_group` | int | 1=18-34, 2=35-54, 3=55+ | Reference only (not in poststrat) |
| `region9` | string | 9 Census divisions (e.g. "Pacific") | Geographic grouping |
| `happening_bin` | float | 0, 1, or NaN (don't-know) | **Outcome variable** |
| *(29 more `_bin` columns)* | float | 0, 1, or NaN | Future outcomes |

### Poststratification Frame: `test_data/processed/poststrat_state.csv`

1,632 rows (51 states × 2 genders × 4 races × 4 education levels).

| Column | Type | Description |
|--------|------|-------------|
| `state_fips` | string | 2-digit FIPS |
| `gender` | int | 1=Male, 2=Female |
| `race4` | string | "White", "Black", "Hispanic", "Other" |
| `educ_category` | int | 1–4 |
| `N` | float | ACS weighted population count (`PWGTP` sum) |

### State-Level Covariates

Four files in `test_data/raw/` are merged onto both survey data and poststrat frame by `state_fips`. These are the four Howe et al. (2015) covariates:

| File | Key Column | Covariate Column | Description |
|------|-----------|-------------------|-------------|
| `carbon_state.csv` | `state_fips` | `co2_per_capita` | Per-capita CO₂ emissions (tons) |
| `pres_state.csv` | `state_fips` | `dem_share_two_party` | 2024 Democratic vote share D/(D+R) |
| `drive_state.csv` | `state_fips` | `drive_alone_share` | Share of workers who drove alone to work (ACS 2023 5-year, table B08301) |
| `samesex_state.csv` | `state_fips` | `samesex_share` | Share of households that are same-sex coupled (ACS 2023 5-year, table B11009; B11009_004 + B11009_009) |

---

## Model Specifications

### Model 1: Baseline (Naive Disaggregation)

No model. For each state, compute the raw mean of `happening_bin` from survey respondents in that state. States with zero respondents get `NaN`.

```
estimate_s = mean(happening_bin) for respondents where state_fips == s
```

### Model 2: GLM-MRP (Logistic Regression + Poststratification)

Frequentist logistic regression with **all predictors as fixed effects** (no random effects, no partial pooling). Poststratified using ACS population weights.

**Formula (statsmodels GLM):**
```
happening_bin ~ C(gender) + C(race4) + C(educ_category) + C(region9) + co2_per_capita + dem_share_two_party + drive_alone_share + samesex_share
```

- Family: Binomial, Link: Logit
- All demographic and geographic variables enter as categorical dummies (`C()`)
- State-level covariates enter as continuous fixed effects
- `state_fips` does NOT enter the model (no state fixed/random effects — state variation comes only through region + covariates)
- After fitting, predict P(y=1) for every cell in the poststrat frame, then take weighted average by `N` within each state

### Model 3: GLMER-MRP (Mixed-Effects Logistic Regression + Poststratification)

Frequentist GLMM via `gpboost` with **Laplace approximation** (equivalent to R's `lme4::glmer`). Follows Howe et al. (2015) specification.

**Formula (conceptual `lme4` equivalent):**
```
happening_bin ~ co2_per_capita + dem_share_two_party + drive_alone_share + samesex_share +
                (1|race4) + (1|educ_category) + (1|gender) + (1|region9) + (1|state_fips)
```

- Likelihood: `bernoulli_logit` (Bernoulli with logit link)
- Random intercepts on: `race4` (4 levels), `educ_category` (4 levels), `gender` (2 levels), `region9` (9 levels), `state_fips` (51 levels)
- Fixed effects: intercept + all four Howe covariates (`co2_per_capita`, `dem_share_two_party`, `drive_alone_share`, `samesex_share`)
- State-level covariates shift the mean of each state's random intercept (Howe-style)
- After fitting, predict P(y=1) for every poststrat cell using `gp_model.predict()`, then take weighted average by `N` within each state

**gpboost implementation notes:**
- `group_data`: matrix with columns for each random-effect grouping variable
- `X`: design matrix with intercept column + continuous covariates
- `likelihood="bernoulli_logit"` for logit link (not default probit)
- Use `predict_response=True` to get probabilities (not latent scale)

---

## Shared Utilities: `models/utils.py`

This module is imported by every model script. It handles:

1. **`load_survey(outcome_var)`** — Load recoded survey CSV, drop NaN rows for the specified outcome, return clean DataFrame
2. **`load_poststrat_state()`** — Load state poststrat frame, merge on all four state-level covariates (`carbon_state.csv`, `pres_state.csv`, `drive_state.csv`, `samesex_state.csv`), merge on `region9` (from a state→region lookup), return DataFrame with columns: `state_fips`, `gender`, `race4`, `educ_category`, `region9`, `co2_per_capita`, `dem_share_two_party`, `drive_alone_share`, `samesex_share`, `N`
3. **`poststratify(poststrat_df, prob_col)`** — Given poststrat frame with a column of predicted probabilities, compute weighted mean by state. Return DataFrame: `state_fips`, `state_name`, `estimate`
4. **`save_estimates(estimates_df, model_name)`** — Save to `outputs/estimates/{model_name}_state_estimates.csv`
5. **State-to-region mapping** — A dictionary or small CSV mapping each `state_fips` to one of the 9 Census divisions. This is needed because the poststrat frame may not include `region9`.

---

## Code Conventions

### File Headers

Every `.py` file must begin with an author docstring:

```python
"""
<Brief description of what this script does>.
Author: KO
Created: YYYY-MM-DD
"""
```

### Comments

Minimal. Section headers for logical blocks. Key decisions commented. Detailed explanations go in the corresponding `docs/*_explained.md` file, not inline.

### Output File Naming

```
outputs/estimates/{model_name}_state_estimates.csv
```

Where `model_name` is: `baseline`, `glm_mrp`, `glmer_mrp`

Each CSV has columns: `state_fips`, `state_name`, `estimate`, `n_respondents`

### Diagnostics

```
outputs/diagnostics/{model_name}_summary.txt
```

Plain text model summary — coefficients, sample sizes, convergence info. Printed to console AND saved to file.

---

## Dependencies

```
pandas
numpy
statsmodels
gpboost
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Howe-style spec (not simulation spec) | Matches the published MRP methodology; uses state-level covariates and nesting |
| All four Howe covariates: `co2_per_capita`, `dem_share_two_party`, `drive_alone_share`, `samesex_share` | Mirrors Howe et al. (2015). `drive` and `samesex` pulled from ACS 2023 5-year via Census Data API (tables B08301 and B11009) |
| `gpboost` for GLMER-MRP (not `pymer4`) | Pure Python, no R dependency, same Laplace approximation as `lme4` |
| `statsmodels` for GLM-MRP | Standard, well-tested for fixed-effects GLM; only the GLMM is problematic in statsmodels |
| State-level only (no county yet) | County poststrat frame needs covariate merging; add later |
| `happening_bin` only | Start with one outcome; loop over all 30 later |
| NaN rows dropped per-outcome | Don't-know responses excluded from modeling (not coded 0 or 1) |
| Shared `utils.py` | DRY principle — data loading and poststratification logic defined once |

---

## Explanation File Format

Each model gets a `docs/XX_model_explained.md`. See `code_exp.md` for line-by-line style reference and `Howe_2015_MRP_Guide.md` for conceptual depth reference.

### Required Sections

**1. Big Picture: What This Script Accomplishes**
- 1 paragraph summarizing purpose
- Numbered list of conceptual steps (matching `code_exp.md` format)

**2. Package Selection & Design Choices**
- Why this specific Python library was chosen over alternatives (e.g. why `gpboost` over `pymer4`, `statsmodels.BinomialBayesMixedGLM`, or `bambi`)
- What the library does under the hood (estimation method, link function, optimization algorithm)
- How the Python implementation relates to the R equivalent (e.g. `gpboost` Laplace ↔ `lme4::glmer()` Laplace)
- Known differences or trade-offs vs. the R implementation
- Why specific model parameters/options were set the way they are (e.g. `likelihood="bernoulli_logit"` not the default probit)
- For Baseline: why no modeling is a deliberate choice, and what it reveals as a benchmark

**3. Line-by-Line Technical Breakdown**
- Table format: `| Code | Technical Explanation |`
- Every meaningful line gets a row
- Explanations should be precise and technical (match `code_exp.md` tone)
- Cover: what each function call does, what its arguments control, what data structures are produced

**4. Model Specification** (for models 2+)
- The statistical model in mathematical notation
- How each term maps to the code
- What the model produces (coefficients, variance components, predictions)
- How predictions flow into poststratification

**5. Connection to Howe et al. (2015)**
- How this model relates to the Howe MRP framework
- What this model captures vs. what it misses compared to the full Howe specification
- Where this model sits on the pooling spectrum (no pooling → partial pooling → complete pooling)

**6. Output Interpretation**
- What the output CSV contains and what each column means
- How to interpret the state-level estimates
- Caveats or limitations specific to this model (e.g. states with zero respondents in Baseline)

### Style Notes

- Match `code_exp.md` for line-by-line table tone: precise, technical, no filler
- Match `Howe_2015_MRP_Guide.md` for conceptual depth: walk the reader through *why*, not just *what*
- The Package Selection section should read as a justified decision log — someone reviewing the project should understand why each tool was chosen and what alternatives were considered
