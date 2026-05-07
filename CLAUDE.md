# CLAUDE.md — MRdeeP: Deep Learning Extension of MRP for Climate Opinion Estimation

## Project Goal

This project develops and benchmarks **MRdeeP** — a deep learning extension of Multilevel Regression and Poststratification (MRP) — that enables simultaneous estimation of multiple climate change and policy opinion survey outcomes at subnational levels. We collect climate change and policy opinion via a national survey  and estimate state- and local-level public opinion using six modeling approaches of increasing complexity. The model predictors and structure follow the Yale Program on Climate Change Communication framework established in Howe et al. (2015), *"Geographic variation in opinions on climate change at state and local scales in the USA"* (*Nature Climate Change*).

The MRdeeP model is tested against five comparison models (naive baseline through ensemble methods) to evaluate whether deep learning can improve small-area estimation of public opinion.

---

## Key References (in project files)

| File | Citation | Role |
|------|----------|------|
| `Howe_2015.pdf` | Howe, Mildenberger, Marlon & Leiserowitz (2015). *Nature Climate Change* | Primary methodological reference — MRP specification, covariates, poststratification, validation |
| `Change_in_US_statelevel_public_opinion_2022.pdf` | Marlon et al. (2022). *Environ. Res. Lett.* | Updated MRT model with time-varying covariates, state-level trends 2008–2020 |
| `climate_worry_in_US_frontline_communities_2026.pdf` | (2026). *One Earth* | Frontline community worry analysis, CVI/CEJST vulnerability tools |

---

## Local File Paths

| Purpose | Path |
|---------|------|
| Capstone project root | `/Users/carmenk/Documents/CSS/Capstone` |
| Testing pipeline repo | `/Users/carmenk/Documents/GitHub/MRdeeP-Deep-Learning-MRP` |

---

## Pipeline Overview

The project has three phases: (1) build and test the modeling pipeline on synthetic data, (2) run all models on real survey data, and (3) visualize and compare results via dashboard.

### Phase 1: Build Pipeline with Synthetic Data

Before the real survey is fielded, we generated ~12,000 synthetic observations that mimic responses to the QSF survey instrument. This synthetic data is used to build and validate the full modeling pipeline end-to-end.

Steps completed:
- Preprocess ACS census data to produce the poststratification frame (race × education × gender × geography)
- Recode synthetic survey responses into binary outcome variables
- Run all modeling approaches on the synthetic data + ACS poststratification frame

### Phase 2: Six Model Comparison

All six models use the same predictor structure from Howe et al. (2015): individual-level demographics (gender, race, education) as categorical predictors, state/region geographic groupings, and four state-level covariates (CO₂ per capita, Democratic vote share, drive-alone share, same-sex household share). Models differ in estimation strategy.

| # | Model | Type | Estimation | Pooling |
|---|-------|------|------------|---------|
| 1 | **Naive Baseline** | Simple disaggregation | Raw state-level means, no modeling | No pooling |
| 2 | **GLM-MRP** | Frequentist GLM | Logistic regression (MLE), all fixed effects, poststratified | Complete pooling (within predictors) |
| 3 | **GLMER-MRP** | Frequentist GLMM | Mixed-effects logistic regression (Laplace approximation via `gpboost`), random intercepts on demographics + geography, poststratified | Partial pooling |
| 4 | **GLMER-Stan (Bayesian MRP)** | Bayesian GLMM | Full Bayesian estimation via NUTS sampler (`bambi`/`pymc` or `rstanarm`), posterior distributions, uncertainty quantification | Partial pooling + posterior uncertainty |
| 5 | **SRP** | Ensemble | Super Learner stack — combines multiple base learners with cross-validated weighting | Data-adaptive |
| 6 | **MRdeeP** | Deep Learning | Autoencoder + WGAN-GP architecture for simultaneous multi-outcome estimation | Learned representations |


### Phase 3: Dashboard and Comparison

Three deliverables:
1. **Recreate the Yale YPCCC dashboard** using baseline data from the published Yale Climate Opinion Maps
2. **Feed model predictions into the dashboard** with a filter-by-model selector — users can toggle between all six models' state-level estimates
3. **Compute and display the difference** between our model estimates and the Yale published values for each state

---

## MRP Methodology (from Howe et al. 2015)

MRP comprises two stages:

**Stage 1 — Multilevel Regression:** Individual survey responses are modeled as a function of individual-level demographics and geography-level covariates. The Howe specification uses:
- Random intercepts on race (4 levels), education (4 levels), gender (2 levels), census region (9 levels), and state (51 levels)
- Fixed effects for four state-level covariates: percentage who drive alone, percentage same-sex households, per-capita CO₂ emissions, and Democratic presidential vote share
- State random effects are nested within region random effects, with state-level covariates shifting the state intercept mean

**Stage 2 — Poststratification:** Fitted probability estimates for each demographic-geographic cell type are weighted by ACS census-derived population counts. The MRP estimate for any geographic subunit is the population-weighted average of cell-level predictions.

---

## Data

### Survey Data
~1,200 respondents (synthetic for pipeline testing; real survey TBD). Each respondent is geolocated and has binary-coded responses for ~30 climate opinion outcomes (e.g., `happening_bin`, `human_caused_bin`, `regulate_co2_bin`, `worried_bin`, etc.). Survey questions follow the Yale/George Mason CCOM instrument.

### Poststratification Frame
Built from ACS 5-year estimates. Cross-tabulates race × education × gender for all 51 states (1,632 state-level cells) and all ~3,143 counties (~100,000 county-level cells). Population counts (`N`) come from ACS person weights (`PWGTP`).

### State-Level Covariates (Howe et al. 2015)

| Covariate | Source | Description |
|-----------|--------|-------------|
| `co2_per_capita` | Vulcan Project / EPA | Per-capita CO₂ emissions (tons) |
| `dem_share_two_party` | Election data | Democratic presidential vote share D/(D+R) |
| `drive_alone_share` | ACS 5-year, table B08301 | Share of workers who drive alone to work |
| `samesex_share` | ACS 5-year, table B11009 | Share of households that are same-sex coupled |

---

## Repository Structure

```
MRdeeP-Deep-Learning-MRP/          (testing pipeline repo)
├── post_stratification_frame/      # ACS census frame builders
│   ├── extract_acs.ipynb
│   └── acs_post_stratification.ipynb
│
├── test_data/                      # Survey data + covariates
│   ├── raw/                        # State-level covariate CSVs
│   │   ├── climate_survey_responses.csv
│   │   ├── carbon_state.csv
│   │   ├── pres_state.csv
│   │   ├── drive_state.csv
│   │   └── samesex_state.csv
│   ├── processed/
│   │   ├── climate_survey_responses_recoded.csv
│   │   ├── poststrat_state.csv     # 1,632 rows (51 × 32 demo cells)
│   │   └── poststrat_county.csv    # ~100,000 rows
│   ├── recode_survey.py
│   └── 4_GLMERStan_climate.ipynb
│
├── models/                         # One script per model
│   ├── __init__.py
│   ├── utils.py                    # Shared data loading + poststratification
│   ├── 01_baseline.py
│   ├── 02_glm_mrp.py
│   ├── 03_glmer_mrp.py
│   ├── 04_glmer_stan.py           
│   ├── 05_srp.py                  
│   └── 06_mrdeep.py            
│
├── outputs/
│   ├── estimates/                  # State-level estimate CSVs per model
│   └── diagnostics/                # Model summaries, convergence logs
│
├── docs/                           # Model explanation files
│   ├── Howe_2015_MRP_Guide.md
│   ├── 01_baseline_explained.md
│   ├── 02_glm_mrp_explained.md
│   └── 03_glmer_mrp_explained.md
│
├── dashboard/                      # (future) Visualization app
│
├── run_all.py                      # Master runner
└── requirements.txt
```
---

## Shared Utilities: `models/utils.py`

Imported by every model script:

1. **`load_survey(outcome_var)`** — Load recoded survey CSV, drop NaN rows for the specified outcome
2. **`load_poststrat_state()`** — Load state poststrat frame, merge all four state-level covariates + region9 lookup
3. **`poststratify(poststrat_df, prob_col)`** — Compute population-weighted mean by state from cell-level predictions
4. **`save_estimates(estimates_df, model_name)`** — Save to `outputs/estimates/{model_name}_state_estimates.csv`
5. **State-to-region mapping** — Dictionary mapping each `state_fips` to one of 9 Census divisions

---

## Code Conventions

No authorship or attribution in git comments

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.


## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"


### Comments
Minimal inline. Section headers for logical blocks. Detailed explanations go in `docs/*_explained.md`.



Required sections:
1. **Big Picture** — Purpose + numbered conceptual steps
2. **Package Selection & Design Choices** — Why this library, what it does under the hood, relation to R equivalents, trade-offs
3. **Line-by-Line Technical Breakdown** — Table: `| Code | Technical Explanation |`
4. **Model Specification** — Math notation, code mapping, prediction flow
5. **Connection to Howe et al. (2015)** — Where this model sits on the pooling spectrum
6. **Output Interpretation** — What the CSV contains, caveats, limitations



