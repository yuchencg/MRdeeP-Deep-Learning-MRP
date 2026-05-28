# MRdeeP — Project Structure & Pipeline Guide

This document walks through every folder and file in the project root, explains what it does, and shows how the pieces connect to form the end-to-end MRdeeP pipeline.

For project goals, model definitions, and methodology, see [CLAUDE.md](../CLAUDE.md). For per-model technical explanations, see the `*_explained.md` files in this folder.

---

## 1. High-level pipeline

The project moves data through three phases:

```
                  ┌─────────────────────────────────┐
                  │  ACS census microdata           │
                  └───────────────┬─────────────────┘
                                  │
                  post_stratification_frame/
                                  │
                                  ▼
                  ┌─────────────────────────────────┐
                  │  poststrat_state.csv            │
                  │  poststrat_county.csv           │
                  │  (cell-level population counts) │
                  └───────────────┬─────────────────┘
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       │                          │                          │
       ▼                          ▼                          ▼
┌──────────────┐         ┌──────────────┐          ┌──────────────────┐
│ test_data/   │         │ CES/         │          │ state-level      │
│ (synthetic   │         │ (real CES    │          │ covariates       │
│  survey)     │         │  survey)     │          │ (Howe et al.)    │
└──────┬───────┘         └──────┬───────┘          └──────┬───────────┘
       │                        │                         │
       │                        │                         │
       ▼                        ▼                         │
┌──────────────┐         ┌──────────────────┐             │
│ model_test/  │◄────────│ model_run_ces/   │◄────────────┘
│ (validate    │         │ (production runs │
│  pipeline on │         │  on real data)   │
│  synthetic)  │         └────────┬─────────┘
└──────┬───────┘                  │
       │                          │
       └────────────┬─────────────┘
                    ▼
            ┌──────────────┐
            │ outputs/     │ (per-model estimate CSVs + diagnostics)
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ map_app/     │ (Dash dashboard for visualization)
            └──────────────┘
```

**Phase 1 — Pipeline validation:** synthetic survey (`test_data/`) is fed through all six models (`model_test/`) to prove the modeling code end-to-end before the real survey arrives.

**Phase 2 — Production runs:** real CES survey data (`CES/`) is sampled and run through the same six models (`model_run_ces/`).

**Phase 3 — Dashboard:** model estimates are loaded into `map_app/` to recreate the Yale YPCCC climate-opinion map and compare across models.

---

## 2. Folder-by-folder reference

### `CLAUDE.md`
Master project spec. Covers goals, the six modeling approaches, Howe et al. (2015) methodology, data sources, repo conventions, and code style rules. Read this first.

### `requirements.txt`
Python dependencies for the modeling pipeline.

### `.gitignore`
Standard ignores.

---

### `post_stratification_frame/` — ACS poststratification builder

Builds the population frame every MRP model uses for poststratification: a table of cells (race × education × gender × geography) with a population count `N` per cell.

| File | Purpose |
|------|---------|
| `extract_acs.ipynb` | Pull ACS 5-year PUMS microdata |
| `acs_post_stratification.ipynb` | Cross-tabulate demographics × geography, sum PWGTP weights to get cell counts |
| `poststrat_state.csv` | ~1,632 rows = 51 states × 32 demographic cells |
| `poststrat_county.csv` | ~100,000 rows = ~3,143 counties × demographic cells |

These two CSVs are loaded by every model script via `utils.load_poststrat_state()` (see `models/utils.py` spec in CLAUDE.md).

---

### `test_data/` — Synthetic survey (Phase 1)

~12,000 synthetic respondents mimicking the planned QSF survey instrument. Used to build and validate the pipeline before the real survey is fielded.

| Path | Purpose |
|------|---------|
| `raw/climate_survey_responses.csv` | Raw synthetic survey |
| `raw/carbon_state.csv` | Per-capita CO₂ emissions by state (Vulcan/EPA) |
| `raw/pres_state.csv` | Democratic two-party presidential vote share |
| `raw/drive_state.csv` | Share of workers driving alone (ACS B08301) |
| `raw/samesex_state.csv` | Same-sex household share (ACS B11009) |
| `recode_survey.py` | Recodes ~30 survey items into binary outcome variables (e.g. `happening_bin`, `worried_bin`) |
| `processed/climate_survey_responses_recoded.csv` | Output of `recode_survey.py` |
| `processed/poststrat_state.csv` | Local copy of the state frame for model loading |
| `processed/poststrat_county.csv` | Local copy of the county frame |
| `variable_summary_carbon.md` | Documentation of the carbon covariate |
| `variable_summary_pres_s.md` | Documentation of the presidential vote share covariate |

---

### `CES/` — Real CES survey (Phase 2 input)

Cooperative Election Study 2022 data. Now the active production survey source.

| Path | Purpose |
|------|---------|
| `CES_Info/CES 2022 - Sheet1(1).csv` | Variable codebook |
| `CES_Info/ces22_common_pre_qx-1.pdf` | Pre-election questionnaire |
| `CES_Info/ces22_common_post_qx-1.pdf` | Post-election questionnaire |
| `data_raw/questions_to_include.csv` | Selected CES variables for the analysis |
| `data_raw/fips2county.tsv` | County FIPS lookup |
| `data_raw/state_county_city_FIPS_reference_table_20260509.csv` | Geography reference |
| `data_raw/co-est2025-pop.xlsx` | County population estimates |
| `build_filtered_responses.py` | Filters CES to selected questions, recodes to binary outcomes, attaches geography |
| `draw_sample_responses.py` | Draws 1k / 3k / 5k respondent subsets for model runs |
| `data_processed/filtered_responses_preprocessed.csv` | Full filtered CES |
| `data_processed/filtered_responses_preprocessing.dta` | Stata version |
| `data_processed/sample_ces_{1000,3000,5000}.csv` | Three subset samples |
| `notebook/data_exploration.ipynb` | Exploratory CES analysis |

---

### `model_test/` — Pipeline validation on synthetic data

Runs all six models against `test_data/`. Originally the only modeling folder; now serves as the reference implementation.

**`model_test/state/`** — state-level estimation
| File | Model |
|------|-------|
| `01_baseline.py` / `1_baseline.ipynb` | Naive disaggregation baseline |
| `02_glm_mrp.py` / `2_glm_mrp.ipynb` | Frequentist GLM MRP |
| `03_glmer_mrp.py` / `3_glmer_mrp.ipynb` | Frequentist GLMM MRP (`gpboost`) |
| `4_GLMERStan_state.ipynb` | Bayesian GLMM MRP (NUTS via `bambi`/`pymc`) |
| `5_SRP.ipynb` | Super Learner ensemble |
| `6_MRdeeP.ipynb` | Deep learning autoencoder + WGAN-GP |
| `model_comparison_state.ipynb` | Side-by-side comparison across all six |
| `utils.py` | Shared loaders and `poststratify()` helper |

**`model_test/county/`** — same six models adapted to county geography (notebooks only).

**`model_test/outputs/`**
- `estimates/` — one CSV per model (e.g. `glm_mrp_state_estimates.csv`), plus PNG choropleths and a comparison figure (`fig5_happening_comparison.png`)
- `diagnostics/` — text summaries (`baseline_summary.txt`, `glm_mrp_summary.txt`, `glmer_mrp_summary.txt`) with convergence and fit info

---

### `model_run_ces/` — Production runs on real CES data

Mirrors `model_test/state/` but consumes real CES samples from `CES/data_processed/`.

**`model_run_ces/sample1_state/`** — first sample subset, state-level
- `01_baseline.py` through `06_mrdeep.ipynb` — the six models
- `utils.py` — shared utilities (mirrors `model_test/state/utils.py`)
- `outputs/` — per-model estimate CSVs for this sample

Per recent commits, this is the active workstream. Additional `sampleN_state/` folders may be added for the 3k and 5k subsets.

---

### `model_test_updated/`
Currently empty (placeholder for an in-progress refresh of the validation pipeline).

---

### `outputs/`
Top-level outputs directory (currently empty — per-model outputs live inside `model_test/outputs/` and `model_run_ces/sample1_state/outputs/`).

---

### `docs/` — Per-model explanations
| File | Covers |
|------|--------|
| `01_baseline_explained.md` | Naive disaggregation: math, code, limitations |
| `02_glm_mrp_explained.md` | Frequentist GLM MRP: specification, poststratification |
| `03_glmer_mrp_explained.md` | Frequentist GLMM MRP: random effects, partial pooling |
| `PROJECT_STRUCTURE.md` | This file |

Each per-model doc follows the six-section structure defined in CLAUDE.md (Big Picture → Package Selection → Line-by-Line → Model Specification → Connection to Howe et al. → Output Interpretation).

---

### `map_app/` — Dashboard (Phase 3)

Dash/Plotly app that recreates the Yale YPCCC climate-opinion map and overlays the project's six model estimates.

| File | Role |
|------|------|
| `app.py` | Dash app entry point |
| `map_builder.py` | Choropleth construction |
| `data_loader.py` | Loads model estimate CSVs |
| `constants.py` | State/region metadata, color scales |
| `components/` | Dash UI components (model selector, legend, etc.) |
| `assets/` | Static CSS / images |
| `data/` | Bundled estimate snapshots for the app |
| `tests/` | App tests |
| `README.md` | Run instructions |
| `requirements.txt` | Dashboard-specific deps |

The three deliverables this app provides (per CLAUDE.md):
1. Recreate the published Yale YPCCC dashboard
2. Filter-by-model selector across the six models
3. Per-state difference between model estimate and Yale published value

---

## 3. How data flows through a single model run

Using GLMER-MRP at the state level as an example:

1. **Load survey** — `utils.load_survey('happening_bin')` reads the recoded survey CSV and drops NaN rows.
2. **Load poststratification frame** — `utils.load_poststrat_state()` loads `poststrat_state.csv` and merges in the four Howe covariates (`carbon_state.csv`, `pres_state.csv`, `drive_state.csv`, `samesex_state.csv`) plus a state→region lookup.
3. **Fit model** — `03_glmer_mrp.py` fits a GLMM with random intercepts on demographics + geography and fixed effects on the four state covariates.
4. **Predict** — apply the fitted model to every cell in the poststratification frame to get a predicted probability per cell.
5. **Poststratify** — `utils.poststratify(frame, 'pred')` computes the population-weighted mean by state.
6. **Save** — `utils.save_estimates(df, 'glmer_mrp')` writes `outputs/estimates/glmer_mrp_state_estimates.csv`.
7. **Visualize** — `map_app/` loads the CSV and renders the choropleth.

The other five models follow the same load → fit → predict → poststratify → save shape; only step 3 changes.

---

## 4. Two parallel modeling tracks — why both exist

| Track | Input | Purpose |
|-------|-------|---------|
| `model_test/` | Synthetic survey (`test_data/`) | Prove the pipeline end-to-end with known structure before real data arrives |
| `model_run_ces/` | Real CES samples (`CES/data_processed/sample_ces_*.csv`) | Production estimates that feed the dashboard |

Keeping them separate means changes to the production track don't break the validation reference, and synthetic-vs-real comparisons remain reproducible.

---

## 5. Where to look for what

| If you want to… | Go to |
|------------------|-------|
| Understand project goals or methodology | [CLAUDE.md](../CLAUDE.md) |
| See how a specific model works | `docs/0N_*_explained.md` |
| Rebuild the ACS frame | `post_stratification_frame/` |
| Inspect or recode the synthetic survey | `test_data/` |
| Inspect or sample the real CES survey | `CES/` |
| Run a model on synthetic data | `model_test/state/` or `model_test/county/` |
| Run a model on real CES data | `model_run_ces/sample1_state/` |
| Compare estimates across models | `model_test/state/model_comparison_state.ipynb` |
| Visualize state estimates on a map | `map_app/` |
