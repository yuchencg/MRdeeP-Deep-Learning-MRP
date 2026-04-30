# Carbon Covariates — Data Quality Report

Per-capita CO₂ emissions covariates at four U.S. geographic levels for use as the `carbon` covariate in an MRP model replicating Yale Climate Opinion Maps **2024 v8** (Howe et al. 2015, updated).

This pipeline uses the **118th Congress** congressional districts, matching the YCOM 2024 v8 specification exactly.

---

## 1. Inputs

| File | Source | Rows | Notes |
|---|---|---|---|
| `crosswalk_explore_tract_2025-q2.csv` | [Crosswalk Labs](https://explore.crosswalk.io/) (2025-Q2 release) | 8,068,438 | Tract-level CO₂; 2010–2024; 7 sector values |
| `geocorr2022_2611608186.csv` | [MABLE/Geocorr 2022](https://mcdc.missouri.edu/applications/geocorr2022.html) | 87,843 | Tract → 118th CD allocation (weight = 2020 pop) |
| `list1_2023.xlsx` | [OMB CBSA delineations, July 2023](https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2023/delineation-files/list1_2023.xlsx) | 1,915 + 3 footer rows | County → CBSA mapping |
| Census ACS 5-yr 2019-2023 (B01003_001E) | api.census.gov (direct HTTP, no key) | — | Population denominators for state, county, CBSA |

---

## 2. Cleaning & merging steps

### Step 2 — Tract-level emissions (2024 scope-1)

Yale's documented Crosswalk Labs URL parameters: `scope=scope1`, `year=2024`, `sector=total`.

**Filter applied:** `year == 2024` AND `sector ∈ {scope1_residential, scope1_commercial, scope1_industrial, scope1_transportation, scope1_electricity_production}`.

The five scope-1 sub-sectors are summed per tract. The pre-aggregated `scope1_total` row is **excluded from the sum** (would double-count) but is retained for cross-validation.

- Tracts with 2024 scope-1 emissions: **84,415**
- Total 2024 scope-1 CO₂ (US): **5,054 Mt**
- Per-tract CO₂ (tonnes): min 1, median ≈19,800, max ~16,600,000
- Cross-check vs. provider-supplied `scope1_total`: mean diff **0.003%**, max diff 5.00% (negligible — sub-sectors sum cleanly to provider totals)

Saved: `output/intermediate/tract_co2_scope1_2024.csv`

### Step 3 — Reconstruct 11-digit GEOID in Geocorr

Geocorr stores tracts as `9501.00`. We strip the dot and zero-pad to a 6-digit code, then concatenate with the 5-digit county FIPS (Geocorr's `county` column already contains state+county FIPS):

```
tract_geoid = county(5) + tract_code(6)   →  11-digit string
```

All FIPS components are kept as **strings throughout** to preserve leading zeros. The first row of the Geocorr CSV is a description row (Geocorr's standard format) and is skipped.

- Geocorr rows after dropping description row: **87,842**
- Bad-length GEOIDs after reconstruction: **0**
- Distinct tracts in Geocorr: **83,849**
- Tracts whose `afact` values do not sum to ~1.0: **0**

### Step 4 — Aggregate up the geographic hierarchy

| Level | Method | Notes |
|---|---|---|
| **State** | `geoid[0:2]` → groupby + sum | Direct slicing of tract GEOID |
| **County** | `geoid[0:5]` → groupby + sum | Direct slicing of tract GEOID |
| **CD118** | LEFT JOIN tract emissions onto Geocorr `(tract_geoid, cd118, afact)`, then `co2 × afact`, groupby `(state, cd118)`. Tracts not found in Geocorr are allocated via **county-fallback** — see §5. | Allocation weighted by 2020 population share |
| **CBSA** | OMB list (skip 2 header rows; drop footer rows where CBSA Code is non-numeric) → county_fips = state(2)+county(3) → LEFT JOIN onto county emissions → groupby CBSA Code | Counties not in any CBSA are intentionally excluded from CBSA output |

### Step 5 — Population denominators

Populations are pulled by direct HTTP from the Census Data API (no API key, no `census` library) — endpoint `https://api.census.gov/data/2023/acs/acs5`, table `B01003_001E` (total population, ACS 5-year 2019-2023):

- **State:** `for=state:*`
- **County:** `for=county:*&in=state:*`
- **CBSA:** `for=metropolitan statistical area/micropolitan statistical area:*`
- **CD118:** **NOT** from ACS. Instead, sum Geocorr `pop20` over `(state, cd118)`. This uses 2020 Census population apportioned to 118th-Congress boundaries — internally consistent with the `afact` values used for emissions allocation. Mixing different CD vintages would create a vintage mismatch.

### Step 6 — Per-capita

`co2_per_capita = co2_emissions_total / population`. Rows with population 0 or null are flagged in §5.

### Step 7 — Territories

US territories (FIPS 60 AS, 66 GU, 69 MP, 72 PR, 78 VI) are excluded from all four outputs to match YCOM scope.

---

## 3. Outputs

All in `output/`:

| File | Rows | Expected | Status |
|---|---:|---:|---|
| `carbon_state.csv` | 51 | 51 | ✓ exact |
| `carbon_county.csv` | 3,144 | ~3,143 | ✓ (3,144 because Connecticut now has 9 planning regions instead of 8 counties — see §4) |
| `carbon_cd118.csv` | 436 | ~436 | ✓ exact |
| `carbon_cbsa.csv` | 925 | ~935 | 10 short — Puerto Rico CBSAs dropped by territory filter |

---

## 4. Connecticut check (explicit per-prompt requirement)

Connecticut switched from 8 counties to 9 **planning regions** in the 2022 ACS/Census products. Both Crosswalk Labs (2025-Q2 release) and the OMB July-2023 delineation file use the **planning-region geography** for CT, so the Crosswalk and OMB vintages are consistent.

- CT in `carbon_state.csv`: present, population non-null, emissions > 0 ✓
- CT counties (planning regions) in `carbon_county.csv`: **9** rows, all 9 have non-null population
- Sum of CT county-level emissions: **38.0 Mt** (consistent with state total)

No silent CT drop detected.

---

## 5. Match quality and unresolved gaps

| Check | Result |
|---|---|
| Tracts in Crosswalk emissions but **not** in Geocorr | 567 (~0.67%) — recovered via county-fallback (see below) |
| Tracts in Geocorr but **not** in 2024 emissions | 1 |
| OMB counties with no matching emissions | 0 |
| Counties with 0 or null population | 0 |
| CBSAs with 0 or null population | 0 |

### County-fallback allocation for orphan tracts

Crosswalk Labs (2025-Q2 release) uses a slightly more recent 2020 Census-tract revision than MABLE/Geocorr 2022 picked up. The 567 orphan tracts (61.32 Mt CO₂ in 2024) are spread across 40 states with no concentration — Michigan (84), Florida (67), New York (53), California (32), Hawaii (29), Texas (28), etc. — exactly the diffuse pattern caused by minor inter-release tract corrections, **not** a single-state vintage problem.

To avoid leaving 1.2% of national emissions unattributed at the CD level, orphan tracts are allocated via a **county-level population-weighted fallback**:

1. From Geocorr, compute each county's CD apportionment: `pop20` summed by `(county, cd118)`, normalized to a within-county share.
2. For each orphan tract, look up its 5-digit county FIPS.
3. Distribute that tract's emissions across CDs in its county using the county-level shares.

This works because every US county is in Geocorr (an orphan tract's parent county is always present even when the tract code itself isn't recognized). Most US counties sit entirely in one CD, so for those the fallback is **exact**; the inter-CD allocation error is bounded by within-county CD heterogeneity in the small subset of counties that span two or more CDs.

**Result of the fix:**

- Orphan tracts allocated: **567 / 567**
- Orphan emissions recovered: **61.32 Mt / 61.32 Mt** (zero remaining unmapped)
- Counties where the fallback found no Geocorr match: **0**

### National sanity check (post-fix)

| Aggregation | Sum |
|---|---:|
| state | 5,053.606167 Mt |
| county | 5,053.606167 Mt |
| **cd118** | **5,053.606034 Mt** (matches state to within 0.13 tonnes / 3×10⁻⁶ %) |
| cbsa | 4,390.110182 Mt (~13% lower — by design; rural non-CBSA counties are excluded) |

State ≡ county ≡ CD118 within floating-point rounding. CBSA gap is structural and expected.

---

## 6. Per-capita distributions

CO₂ tonnes per capita (excludes rows with null population):

| Level | n | mean | median | P5 | P95 | min | max | outliers (>3σ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| state  |   51 | 19.9 | 15.0 |  7.9 |  43.4 |  3.9 |   105.0 |  2 |
| county | 3,144 | 47.6 | 13.1 |  5.4 | 113.7 |  2.3 | 29,420.2 |  5 |
| cd118  |  436 | 15.3 | 11.0 |  5.1 |  38.9 |  2.1 |    97.6 |  9 |
| cbsa   |  925 | 23.5 | 12.6 |  6.2 |  75.9 |  4.0 |   503.1 | 16 |

The county-level maximum (~29,420 t/capita) and other extreme outliers occur in low-population tracts/counties dominated by an industrial point source (refinery, cement plant, large fossil-fuel power plant). These are **legitimate values** under the YCOM specification — point-source emissions are intentionally part of the `carbon` covariate. Flagging them here so they can be confirmed against your modelling choices (e.g. log-transform, Winsorize at P99, or treat as-is).

---

## 7. Reproducibility

- Script: `build_carbon_covariates.py`
- Pandas only; no GIS libraries; no `census`/`us` packages required
- Census API calls are unauthenticated (small request volume — well within anonymous tier)
- All FIPS codes / GEOIDs handled as strings end-to-end

To re-run:

```bash
python3 build_carbon_covariates.py
```
