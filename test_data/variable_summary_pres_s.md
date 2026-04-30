# Variable Summary: pres_s (Democratic Presidential Vote Share)

| Field | Value |
|---|---|
| Variable name in model | `pres_s` (state), `pres_c` (county), `pres_cd` (CD), `pres_m` (CBSA) |
| Used in | Both Howe et al. 2015 & YCOM 2024 v8 |
| Form in the model | Continuous proportion at the geographic unit level; enters linearly via fixed coefficient γ^pres |
| Howe 2015 source | 2008 Presidential Democratic vote share (election returns; specific aggregator not named in paper) |
| YCOM 2024 v8 source | 2020 Presidential vote share, Redistricting Data Hub |
| Source used here | (1) MIT Election Data and Science Lab `countypres_2000-2024` for state/county/CBSA; (2) The Downballot for 119th-CD-level percentages |
| Election year produced | **2024** |
| Departure from YCOM v8 | Aggregator changed (RDH → MEDSL + Downballot); election year (updated to 2024 vs YCOM v8 which uses 2020); CD vintage 119th vs YCOM's 118th |
| Geographic levels produced | state, county, CD119, CBSA |
| Aggregation method | MEDSL counties summed by `state_fips[:2]` for state; county FIPS passthrough for county; counties→CBSA via OMB July-2023 delineation; CD-level read directly from Downballot (no aggregation needed) |
| Vote share formulas produced | `dem_share_two_party = D/(D+R)`; `dem_share_total = D/(D+R+L+other)` |
| CD output limitation | Downballot publishes rounded percentages only, not vote counts; CD CSV has share columns only |
| Source ambiguity | YCOM phrasing "percent who voted Democrat" does not specify denominator; both variants computed |
| Run timestamp | 2026-04-26 15:54:39 |
| Output files | `pres_state.csv`, `pres_county.csv`, `pres_cd119.csv`, `pres_cbsa.csv` |

## Output row counts

| File | Rows |
|---|---:|
| `pres_state.csv` | 51 |
| `pres_county.csv` | 3,145 |
| `pres_cd119.csv` | 435 |
| `pres_cbsa.csv` | 925 |

## National totals (from MEDSL county aggregation)

| Party bucket | Votes |
|---|---:|
| Democrat   |     74,990,538 |
| Republican |     77,267,503 |
| Libertarian|        649,741 |
| Other      |      2,242,512 |
| **Total**  | **   155,150,294** |

## State-level dem_share_two_party extremes

| Extreme | State | Value |
|---|---|---:|
| Min | WY (Wyoming) | 26.52% |
| Max | DC (District of Columbia) | 93.31% |

## CD119-level dem_share_two_party extremes

| Extreme | District | Value |
|---|---|---:|
| Min | AL-04 | 16.16% |
| Max | PA-03 | 88.89% |

## CBSA coverage

CBSAs cover 92.9% of national votes (144,202,632 of 155,150,294). Counties not in any CBSA (rural / non-CBSA) are excluded by design — same convention as the carbon covariate.
