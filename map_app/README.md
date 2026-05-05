# MRP Method Comparison Map (v1)

Interactive Dash app for visualizing state-level public-opinion estimates from a
climate survey, with side-by-side comparison across modeling methods (Baseline,
GLM-MRP, GLMER-MRP, and — once their CSVs land — Stan-MRP, SRP, MRdeeP).

Inspired by the [Yale Climate Opinion Maps (YCOM)](https://climatecommunication.yale.edu/visualizations-data/ycom-us/).

## Run locally

```bash
cd map_app
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:8050>.

## Layout

```
map_app/
├── app.py                  # Dash entrypoint, layout, callbacks
├── data_loader.py          # CSV discovery, schema validation, caching
├── map_builder.py          # Pure Plotly figure construction
├── constants.py            # File patterns, color scale, FIPS widths, caveats
├── components/             # Header, controls, info-strip
├── data/
│   ├── outcomes.csv        # outcome_id -> human-readable label
│   ├── models.csv          # model_id -> human-readable label
│   ├── state/              # <model_id>_<outcome_id>_state.csv
│   └── county/             # <model_id>_<outcome_id>_county.csv (empty in v1)
├── assets/style.css        # Auto-loaded by Dash
└── tests/test_data_loader.py
```

## File-naming convention

The data layer is **discovery-driven** — the app scans `data/state/` and parses
filenames to figure out which `(model, outcome)` pairs are available:

```
<model_id>_<outcome_id>_state.csv
<model_id>_<outcome_id>_county.csv
```

Both `model_id` and `outcome_id` are lowercase snake_case, and either may
contain underscores (e.g. `glm_mrp` + `policy_renewable_bin`). Because of
that ambiguity, the loader splits filenames against the **known model_id
catalog** in `data/models.csv` — a model_id must appear there before its
files will be picked up. Examples:

- `baseline_happening_bin_state.csv`
- `glm_mrp_policy_renewable_bin_state.csv`
- `mrdeep_worried_bin_county.csv`

Each CSV must contain these columns (column order doesn't matter):

| column          | type                       | notes                          |
|-----------------|----------------------------|--------------------------------|
| `state_fips`    | int or zero-padded string  | loader normalizes to `"01"` etc. |
| `state_name`    | string                     |                                |
| `estimate`      | float in `[0, 1]`          |                                |
| `n_respondents` | int                        |                                |

(County files use `county_fips` / `county_name` and a 5-digit FIPS.)

## How to add a new model

1. Add a row to `data/models.csv` with the new `model_id`, label, and
   description. (This step is required — the loader uses the catalog to
   parse filenames.)
2. Drop a CSV into `data/state/` following the naming convention above.
3. (Optional) If the model warrants a caveat in the info strip, add an entry
   to `MODEL_CAVEATS` in `constants.py`.
4. Restart the app — the new model will appear in the dropdown automatically
   for any outcome whose CSV you provided.

## How to add a new outcome

1. Add a row to `data/outcomes.csv` with the new `outcome_id`, label, and
   description.
2. Drop one or more CSVs into `data/state/` named
   `<model_id>_<new_outcome_id>_state.csv`.
3. Restart the app.

## County support

Not wired into the UI yet. The data loader already scans `data/county/` and
the same naming convention applies, so once county CSVs land, v2 will need
only a UI toggle and the same `build_state_choropleth`-equivalent for
counties (which will use the standard counties GeoJSON at
`https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json`).

## Tests

```bash
cd map_app
python -m pytest tests/ -v
```

Covers file discovery, schema validation (FIPS padding, range checks,
column-order tolerance), missing-file behavior, and a sanity check that the
three shipped CSVs load.

## v1 scope (explicit non-goals)

- No state ↔ county UI toggle (ships with the data).
- No deployment / CI configs.
- No auth.
- No Mapbox token (we use Plotly's `albers usa` projection).
- No posterior intervals (will arrive with Stan-MRP / MRdeeP).
- No side-by-side model comparison (v2).
- No export buttons.
