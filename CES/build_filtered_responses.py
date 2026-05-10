"""
Build filtered_responses_preprocessing.dta from the CES 2022 common output.

Inputs:
  - data_raw/CCES22_Common_OUTPUT_vv_topost.dta
  - data_raw/questions_to_include.csv
  - data_raw/state_county_city_FIPS_reference_table_20260509.csv
  - data_raw/co-est2025-pop.xlsx  (Census CO-EST2025-POP county estimates)

Output:
  - data_processed/filtered_responses_preprocessing.dta
    Outcome vars are renamed (see RENAME_MAP) and support/oppose items
    recoded to binary 1=Support / 0=Oppose (see SUPPORT_BINARY_VARS).

Defaults (edit the CONFIG block to change):
  - Keeps both pre-wave weights: commonweight, vvweight
  - Filters rows to pre-wave respondents (commonweight non-null)
  - Drops any *_post columns; only includes the pre-wave variables listed
    in questions_to_include.csv 
  - county_fips is the 5-digit state+county FIPS string (e.g. "26163");
    state_fips is the 2-digit state FIPS string (e.g. "26")
  - Adds state_name (from FIPS reference) and county_name (from Census file
    when matched, with a fallback to the dataset's own countyname field)
  - Adds state_pop and county_pop from co-est2025-pop.xlsx for POP_YEAR
    (default 2022 to align with the survey). State pop is summed across
    counties + DC. Connecticut counties don't match (file uses 2022 CT
    Planning Regions, CES uses pre-2022 CT counties) -> county_pop is NaN
    for CT respondents; CT state_pop is still correct.

Run: python build_filtered_responses.py
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DTA_IN = ROOT / "data_raw" / "CCES22_Common_OUTPUT_vv_topost.dta"
QUESTIONS_CSV = ROOT / "data_raw" / "questions_to_include.csv"
FIPS_CSV = ROOT / "data_raw" / "state_county_city_FIPS_reference_table_20260509.csv"
COEST_XLSX = ROOT / "data_raw" / "co-est2025-pop.xlsx"
DTA_OUT = ROOT / "data_processed" / "filtered_responses_preprocessing.dta"

PRE_WEIGHTS = ["commonweight", "vvweight"]  # pre-wave weights to retain
ROW_FILTER_WEIGHT = "commonweight"          # require non-null on this weight
POP_YEAR = 2022                             # column from co-est2025-pop.xlsx

# Support/oppose vars recoded to binary: 1=Support, 0=Oppose, others→NA.
# Original CES coding: 1=Support, 2=Oppose, 8=skipped, 9=not asked.
SUPPORT_BINARY_VARS = [
    "CC22_333a", "CC22_333b", "CC22_333c", "CC22_333d", "CC22_333e",
    "CC22_355a",
]

# Rename outcome variables to descriptive names for MRP models.
# Keys are original CES variable names; values are new column names.
RENAME_MAP = {
    "CC22_333":  "climate_problem",   # 5-cat: how serious is climate change?
    "CC22_333a": "regulate_carbon",   # Support: EPA regulate CO2 emissions
    "CC22_333b": "renewable_fuel",    # Support: require renewable fuels per state
    "CC22_333c": "clean_air_water",   # Support: strengthen EPA Clean Air/Water Act
    "CC22_333d": "fuel_efficiency",   # Support: raise fuel efficiency (40→54.5 mpg)
    "CC22_333e": "fossil_fuel",       # Support: increase fossil fuel production
    "CC22_355a": "paris_agreement",   # Support: US rejoins Paris Climate Agreement
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_question_varnames(path: Path) -> list[str]:
    """Read questions_to_include.csv and return unique Var_name values.
    """
    df = pd.read_csv(path, dtype=str, on_bad_lines="skip")
    if "Var_name" not in df.columns:
        raise ValueError(f"Expected a 'Var_name' column in {path}")
    names = (
        df["Var_name"]
        .dropna()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    # Drop anything that looks like a header row leak.
    names = [n for n in names if n.lower() != "var_name"]
    return names


def load_state_fips_map(path: Path) -> dict[str, str]:
    """Map 2-digit state FIPS string -> proper-cased state name."""
    ref = pd.read_csv(path, dtype=str)
    ref = ref[["State FIPS Code", "State Name"]].drop_duplicates()
    return {
        row["State FIPS Code"].zfill(2): row["State Name"].title()
        for _, row in ref.iterrows()
        if pd.notna(row["State FIPS Code"]) and pd.notna(row["State Name"])
    }


def load_county_fips_map(path: Path) -> dict[str, str]:
    """Map 5-digit StCnty FIPS string -> proper-cased county name."""
    ref = pd.read_csv(path, dtype=str)
    ref = ref[["StCnty FIPS Code", "County Name"]].drop_duplicates(
        subset=["StCnty FIPS Code"]
    )
    return {
        row["StCnty FIPS Code"].zfill(5): row["County Name"].title()
        for _, row in ref.iterrows()
        if pd.notna(row["StCnty FIPS Code"]) and pd.notna(row["County Name"])
    }


# County-equivalent suffixes from the Census file we strip during matching.
# Order matters: longer phrases first.
_COUNTY_SUFFIXES = [
    "City and Borough",
    "Census Area",
    "Municipality",
    "Borough",
    "Parish",
    "County",
]


def _smash(s: str) -> str:
    """Aggressive normalization: ASCII-fold, drop everything non-alnum, upper.

    This collapses 'DeKalb' / 'De Kalb', 'Matanuska-Susitna' / 'Matanuska
    Susitna', 'Doña Ana' / 'DONA ANA', 'O'Brien' / 'OBRIEN', etc. into the
    same key, which is what we want for matching across the Census file
    and the FIPS reference table.
    """
    s = unicodedata.normalize("NFKD", str(s)).encode("ASCII", "ignore").decode()
    return re.sub(r"[^A-Z0-9]", "", s.upper())


# Trailing tokens (with whitespace before them) that the FIPS reference
# table sometimes carries on county names. Stripped before smashing so
# names like HILLSBOROUGH or ITASCA aren't mangled by the smasher.
_REF_SUFFIX_PATTERNS = [
    r"\s+CITY AND BOROUGH$",
    r"\s+CENSUS AREA$",
    r"\s+MUNICIPALITY$",
    r"\s+BOROUGH$",
    r"\s+PARISH$",
    r"\s+COUNTY$",
    r"\s+CA$",  # Census Area abbreviation
]


def _normalize_ref_county(name: str, is_city_fips: bool) -> str:
    """Build the smashed match key for a row from the FIPS reference table.

    Strips CITY markers only when the row is an independent-city FIPS, then
    strips county-equivalent suffixes via word-boundary regexes (so e.g.
    "HILLSBOROUGH" doesn't lose its "BOROUGH"), then smashes.
    """
    s = str(name)
    if is_city_fips:
        s = re.sub(r"\s*\(\s*CITY\s*\)\s*$", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+CITY\s*$", "", s, flags=re.IGNORECASE)
    for pat in _REF_SUFFIX_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    return _smash(s)


def _strip_suffix(county: str) -> tuple[str, bool]:
    """Strip a Census county-equivalent suffix from a county name.

    Returns (stripped_name, is_independent_city). The `city` suffix is the
    only one that signals an independent city (FIPS code >= 500); we treat
    other suffixes as plain counties.
    """
    c = county.strip()
    # Independent-city marker is lowercase 'city' as a trailing word.
    if re.search(r"\bcity$", c):
        return c[: -len("city")].strip(), True
    for suf in _COUNTY_SUFFIXES:
        if c.lower().endswith(" " + suf.lower()):
            return c[: -(len(suf) + 1)].strip(), False
    return c, False


def load_population_table(
    xlsx: Path, fips_csv: Path, year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (county_pop_df, state_pop_df) for the requested year.

    county_pop_df has columns: stcnty_fips (5-digit), county_pop.
    state_pop_df has columns:  state_fips (2-digit), state_pop.

    State pop sums across every row in the Census file for that state,
    independent of whether each county row resolved to a FIPS code. This
    matters for CT, whose 2022 Planning Regions don't map to the old
    county FIPS used by the CES.
    """
    raw = pd.read_excel(xlsx, sheet_name=0, header=None)
    # Year columns sit on row index 3 (0-based) at positions 2..7
    # corresponding to 2020,2021,2022,2023,2024,2025.
    year_cols = {2020: 2, 2021: 3, 2022: 4, 2023: 5, 2024: 6, 2025: 7}
    if year not in year_cols:
        raise ValueError(f"POP_YEAR must be one of {sorted(year_cols)}; got {year}")
    pop_col = year_cols[year]

    # Data rows: skip the 5-row header and the trailing footnote rows.
    # Real rows are those whose first column starts with '.'.
    data = raw.iloc[:, [0, pop_col]].copy()
    data.columns = ["geo", "pop"]
    data["geo"] = data["geo"].astype(str)
    data = data[data["geo"].str.startswith(".")].copy()
    data["geo"] = data["geo"].str.lstrip(".").str.strip()
    data["pop"] = pd.to_numeric(data["pop"], errors="coerce")
    data = data.dropna(subset=["geo", "pop"])

    parts = data["geo"].str.rsplit(",", n=1, expand=True)
    data["county_raw"] = parts[0].str.strip()
    data["state_name"] = parts[1].str.strip()

    stripped = data["county_raw"].apply(_strip_suffix)
    data["county_key"] = [_smash(s) for s, _ in stripped]
    data["is_city"] = [is_city for _, is_city in stripped]
    data["state_key"] = data["state_name"].apply(_smash)

    # Build candidates from the FIPS reference table.
    ref = pd.read_csv(fips_csv, dtype=str)
    ref = ref[["State Name", "County Name", "StCnty FIPS Code"]].drop_duplicates(
        subset=["StCnty FIPS Code"]
    )
    ref = ref.dropna(subset=["State Name", "County Name", "StCnty FIPS Code"])
    ref["state_key"] = ref["State Name"].apply(_smash)
    ref["county_code"] = ref["StCnty FIPS Code"].str.zfill(5).str[2:]
    ref["is_city_fips"] = ref["county_code"].astype(int) >= 500

    # Build the ref-side match key. Independent cities (FIPS code >= 500)
    # get the trailing CITY marker stripped so "BALTIMORE CITY" -> "BALTIMORE";
    # counties whose actual name contains "City" (e.g. VA "Charles City
    # County", FIPS 51036) keep "City" because they're not flagged as
    # independent. Other suffix abbreviations get stripped before smashing.
    ref["county_key"] = ref.apply(
        lambda r: _normalize_ref_county(r["County Name"], r["is_city_fips"]),
        axis=1,
    )

    # Hardcoded override: DC's ref entry has county "WASHINGTON, DC" which
    # smashes to "WASHINGTONDC" and won't match the Census file's "District
    # of Columbia". Add the alternate key explicitly.
    dc_row = ref.loc[ref["StCnty FIPS Code"] == "11001"].copy()
    if len(dc_row):
        dc_row["county_key"] = _smash("District of Columbia")
        ref = pd.concat([ref, dc_row], ignore_index=True)

    # Join candidates per (state_key, county_key).
    merged = data.merge(
        ref[["state_key", "county_key", "StCnty FIPS Code", "is_city_fips"]],
        on=["state_key", "county_key"],
        how="left",
    )

    # When multiple FIPS candidates exist for a Census row, prefer the one
    # whose city/non-city status matches the Census file's suffix. We do this
    # by sorting so the preferred row sorts first, then dropping duplicates.
    merged["match_priority"] = (
        merged["is_city"].fillna(False) != merged["is_city_fips"].fillna(False)
    ).astype(int)  # 0 = matching status (preferred), 1 = mismatched
    merged = merged.sort_values(
        ["state_key", "county_key", "is_city", "match_priority"]
    )
    picked = merged.drop_duplicates(
        subset=["state_key", "county_key", "is_city"], keep="first"
    ).reset_index(drop=True)

    # Diagnostics
    unmatched = picked[picked["StCnty FIPS Code"].isna()]
    if len(unmatched):
        print(
            f"  WARNING: {len(unmatched)} county rows from {xlsx.name} did not "
            f"resolve to a FIPS code (expected for CT planning regions):"
        )
        for _, r in unmatched.head(15).iterrows():
            print(f"    - {r['county_raw']}, {r['state_name']}")
        if len(unmatched) > 15:
            print(f"    ... and {len(unmatched) - 15} more")

    county_out = picked.dropna(subset=["StCnty FIPS Code"])[
        ["StCnty FIPS Code", "pop"]
    ].rename(columns={"StCnty FIPS Code": "stcnty_fips", "pop": "county_pop"})
    county_out["stcnty_fips"] = county_out["stcnty_fips"].str.zfill(5)

    # State pop: sum every row in the Census file for that state, but only
    # emit a value when every county-equivalent row in that state resolved
    # to a FIPS code. States with any unmatched county rows (e.g. CT, with
    # all 9 Planning Regions unmatched) get dropped here so respondents in
    # those states end up with state_pop = NaN.
    state_fips_by_key = (
        ref[["state_key", "StCnty FIPS Code"]]
        .assign(state_fips=lambda d: d["StCnty FIPS Code"].str[:2])
        .drop_duplicates(subset=["state_key"])
        .set_index("state_key")["state_fips"]
        .to_dict()
    )
    fully_matched_states = (
        picked.groupby("state_key")["StCnty FIPS Code"]
        .apply(lambda s: s.notna().all())
        .pipe(lambda s: s[s].index.tolist())
    )
    state_out = (
        data[data["state_key"].isin(fully_matched_states)]
        .groupby("state_key", as_index=False)["pop"]
        .sum()
        .rename(columns={"pop": "state_pop"})
    )
    state_out["state_fips"] = state_out["state_key"].map(state_fips_by_key)
    state_out = state_out.dropna(subset=["state_fips"])[["state_fips", "state_pop"]]

    return county_out.reset_index(drop=True), state_out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

    print(f"Reading question list from {QUESTIONS_CSV.name}")
    question_vars = load_question_varnames(QUESTIONS_CSV)
    print(f"  {len(question_vars)} unique Var_name entries")

    # Discover which of those columns actually exist in the .dta, and grab
    # the variable + value labels from the source so we can propagate them
    # into the output.
    print(f"Inspecting columns and labels in {DTA_IN.name}")
    with pd.io.stata.StataReader(str(DTA_IN)) as reader:
        src_variable_labels = reader.variable_labels()
        src_value_labels = reader.value_labels()
    header = next(
        iter(pd.read_stata(str(DTA_IN), chunksize=1, convert_categoricals=False))
    )
    available = set(header.columns)

    present = [v for v in question_vars if v in available]
    missing = [v for v in question_vars if v not in available]
    if missing:
        print(f"  WARNING: {len(missing)} requested vars not found and skipped:")
        for m in missing:
            print(f"    - {m}")
    print(f"  {len(present)} question vars will be included")

    # Drop any post-wave variants that may have crept in (defensive — none of
    # the listed question vars end in _post, but enforce the rule).
    present = [v for v in present if not v.lower().endswith("_post")]

    # Required identifier / FIPS / weight columns.
    required = ["caseid", "inputstate", "countyfips", "countyname"]
    required = [c for c in required if c in available]
    weights = [w for w in PRE_WEIGHTS if w in available]
    if not weights:
        raise RuntimeError("None of the configured pre-wave weights exist in the .dta")

    columns = required + weights + present
    # de-dupe while preserving order
    seen = set()
    columns = [c for c in columns if not (c in seen or seen.add(c))]

    print(f"Loading {len(columns)} columns from {DTA_IN.name} ...")
    df = pd.read_stata(str(DTA_IN), columns=columns, convert_categoricals=False)
    print(f"  loaded {len(df):,} rows")

    # --- Row filter: keep pre-wave respondents only ---
    if ROW_FILTER_WEIGHT in df.columns:
        before = len(df)
        df = df[df[ROW_FILTER_WEIGHT].notna()].copy()
        print(
            f"Filtered to pre-wave respondents on {ROW_FILTER_WEIGHT}: "
            f"{len(df):,} of {before:,} kept"
        )

    # --- Build FIPS / name columns ---
    state_fips_map = load_state_fips_map(FIPS_CSV)
    county_name_map = load_county_fips_map(FIPS_CSV)

    # state_fips: 2-digit zero-padded string from numeric inputstate
    df["state_fips"] = (
        df["inputstate"].astype("Int64").astype(str).str.zfill(2).where(
            df["inputstate"].notna()
        )
    )

    # county_fips: 5-digit state+county FIPS string (e.g. "26163").
    df["countyfips"] = df["countyfips"].astype(str).str.strip()
    df.loc[df["countyfips"].isin(["", "nan", "None"]), "countyfips"] = pd.NA
    df["county_fips"] = df["countyfips"].str.zfill(5)

    df["state_name"] = df["state_fips"].map(state_fips_map)
    df["county_name"] = df["county_fips"].map(county_name_map)

    # Fall back to the dataset's countyname field (e.g., "Wayne MI") with
    # the trailing state abbrev stripped, when the reference table has no hit.
    fallback = df["countyname"].astype(str).str.replace(
        r"\s+[A-Z]{2}$", "", regex=True
    ).str.strip().str.title()
    df["county_name"] = df["county_name"].fillna(fallback)

    # --- Population merge (county + state) ---
    print(f"Loading population (year {POP_YEAR}) from {COEST_XLSX.name}")
    county_pop_df, state_pop_df = load_population_table(
        COEST_XLSX, FIPS_CSV, POP_YEAR
    )
    print(
        f"  matched {len(county_pop_df):,} counties to FIPS; "
        f"computed state pop for {len(state_pop_df):,} states"
    )
    df = df.merge(
        county_pop_df, how="left", left_on="county_fips", right_on="stcnty_fips"
    ).drop(columns=["stcnty_fips"])
    df = df.merge(state_pop_df, on="state_fips", how="left")

    df["county_pop"] = df["county_pop"].astype("Int64")
    df["state_pop"] = df["state_pop"].astype("Int64")

    n_county_missing = df["county_pop"].isna().sum()
    n_state_missing = df["state_pop"].isna().sum()
    print(
        f"  county_pop missing on {n_county_missing:,} rows "
        f"(includes CT respondents); state_pop missing on {n_state_missing:,} rows"
    )

    # --- Final column order ---
    final_cols = (
        [
            "caseid",
            "state_fips",
            "state_name",
            "state_pop",
            "county_fips",
            "county_name",
            "county_pop",
        ]
        + weights
        + present
    )
    final_cols = [c for c in final_cols if c in df.columns]
    out = df[final_cols].copy()

    # Stata-friendly: ensure no object-dtype columns hold mixed types.
    for c in out.select_dtypes(include="object").columns:
        out[c] = out[c].astype("string").astype(object).where(out[c].notna(), "")

    # --- Build label dicts to write into the output .dta ---
    # value labels: only for question vars whose source had value labels and
    # whose codes are integer-compatible.
    value_labels: dict[str, dict[int, str]] = {}
    for col in present:
        if col not in out.columns or col not in src_value_labels:
            continue
        labels = src_value_labels[col]
        # Stata value labels must be keyed by int. Source codes from pandas
        # may come back as floats — cast safely.
        clean = {}
        for k, v in labels.items():
            try:
                ik = int(k)
            except (TypeError, ValueError):
                continue
            clean[ik] = str(v)
        if not clean:
            continue
        # to_stata writes value labels for integer columns. Convert the
        # column to nullable Int32 so NaNs survive as missing.
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            out[col] = s.astype("Int32")
            value_labels[col] = clean

    # variable labels: the source's question text for each kept column,
    # plus our own labels for the FIPS / name columns we constructed.
    variable_labels: dict[str, str] = {}
    for col in out.columns:
        if col in src_variable_labels and src_variable_labels[col]:
            variable_labels[col] = src_variable_labels[col][:80]  # Stata cap
    variable_labels.update(
        {
            "state_fips": "State FIPS code (2-digit)",
            "state_name": "State name",
            "state_pop": f"State population, {POP_YEAR} (CO-EST2025-POP)",
            "county_fips": "County FIPS (5-digit state+county)",
            "county_name": "County name",
            "county_pop": f"County population, {POP_YEAR} (CO-EST2025-POP)",
            "commonweight": variable_labels.get(
                "commonweight", "Pre-wave common weight"
            ),
            "vvweight": variable_labels.get(
                "vvweight", "Pre-wave validated-voter weight"
            ),
        }
    )

    # --- Recode support/oppose outcomes to binary (1=Support, 0=Oppose) ---
    # CES original: 1=Support, 2=Oppose, 8=skipped, 9=not asked → all non-1/2 → NA.
    support_recoded = []
    for var in SUPPORT_BINARY_VARS:
        if var not in out.columns:
            continue
        out[var] = out[var].map({1: 1, 2: 0}).astype("Int8")
        value_labels[var] = {1: "Support", 0: "Oppose"}
        support_recoded.append(var)
    if support_recoded:
        print(
            f"Recoded {len(support_recoded)} vars to binary 1=Support/0=Oppose: "
            + ", ".join(support_recoded)
        )

    # --- Rename outcome variables to descriptive names ---
    active_renames = {k: v for k, v in RENAME_MAP.items() if k in out.columns}
    if active_renames:
        out = out.rename(columns=active_renames)
        value_labels  = {active_renames.get(k, k): v for k, v in value_labels.items()}
        variable_labels = {active_renames.get(k, k): v for k, v in variable_labels.items()}
        print("Renamed columns: " + ", ".join(f"{k}→{v}" for k, v in active_renames.items()))

    # --- Recode climate_problem to binary ---
    # 1 ("serious problem / immediate action") + 2 ("enough evidence / some action") → 1
    # 3 ("more research needed") + 4 ("concern exaggerated") + 5 ("not occurring") → 0
    # 8 (skipped), 9 (not asked), NA → NA
    if "climate_problem" in out.columns:
        out["climate_problem"] = (
            out["climate_problem"].map({1: 1, 2: 1, 3: 0, 4: 0, 5: 0}).astype("Int8")
        )
        value_labels["climate_problem"] = {1: "Climate action needed", 0: "No action needed"}
        print("Recoded climate_problem: cat 1-2 → 1 (action needed), cat 3-5 → 0 (no action)")

    DTA_OUT.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Writing {len(out):,} rows x {out.shape[1]} cols to {DTA_OUT} "
        f"(value labels on {len(value_labels)} cols, "
        f"variable labels on {len(variable_labels)} cols)"
    )
    out.to_stata(
        str(DTA_OUT),
        write_index=False,
        version=118,
        value_labels=value_labels,
        variable_labels=variable_labels,
    )
    print("Done.")


if __name__ == "__main__":
    main()
