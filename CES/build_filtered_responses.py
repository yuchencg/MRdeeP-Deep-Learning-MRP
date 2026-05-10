"""
Build filtered_responses_preprocessing.dta from the CES 2022 common output.

Inputs:
  - data_raw/CCES22_Common_OUTPUT_vv_topost.dta
  - data_raw/questions_to_include.csv
  - data_raw/fips2county.tsv         (FIPS ↔ county/state names)
  - data_raw/co-est2025-pop.xlsx     (Census CO-EST2025-POP county estimates)

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
  - Adds state_name and county_name from fips2county.tsv
  - Adds state_pop and county_pop from co-est2025-pop.xlsx for POP_YEAR.
    State pop is summed by state name directly from the Census file — it
    does not depend on county matching succeeding. County pop is matched
    by stripping the " County" suffix from the Excel name; CT planning
    regions and Doña Ana NM have county_pop = NaN.

Run: python build_filtered_responses.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent
DTA_IN        = ROOT / "data_raw" / "CCES22_Common_OUTPUT_vv_topost.dta"
QUESTIONS_CSV = ROOT / "data_raw" / "questions_to_include.csv"
FIPS_TSV      = ROOT / "data_raw" / "fips2county.tsv"
COEST_XLSX    = ROOT / "data_raw" / "co-est2025-pop.xlsx"
DTA_OUT       = ROOT / "data_processed" / "filtered_responses_preprocessing.dta"

PRE_WEIGHTS      = ["commonweight", "vvweight"]
ROW_FILTER_WEIGHT = "commonweight"
POP_YEAR         = 2022

# Support/oppose vars recoded to binary: 1=Support, 0=Oppose, others→NA.
# Original CES coding: 1=Support, 2=Oppose, 8=skipped, 9=not asked.
SUPPORT_BINARY_VARS = [
    "CC22_333a", "CC22_333b", "CC22_333c", "CC22_333d", "CC22_333e",
    "CC22_355a",
]

# Rename outcome variables to descriptive names for MRP models.
RENAME_MAP = {
    "CC22_333":  "climate_problem",
    "CC22_333a": "regulate_carbon",
    "CC22_333b": "renewable_fuel",
    "CC22_333c": "clean_air_water",
    "CC22_333d": "fuel_efficiency",
    "CC22_333e": "fossil_fuel",
    "CC22_355a": "paris_agreement",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_question_varnames(path: Path) -> list[str]:
    """Read questions_to_include.csv and return unique Var_name values."""
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
    return [n for n in names if n.lower() != "var_name"]


def load_state_fips_map(path: Path) -> dict[str, str]:
    """Map 2-digit state FIPS string → state name."""
    tsv = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig")
    return (
        tsv[["StateFIPS", "StateName"]]
        .drop_duplicates("StateFIPS")
        .set_index("StateFIPS")["StateName"]
        .to_dict()
    )


def load_county_fips_map(path: Path) -> dict[str, str]:
    """Map 5-digit county FIPS string → county name."""
    tsv = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig")
    return (
        tsv[["CountyFIPS", "CountyName"]]
        .drop_duplicates("CountyFIPS")
        .set_index("CountyFIPS")["CountyName"]
        .to_dict()
    )


def load_population_table(
    xlsx: Path, fips_tsv: Path, year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (county_pop_df, state_pop_df) for the requested year.

    county_pop_df has columns: stcnty_fips (5-digit), county_pop.
    state_pop_df  has columns: state_fips  (2-digit), state_pop.

    State pop is summed by state name directly from the Census file and does
    not depend on county-level matching. County pop is resolved by stripping
    the trailing ' County' suffix from the Excel name and joining to
    fips2county.tsv on (county_name, state_name). CT planning regions and
    Doña Ana NM will be unmatched (county_pop = NaN for those respondents).
    """
    raw = pd.read_excel(xlsx, sheet_name=0, header=None)
    year_cols = {2020: 2, 2021: 3, 2022: 4, 2023: 5, 2024: 6, 2025: 7}
    if year not in year_cols:
        raise ValueError(f"POP_YEAR must be one of {sorted(year_cols)}; got {year}")
    pop_col = year_cols[year]

    data = raw.iloc[:, [0, pop_col]].copy()
    data.columns = ["geo", "pop"]
    data["geo"] = data["geo"].astype(str)
    data = data[data["geo"].str.startswith(".")].copy()
    data["geo"] = data["geo"].str.lstrip(".").str.strip()
    data["pop"] = pd.to_numeric(data["pop"], errors="coerce")
    data = data.dropna(subset=["geo", "pop"])

    parts = data["geo"].str.rsplit(",", n=1, expand=True)
    data["county_raw"] = parts[0].str.strip()
    data["state_raw"]  = parts[1].str.strip()

    tsv = pd.read_csv(fips_tsv, sep="\t", dtype=str, encoding="utf-8-sig")

    # --- State population: sum by state name, independent of county matching ---
    state_name_to_fips = (
        tsv[["StateFIPS", "StateName"]]
        .drop_duplicates("StateName")
        .set_index("StateName")["StateFIPS"]
    )
    state_out = (
        data.groupby("state_raw", as_index=False)["pop"]
        .sum()
        .rename(columns={"state_raw": "state_name_raw", "pop": "state_pop"})
    )
    state_out["state_fips"] = state_out["state_name_raw"].map(state_name_to_fips)
    state_out = state_out.dropna(subset=["state_fips"])[["state_fips", "state_pop"]]

    # --- County population: strip ' County' suffix, join on (county, state) ---
    data["county_key"] = (
        data["county_raw"].str.replace(r" County$", "", regex=True).str.lower()
    )
    data["state_key"] = data["state_raw"].str.lower()

    ref = tsv[["CountyFIPS", "CountyName", "StateName"]].copy()
    ref["county_key"] = ref["CountyName"].str.lower()
    ref["state_key"]  = ref["StateName"].str.lower()
    ref = ref.drop_duplicates(["county_key", "state_key"])

    merged = data.merge(
        ref[["county_key", "state_key", "CountyFIPS"]],
        on=["county_key", "state_key"],
        how="left",
    )

    unmatched = merged[
        merged["CountyFIPS"].isna()
        & ~merged["county_raw"].str.contains("Planning Region", case=False)
    ]
    if len(unmatched):
        print(
            f"  WARNING: {len(unmatched)} county rows from {xlsx.name} did not "
            f"resolve to a FIPS code:"
        )
        for _, r in unmatched.head(10).iterrows():
            print(f"    - {r['county_raw']}, {r['state_raw']}")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched) - 10} more")

    county_out = (
        merged.dropna(subset=["CountyFIPS"])[["CountyFIPS", "pop"]]
        .rename(columns={"CountyFIPS": "stcnty_fips", "pop": "county_pop"})
    )

    return county_out.reset_index(drop=True), state_out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

    print(f"Reading question list from {QUESTIONS_CSV.name}")
    question_vars = load_question_varnames(QUESTIONS_CSV)
    print(f"  {len(question_vars)} unique Var_name entries")

    print(f"Inspecting columns and labels in {DTA_IN.name}")
    with pd.io.stata.StataReader(str(DTA_IN)) as reader:
        src_variable_labels = reader.variable_labels()
        src_value_labels    = reader.value_labels()
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

    present = [v for v in present if not v.lower().endswith("_post")]

    required = ["caseid", "inputstate", "countyfips", "countyname"]
    required = [c for c in required if c in available]
    weights  = [w for w in PRE_WEIGHTS if w in available]
    if not weights:
        raise RuntimeError("None of the configured pre-wave weights exist in the .dta")

    columns = required + weights + present
    seen    = set()
    columns = [c for c in columns if not (c in seen or seen.add(c))]

    print(f"Loading {len(columns)} columns from {DTA_IN.name} ...")
    df = pd.read_stata(str(DTA_IN), columns=columns, convert_categoricals=False)
    print(f"  loaded {len(df):,} rows")

    if ROW_FILTER_WEIGHT in df.columns:
        before = len(df)
        df = df[df[ROW_FILTER_WEIGHT].notna()].copy()
        print(
            f"Filtered to pre-wave respondents on {ROW_FILTER_WEIGHT}: "
            f"{len(df):,} of {before:,} kept"
        )

    # --- Build FIPS / name columns ---
    state_fips_map  = load_state_fips_map(FIPS_TSV)
    county_name_map = load_county_fips_map(FIPS_TSV)

    df["state_fips"] = (
        df["inputstate"].astype("Int64").astype(str).str.zfill(2).where(
            df["inputstate"].notna()
        )
    )

    df["countyfips"] = df["countyfips"].astype(str).str.strip()
    df.loc[df["countyfips"].isin(["", "nan", "None"]), "countyfips"] = pd.NA
    df["county_fips"] = df["countyfips"].str.zfill(5)

    df["state_name"]  = df["state_fips"].map(state_fips_map)
    df["county_name"] = df["county_fips"].map(county_name_map)

    fallback = df["countyname"].astype(str).str.replace(
        r"\s+[A-Z]{2}$", "", regex=True
    ).str.strip().str.title()
    df["county_name"] = df["county_name"].fillna(fallback)

    # --- Population merge ---
    print(f"Loading population (year {POP_YEAR}) from {COEST_XLSX.name}")
    county_pop_df, state_pop_df = load_population_table(COEST_XLSX, FIPS_TSV, POP_YEAR)
    print(
        f"  matched {len(county_pop_df):,} counties to FIPS; "
        f"computed state pop for {len(state_pop_df):,} states"
    )
    df = df.merge(
        county_pop_df, how="left", left_on="county_fips", right_on="stcnty_fips"
    ).drop(columns=["stcnty_fips"])
    df = df.merge(state_pop_df, on="state_fips", how="left")

    df["county_pop"] = df["county_pop"].astype("Int64")
    df["state_pop"]  = df["state_pop"].astype("Int64")

    n_county_missing = df["county_pop"].isna().sum()
    n_state_missing  = df["state_pop"].isna().sum()
    print(
        f"  county_pop missing on {n_county_missing:,} rows; "
        f"state_pop missing on {n_state_missing:,} rows"
    )

    # --- Final column order ---
    final_cols = (
        ["caseid", "state_fips", "state_name", "state_pop",
         "county_fips", "county_name", "county_pop"]
        + weights
        + present
    )
    final_cols = [c for c in final_cols if c in df.columns]
    out = df[final_cols].copy()

    for c in out.select_dtypes(include="object").columns:
        out[c] = out[c].astype("string").astype(object).where(out[c].notna(), "")

    # --- Build label dicts ---
    value_labels: dict[str, dict[int, str]] = {}
    for col in present:
        if col not in out.columns or col not in src_value_labels:
            continue
        labels = src_value_labels[col]
        clean = {}
        for k, v in labels.items():
            try:
                ik = int(k)
            except (TypeError, ValueError):
                continue
            clean[ik] = str(v)
        if not clean:
            continue
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            out[col] = s.astype("Int32")
            value_labels[col] = clean

    variable_labels: dict[str, str] = {}
    for col in out.columns:
        if col in src_variable_labels and src_variable_labels[col]:
            variable_labels[col] = src_variable_labels[col][:80]
    variable_labels.update(
        {
            "state_fips":  "State FIPS code (2-digit)",
            "state_name":  "State name",
            "state_pop":   f"State population, {POP_YEAR} (CO-EST2025-POP)",
            "county_fips": "County FIPS (5-digit state+county)",
            "county_name": "County name",
            "county_pop":  f"County population, {POP_YEAR} (CO-EST2025-POP)",
            "commonweight": variable_labels.get("commonweight", "Pre-wave common weight"),
            "vvweight":     variable_labels.get("vvweight",     "Pre-wave validated-voter weight"),
        }
    )

    # --- Recode support/oppose outcomes to binary ---
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

    # --- Rename outcome variables ---
    active_renames = {k: v for k, v in RENAME_MAP.items() if k in out.columns}
    if active_renames:
        out = out.rename(columns=active_renames)
        value_labels    = {active_renames.get(k, k): v for k, v in value_labels.items()}
        variable_labels = {active_renames.get(k, k): v for k, v in variable_labels.items()}
        print("Renamed columns: " + ", ".join(f"{k}→{v}" for k, v in active_renames.items()))

    # --- Recode climate_problem to binary ---
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
