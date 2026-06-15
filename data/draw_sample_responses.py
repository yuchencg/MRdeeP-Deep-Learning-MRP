"""
Draw nationally representative samples of 1,000 / 3,000 / 5,000 respondents
from filtered_responses_preprocessing.dta, stratified by state population.

Inputs:
  - data_processed/filtered_responses_preprocessing.dta

Outputs:
  - data_processed/sample_ces_1000.csv
  - data_processed/sample_ces_3000.csv
  - data_processed/sample_ces_5000.csv

Method:
  - Sample sizes are allocated proportionally via Hamilton's largest-remainder
    method, with a minimum of 1 respondent per state.
  - Within each state, respondents are drawn without replacement using
    commonweight as sampling probabilities (improves within-state demographic
    representativeness).

Run: python draw_sample_responses.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent
CSV_IN  = ROOT / "data_processed" / "filtered_responses_preprocessed.csv"
OUT_DIR = ROOT / "data_processed"

N_SAMPLES   = [1_000, 3_000, 5_000]
RANDOM_SEED = 42
WEIGHT_COL  = "commonweight"



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hamilton_allocation(pops: pd.Series, n_total: int) -> pd.Series:
    """Proportional allocation via Hamilton's largest-remainder method.

    Every state receives at least 1 seat; total is guaranteed to equal n_total.
    """
    exact = pops / pops.sum() * n_total
    alloc = exact.apply(np.floor).clip(lower=1).astype(int)
    remainder = n_total - alloc.sum()
    fractions = (exact - alloc).sort_values(ascending=False)
    for idx in fractions.index[:remainder]:
        alloc[idx] += 1
    assert alloc.sum() == n_total
    return alloc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def draw_sample(df: pd.DataFrame, state_meta: pd.DataFrame, n_sample: int) -> pd.DataFrame:
    alloc = hamilton_allocation(state_meta["state_pop"], n_sample)
    rng = np.random.default_rng(RANDOM_SEED)
    parts: list[pd.DataFrame] = []

    for fips, n_alloc in alloc.items():
        pool = df[df["state_fips"] == fips]
        if len(pool) == 0:
            print(f"  WARNING: no respondents for FIPS {fips}, skipping")
            continue
        if n_alloc > len(pool):
            print(f"  WARNING: {state_meta.loc[fips, 'state_name']} needs {n_alloc} "
                  f"but only {len(pool)} available; sampling all {len(pool)}")
            parts.append(pool)
            continue

        w = pool[WEIGHT_COL].fillna(0).clip(lower=0)
        if w.sum() == 0:
            w = pd.Series(1.0, index=pool.index)
        probs = (w / w.sum()).values

        chosen = rng.choice(pool.index, size=n_alloc, replace=False, p=probs)
        parts.append(pool.loc[chosen])

    return pd.concat(parts, ignore_index=True)


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

    print(f"Reading {CSV_IN.name} ...")
    df = pd.read_csv(str(CSV_IN), dtype={"state_fips": str, "county_fips": str})
    print(f"  {len(df):,} rows, {df.shape[1]} columns, {df['state_fips'].nunique()} states")

    # --- Build per-state population table ---
    state_meta = (
        df[["state_fips", "state_name", "state_pop"]]
        .drop_duplicates("state_fips")
        .set_index("state_fips")
        .copy()
    )
    state_meta["state_pop"] = state_meta["state_pop"].astype(float)

    missing_pop = state_meta[state_meta["state_pop"].isna()]
    if len(missing_pop):
        raise RuntimeError(
            f"state_pop missing for: {missing_pop['state_name'].tolist()}. "
            "Re-run build_filtered_responses.py to regenerate the processed file."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for n in N_SAMPLES:
        print(f"\n--- Drawing sample of {n:,} ---")
        sample = draw_sample(df, state_meta, n)
        print(f"  Final sample: {len(sample):,} rows across {sample['state_fips'].nunique()} states")
        csv_out = OUT_DIR / f"sample_ces_{n}.csv"
        sample.to_csv(str(csv_out), index=False)
        print(f"  Saved → {csv_out}")


if __name__ == "__main__":
    main()
