"""
Draw a nationally representative sample of 1,200 respondents from
filtered_responses_preprocessing.dta, stratified by state population.

Inputs:
  - data_processed/filtered_responses_preprocessing.dta

Output:
  - data_processed/sample_ces.dta

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
DTA_IN  = ROOT / "data_processed" / "filtered_responses_preprocessing.dta"
CSV_OUT = ROOT / "data_processed" / "sample_ces.csv"

N_SAMPLE    = 1_200
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
def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

    print(f"Reading {DTA_IN.name} ...")
    df = pd.read_stata(str(DTA_IN), convert_categoricals=False)
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

    # --- Proportional allocation ---
    alloc = hamilton_allocation(state_meta["state_pop"], N_SAMPLE)

    print(f"\nAllocation summary (top 10 by population):")
    display = (
        state_meta["state_name"]
        .to_frame()
        .join(state_meta["state_pop"])
        .join(alloc.rename("n_alloc"))
        .sort_values("state_pop", ascending=False)
    )
    for _, row in display.head(10).iterrows():
        print(f"  {row['state_name']:<25} pop={int(row['state_pop']):>12,}  n={row['n_alloc']:>3}")
    print(f"  ... ({len(alloc)} states total, sum={alloc.sum()})")

    # --- Stratified sampling ---
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

    sample = pd.concat(parts, ignore_index=True)
    print(f"\nFinal sample: {len(sample):,} rows across "
          f"{sample['state_fips'].nunique()} states")

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(str(CSV_OUT), index=False)
    print(f"Saved → {CSV_OUT}")


if __name__ == "__main__":
    main()
