import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUTCOMES = ["climate_problem", "renewable_fuel"]
SAMPLES = {"sample1_state": 1000, "sample2_state": 3000, "sample3_state": 5000}
MODELS = ["baseline", "glm_mrp", "glmer_mrp", "glmerstan", "srp", "mrdeep"]
DISPLAY_NAMES = {
    "baseline": "Naive", "glm_mrp": "GLM-MRP", "glmer_mrp": "GLMER-MRP",
    "glmerstan": "GLMERStan", "srp": "SRP", "mrdeep": "MRdeeP",
}

def load_ground_truth(outcome):
    path = BASE / "all_sample" / "outputs" / f"{outcome}_state_estimates.csv"
    df = pd.read_csv(path, dtype={"state_fips": str})
    return df[["state_fips", "state_name", "estimate"]].rename(columns={"estimate": "ground_truth"})

def load_sample_estimates(sample_dir, outcome):
    """Load model estimates for a sample, using combined file + per-model fallbacks."""
    est_dir = BASE / sample_dir / "outputs" / "estimates"
    combined_path = est_dir / f"{outcome}_state_estimates.csv"
    combined = pd.read_csv(combined_path, dtype={"state_fips": str})

    result = combined[["state_fips"]].copy()
    for model in MODELS:
        if model in combined.columns:
            result[model] = combined[model].values
        else:
            # Try per-model fallback file
            fallback = est_dir / f"{model}_{outcome}_state_estimates.csv"
            if fallback.exists():
                fb = pd.read_csv(fallback, dtype={"state_fips": str})
                merged = result[["state_fips"]].merge(fb[["state_fips", "estimate"]], on="state_fips", how="left")
                result[model] = merged["estimate"].values
            else:
                result[model] = np.nan
    return result

def compute_mad_table(outcome):
    gt = load_ground_truth(outcome)
    gt_mean = gt["ground_truth"].mean()
    rows = []

    for model in MODELS:
        row = {"Model": DISPLAY_NAMES[model], "Ground Truth Mean": round(gt_mean, 4)}
        for sample_dir, n in SAMPLES.items():
            col_name = f"Mean Abs Diff from Ground Truth (n={n:,})"
            estimates = load_sample_estimates(sample_dir, outcome)
            merged = estimates[["state_fips", model]].merge(gt, on="state_fips", how="inner")
            if merged[model].isna().all():
                row[col_name] = "N/A"
            else:
                mad = (merged[model] - merged["ground_truth"]).abs().mean()
                row[col_name] = round(mad, 4)
        rows.append(row)

    return pd.DataFrame(rows)

def compute_max_diff_table(outcome):
    gt = load_ground_truth(outcome)
    gt_mean = gt["ground_truth"].mean()
    rows = []

    for model in MODELS:
        row = {"Model": DISPLAY_NAMES[model], "Ground Truth Mean": round(gt_mean, 4)}
        for sample_dir, n in SAMPLES.items():
            col_name = f"Max Abs Diff from Ground Truth (n={n:,})"
            estimates = load_sample_estimates(sample_dir, outcome)
            merged = estimates[["state_fips", model]].merge(gt, on="state_fips", how="inner")
            if merged[model].isna().all():
                row[col_name] = "N/A"
            else:
                diffs = (merged[model] - merged["ground_truth"]).abs()
                idx = diffs.idxmax()
                max_diff = diffs.loc[idx]
                state_name = merged.loc[idx, "state_name"]
                row[col_name] = f"{max_diff:.0%} ({state_name})"
        rows.append(row)

    return pd.DataFrame(rows)

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    for outcome in OUTCOMES:
        # MAD table
        table = compute_mad_table(outcome)
        print(f"\n{'='*70}")
        print(f"MAD Table: {outcome}")
        print(f"{'='*70}")
        print(table.to_string(index=False))
        csv_path = out_dir / f"mad_table_{outcome}.csv"
        table.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path.name}")

        # Max difference table
        max_table = compute_max_diff_table(outcome)
        print(f"\n{'='*70}")
        print(f"Max Difference Table: {outcome}")
        print(f"{'='*70}")
        print(max_table.to_string(index=False))
        csv_path = out_dir / f"max_diff_table_{outcome}.csv"
        max_table.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path.name}")
