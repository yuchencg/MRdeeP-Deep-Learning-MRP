"""
Recodes climate_survey_responses.csv → climate_survey_responses_recoded.csv

Demographics: mapped to match ACS poststrat frame
  (state_fips × gender × race4 × educ_category)

Outcomes: all 30 opinion variables recoded to binary 0/1.
  Don't-know / uncertain responses → NaN (excluded from modeling).
  Answer codes confirmed from Climate_Survey_-_Updated_Draft.qsf.
"""

import pandas as pd
import numpy as np

RAW = '/Users/carmenk/Documents/GitHub/MRdeeP-Deep-Learning-MRP/test_data/raw/climate_survey_responses.csv'
OUT = '/Users/carmenk/Documents/GitHub/MRdeeP-Deep-Learning-MRP/test_data/processed/climate_survey_responses_recoded.csv'

df = pd.read_csv(RAW)

# ══════════════════════════════════════════════════════════════════════
# DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════

# State: Qualtrics alphabetical index → 2-digit FIPS string (zero-padded)
STATE_TO_FIPS = {
    1:'01', 2:'02', 3:'04', 4:'05', 5:'06', 6:'08', 7:'09', 8:'10',
    9:'11', 10:'12', 11:'13', 12:'15', 13:'16', 14:'17', 15:'18',
    16:'19', 17:'20', 18:'21', 19:'22', 20:'23', 21:'24', 22:'25',
    23:'26', 24:'27', 25:'28', 26:'29', 27:'30', 28:'31', 29:'32',
    30:'33', 31:'34', 32:'35', 33:'36', 34:'37', 35:'38', 36:'39',
    37:'40', 38:'41', 39:'42', 40: None,  # Puerto Rico → drop
    41:'44', 42:'45', 43:'46', 44:'47', 45:'48', 46:'49', 47:'50',
    48:'51', 49:'53', 50:'54', 51:'55', 52:'56',
}
STATE_TO_NAME = {
    1:'Alabama', 2:'Alaska', 3:'Arizona', 4:'Arkansas', 5:'California',
    6:'Colorado', 7:'Connecticut', 8:'Delaware', 9:'District of Columbia',
    10:'Florida', 11:'Georgia', 12:'Hawaii', 13:'Idaho', 14:'Illinois',
    15:'Indiana', 16:'Iowa', 17:'Kansas', 18:'Kentucky', 19:'Louisiana',
    20:'Maine', 21:'Maryland', 22:'Massachusetts', 23:'Michigan', 24:'Minnesota',
    25:'Mississippi', 26:'Missouri', 27:'Montana', 28:'Nebraska', 29:'Nevada',
    30:'New Hampshire', 31:'New Jersey', 32:'New Mexico', 33:'New York',
    34:'North Carolina', 35:'North Dakota', 36:'Ohio', 37:'Oklahoma',
    38:'Oregon', 39:'Pennsylvania', 40:'Puerto Rico', 41:'Rhode Island',
    42:'South Carolina', 43:'South Dakota', 44:'Tennessee', 45:'Texas',
    46:'Utah', 47:'Vermont', 48:'Virginia', 49:'Washington',
    50:'West Virginia', 51:'Wisconsin', 52:'Wyoming',
}
df['state_fips'] = df['state'].map(STATE_TO_FIPS)   # string e.g. '01'
df['state_name'] = df['state'].map(STATE_TO_NAME)

n_before = len(df)
df = df[df['state_fips'].notna()].copy()
print(f"Dropped {n_before - len(df)} Puerto Rico respondents → {len(df)} remain")

# Gender: 1=Male, 2=Female
df['gender'] = df['gender'].map({1: 'Male', 2: 'Female'})

# Race4: Hispanic precedence (mirrors ACS extract logic)
#   race   : 1=Hispanic, 2=Non-Hispanic
#   race_2 : multi-select 1=White-NH, 2=Black-NH, 3=Other-NH, 4=Hispanic
def derive_race4(row):
    if row['race'] == 1:
        return 'Hispanic'
    codes = str(row['race_2']).split(',')
    if '1' in codes: return 'White'
    if '2' in codes: return 'Black'
    return 'Other'

df['race4'] = df.apply(derive_race4, axis=1)

# Education: 1–4 maps directly to poststrat educ_category
#   1=Less than HS, 2=HS graduate, 3=Some college, 4=Bachelor's or higher
df['educ_category'] = df['educational_attainment']

# Age group (reference only — not in poststrat frame)
df['age_group'] = df['age'].map({1: '18-34', 2: '35-54', 3: '55+'})

# Marital status (reference only — not in poststrat frame)
df['marstat'] = df['marital_status'].map({
    1: 'Married', 2: 'Widowed', 3: 'Divorced', 4: 'Separated', 5: 'Never married'
})

# ══════════════════════════════════════════════════════════════════════
# BINARY OUTCOME RECODING  (all 30 variables)
# ══════════════════════════════════════════════════════════════════════

def bin_map(series, yes_vals, dk_vals=None):
    """1 for yes_vals, 0 for everything else, NaN for dk_vals."""
    out = series.isin(yes_vals).astype(float)
    if dk_vals:
        out[series.isin(dk_vals)] = np.nan
    return out

# 1. happening — "Is global warming happening?"
#    1=Yes → 1 | 2=No → 0 | 3=Don't know → NaN
df['happening_bin'] = bin_map(df['happening'], [1], dk_vals=[3])

# 2. cause_human_bin — "Caused mostly by human activities?"
#    1=Human → 1 | 2=Natural,3=NotHappening,4=Other → 0 | 5=DK → NaN
df['cause_human_bin'] = bin_map(df['cause_recoded'], [1], dk_vals=[5])

# 3. sci_consensus_bin — "Most scientists think GW is happening?"
#    1=Most think happening → 1 | 2=Disagreement,4=Most think NOT → 0 | 5=DK → NaN
df['sci_consensus_bin'] = bin_map(df['sci_consensus'], [1], dk_vals=[5])

# 4. weather_bin — "GW is affecting weather in the US"
#    1=StronglyAgree,2=SomewhatAgree → 1 | 3=SomewhatDisagree,4=StronglyDisagree → 0
df['weather_bin'] = bin_map(df['weather'], [1, 2])

# 5–8. gw_[event]_bin — "How much is GW affecting [event]?"
#    2=Some or a lot → 1 | 3=A little or not at all → 0 | 5=Don't know → NaN
for raw_col, new_col in [('gw_affecting_1', 'gw_wildfires_bin'),
                          ('gw_affecting_2', 'gw_droughts_bin'),
                          ('gw_affecting_3', 'gw_flooding_bin'),
                          ('gw_affecting_4', 'gw_hurricanes_bin')]:
    df[new_col] = bin_map(df[raw_col], [2], dk_vals=[5])

# 9. personal_effects_bin — "I have personally experienced effects of GW"
#    1=StronglyAgree,2=SomewhatAgree → 1 | 3=SomewhatDisagree,4=StronglyDisagree → 0
df['personal_effects_bin'] = bin_map(df['personal_effects'], [1, 2])

# 10. worried_bin — "How worried are you about GW?"
#    1=VeryWorried,2=SomewhatWorried → 1 | 3=NotVery,4=NotAtAll → 0
df['worried_bin'] = bin_map(df['worried'], [1, 2])

# 11. when_harm_soon_bin — "When will GW harm people in the US?"
#    4=In25yrs,5=In10yrs,6=RightNow → 1 | 1=Never,2=100yrs,3=50yrs → 0
df['when_harm_soon_bin'] = bin_map(df['when_harm_US'], [4, 5, 6])

# 12–16. harm_[entity]_bin — "How much will GW harm [entity]?"
#    4=ModerateOrGreatDeal → 1 | 1=NotAtAll/Little → 0 | 5=DK → NaN
for raw_col, new_col in [('Q64_1', 'harm_plants_bin'),
                          ('Q64_2', 'harm_future_gen_bin'),
                          ('Q64_3', 'harm_dev_countries_bin'),
                          ('Q64_4', 'harm_US_bin'),
                          ('Q64_5', 'harm_personally_bin')]:
    df[new_col] = bin_map(df[raw_col], [4], dk_vals=[5])

# 17–21. policy_[name]_bin — "Do you support this policy?"
#    1=Support → 1 | 3=Oppose → 0
#    Items: clean economy(1), fund research(2), renewable land(3), reg CO2(4), teach GW(5)
for raw_col, new_col in [('Q72_1', 'policy_clean_econ_bin'),
                          ('Q72_2', 'policy_research_bin'),
                          ('Q72_3', 'policy_renewable_bin'),
                          ('Q72_4', 'policy_reg_co2_bin'),
                          ('Q72_5', 'policy_teach_bin')]:
    df[new_col] = bin_map(df[raw_col], [1])

# 22. clean_energy_priority_bin — "Should clean energy be a priority?"
#    1=VeryHigh,2=High → 1 | 3=Medium,4=Low → 0
df['clean_energy_priority_bin'] = bin_map(df['ce_pres_cong'], [1, 2])

# 23–28. who_[actor]_bin — "Should [actor] do more about GW?"
#    1=More → 1 | 2=RightAmount,3=Less → 0
#    Actors: president(1), congress(2), governor(3), local(4), corps(5), citizens(6)
for raw_col, new_col in [('who_gw_1', 'who_president_bin'),
                          ('who_gw_2', 'who_congress_bin'),
                          ('who_gw_3', 'who_governor_bin'),
                          ('who_gw_4', 'who_local_bin'),
                          ('who_gw_5', 'who_corps_bin'),
                          ('who_gw_6', 'who_citizens_bin')]:
    df[new_col] = bin_map(df[raw_col], [1])

# 29. discuss_gw_bin — "How often do you discuss GW with family/friends?"
#    1=Often,2=Occasionally → 1 | 3=Rarely,4=Never → 0
df['discuss_gw_bin'] = bin_map(df['discuss_GW'], [1, 2])

# 30. hear_gw_bin — "How often do you hear about GW in media?"
#    5=AtLeastOnceAMonth,6=AtLeastOnceAWeek → 1 | 2=Never,3=Once/yr,4=Several/yr → 0
df['hear_gw_bin'] = bin_map(df['hear_GW_media'], [5, 6])

# ══════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════

DEMOG_COLS = [
    'ResponseId', 'state_fips', 'state_name',
    'gender', 'race4', 'educ_category', 'age_group', 'marstat',
]

OUTCOME_COLS = [
    'happening_bin',
    'cause_human_bin', 'sci_consensus_bin', 'weather_bin',
    'gw_wildfires_bin', 'gw_droughts_bin', 'gw_flooding_bin', 'gw_hurricanes_bin',
    'personal_effects_bin', 'worried_bin', 'when_harm_soon_bin',
    'harm_plants_bin', 'harm_future_gen_bin', 'harm_dev_countries_bin',
    'harm_US_bin', 'harm_personally_bin',
    'policy_clean_econ_bin', 'policy_research_bin', 'policy_renewable_bin',
    'policy_reg_co2_bin', 'policy_teach_bin',
    'clean_energy_priority_bin',
    'who_president_bin', 'who_congress_bin', 'who_governor_bin',
    'who_local_bin', 'who_corps_bin', 'who_citizens_bin',
    'discuss_gw_bin', 'hear_gw_bin',
]

out = df[DEMOG_COLS + OUTCOME_COLS]
out.to_csv(OUT, index=False)

# ── Validation ─────────────────────────────────────────────────────────────────
n = len(out)
print(f"\nSaved → {OUT}")
print(f"Shape: {out.shape}  ({len(OUTCOME_COLS)} binary outcomes)\n")

# Check all outcomes are 0/1/NaN only
bad = [c for c in OUTCOME_COLS if not set(out[c].dropna().unique()).issubset({0.0,1.0})]
if bad:
    print(f"WARNING — non-binary values found in: {bad}")
else:
    print("All 30 outcome columns verified as binary (0/1/NaN) ✓\n")

print(f"{'Outcome':<30} {'N_valid':>8}  {'%_Yes':>7}  {'%_DK':>7}")
print("-" * 58)
for col in OUTCOME_COLS:
    valid   = out[col].notna().sum()
    pct_yes = out[col].mean() * 100
    pct_dk  = (n - valid) / n * 100
    print(f"  {col:<28} {valid:>8,}  {pct_yes:>6.1f}%  {pct_dk:>6.1f}%")

print(f"\nDemographic distributions:")
for col, vals in [('gender', None), ('race4', None),
                  ('educ_category', None), ('state_fips', None)]:
    if vals is None:
        vc = out[col].value_counts()
    print(f"  {col}: {dict(vc.head(6))}")
print(f"  state_fips sample (first 5): {sorted(out['state_fips'].unique())[:5]}")
