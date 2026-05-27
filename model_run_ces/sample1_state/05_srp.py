"""
SRP: Stacked Regression Poststratification.
Five base learners (HLM, LASSO, KNN, RF, XGBoost) stacked with 5-fold CV NNLS weights.
CES run: sample_ces_1000, state-level estimates.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import expit
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    OUTPUT_DIR,
    STATE_FIPS_TO_NAME,
    POSTSTRAT_STATE_PATH,
    SURVEY_PATH,
    _load_state_covariates,
    save_estimates,
)

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).resolve().parent.parent.parent
OUTCOME    = ["climate_problem", "renewable_fuel"]
MODEL_NAME = "srp"
SEED         = 42
N_FOLDS      = 5
DEMOG        = ["gender", "race4", "educ_category", "state_fips"]

# ── Load shared data once ─────────────────────────────────────────────────────

ps_frame = pd.read_csv(POSTSTRAT_STATE_PATH, dtype={"state_fips": str})
ps_frame["educ_category"] = ps_frame["educ_category"].astype(str)

state_cov = _load_state_covariates()

# numeric division from poststrat state frame
state_div = ps_frame.groupby("state_fips")["division"].first().reset_index()
state_cov = state_cov.merge(state_div, on="state_fips", how="left")

div_cats   = sorted(state_cov["division"].dropna().unique())
n_div      = len(div_cats)
state_cats = sorted(ps_frame["state_fips"].unique())

sc = state_cov.set_index("state_fips").loc[state_cats].reset_index()

def _std(x):
    return (x - x.mean()) / x.std()

co2_std     = _std(sc["co2_per_capita"].values)
pres_std    = _std(sc["dem_share_two_party"].values)
drive_std   = _std(sc["drive_alone_share"].values)
samesex_std = _std(sc["samesex_share"].values)

state_div_idx = pd.Categorical(
    sc["division"].astype(str),
    categories=[str(d) for d in div_cats],
).codes

# ── Base learner helpers ──────────────────────────────────────────────────────

def fit_hlm(y_tr, gi, si, ri, ei, n_groups):
    """Howe (2015) 3-level logistic MAP."""
    ng, ns, nr, ne = n_groups

    def _unpack(p):
        g0             = p[0]
        sig            = np.exp(p[1:6])
        gc, gp, gd, gs = p[6], p[7], p[8], p[9]
        off  = 10
        ug   = p[off:off+ng];    off += ng
        ur   = p[off:off+nr];    off += nr
        ue   = p[off:off+ne];    off += ne
        ureg = p[off:off+n_div]; off += n_div
        us   = p[off:off+ns]
        return g0, sig, gc, gp, gd, gs, ug, ur, ue, ureg, us

    def neg_lp(p):
        from scipy.optimize import minimize  # noqa: F401
        g0, sig, gc, gp, gd, gs, ug, ur, ue, ureg, us = _unpack(p)
        sgr, sr, se, ss, sreg = sig
        eta    = g0 + ug[gi] + ur[ri] + ue[ei] + us[si]
        ll     = np.sum(y_tr * eta - np.logaddexp(0.0, eta))
        mu_s   = ureg[state_div_idx] + gc*co2_std + gp*pres_std + gd*drive_std + gs*samesex_std
        lp_s   = -0.5*np.sum((us - mu_s)**2)/ss**2    - ns*np.log(ss)
        lp_reg = -0.5*np.sum(ureg**2)/sreg**2          - n_div*np.log(sreg)
        lp_g   = -0.5*np.sum(ug**2)/sgr**2             - ng*np.log(sgr)
        lp_r   = -0.5*np.sum(ur**2)/sr**2              - nr*np.log(sr)
        lp_e   = -0.5*np.sum(ue**2)/se**2              - ne*np.log(se)
        lp_hyp = (-0.5*g0**2/1.5**2
                  - 0.5*gc**2 - 0.5*gp**2 - 0.5*gd**2 - 0.5*gs**2
                  + np.sum(-0.5*sig**2/2.5**2 + p[1:6]))
        return -(ll + lp_s + lp_reg + lp_g + lp_r + lp_e + lp_hyp)

    from scipy.optimize import minimize
    n_p = 1 + 5 + 4 + ng + nr + ne + n_div + ns
    x0  = np.zeros(n_p)
    x0[1:6] = np.log(0.5)
    res = minimize(neg_lp, x0, method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": 1e-9, "gtol": 1e-6})
    g0, _, gc, gp, gd, gs, ug, ur, ue, ureg, us = _unpack(res.x)

    def predict_hlm(X):
        gi_, si_, ri_, ei_ = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        return expit(g0 + ug[gi_] + ur[ri_] + ue[ei_] + us[si_])

    return predict_hlm


def fit_lasso(y_tr, X_tr_ohe, _):
    m = LassoCV(cv=5, random_state=SEED, max_iter=5000).fit(X_tr_ohe, y_tr)
    return m.predict


def fit_knn(y_tr, X_tr_ohe):
    best_k, best_mse = 1, np.inf
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for k in range(1, 202):
        preds = np.zeros(len(y_tr))
        for tr, va in kf.split(X_tr_ohe):
            m = KNeighborsRegressor(n_neighbors=k)
            m.fit(X_tr_ohe[tr], y_tr[tr])
            preds[va] = m.predict(X_tr_ohe[va])
        mse = np.mean((y_tr - preds)**2)
        if mse < best_mse:
            best_mse, best_k = mse, k
    m = KNeighborsRegressor(n_neighbors=best_k).fit(X_tr_ohe, y_tr)
    print(f"  KNN best k={best_k}")
    return m.predict


def fit_rf(y_tr, X_tr):
    m = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return m.predict


def fit_xgb(y_tr, X_tr):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    cv_res = xgb.cv(
        params={"objective": "reg:squarederror", "eval_metric": "rmse",
                "eta": 0.02, "seed": SEED},
        dtrain=dtrain, num_boost_round=int(50 / 0.02),
        nfold=5, early_stopping_rounds=20, verbose_eval=False,
    )
    best_n = int(cv_res["test-rmse-mean"].idxmin()) + 1
    m = xgb.train(
        params={"objective": "reg:squarederror", "eval_metric": "rmse",
                "eta": 0.02, "seed": SEED},
        dtrain=dtrain, num_boost_round=best_n, verbose_eval=False,
    )
    print(f"  XGBoost best n_rounds={best_n}")
    return lambda X: m.predict(xgb.DMatrix(X))


# ── Run for each outcome ──────────────────────────────────────────────────────

import time

for OUTCOME_VAR in OUTCOME:
    print(f"\n{'='*60}\nOutcome: {OUTCOME_VAR}\n{'='*60}")
    start = time.time()

    # load survey
    raw = pd.read_csv(SURVEY_PATH, dtype={"state_fips": str})
    survey = raw.dropna(subset=["gender", "educ_category", OUTCOME_VAR]).copy()
    survey["educ_category"] = survey["educ_category"].astype(str)

    y = survey[OUTCOME_VAR].values.astype(float)

    gender_cats = sorted(survey["gender"].unique())
    race_cats   = sorted(survey["race4"].unique())
    educ_cats   = sorted(survey["educ_category"].unique())
    n_g, n_s, n_r, n_e = len(gender_cats), len(state_cats), len(race_cats), len(educ_cats)
    N_GROUPS = [n_g, n_s, n_r, n_e]

    def label_encode(df):
        return np.column_stack([
            pd.Categorical(df["gender"],        categories=gender_cats).codes,
            pd.Categorical(df["state_fips"],    categories=state_cats).codes,
            pd.Categorical(df["race4"],         categories=race_cats).codes,
            pd.Categorical(df["educ_category"], categories=educ_cats).codes,
        ])

    X_label    = label_encode(survey)
    X_ps_label = label_encode(ps_frame)

    g_idx = X_label[:, 0]; s_idx = X_label[:, 1]
    r_idx = X_label[:, 2]; e_idx = X_label[:, 3]

    enc      = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_ohe    = enc.fit_transform(survey[DEMOG])
    X_ps_ohe = enc.transform(ps_frame[DEMOG])

    print(f"Respondents: {len(y):,}  ({y.mean()*100:.1f}% support)")
    print(f"Label shape: {X_label.shape}  |  OHE shape: {X_ohe.shape}")

    # 5-fold CV to learn stacking weights
    kf  = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), 5))

    for fold, (tr, va) in enumerate(kf.split(X_label)):
        print(f"Fold {fold+1}/{N_FOLDS}")
        y_tr, y_va   = y[tr], y[va]
        Xl_tr, Xl_va = X_label[tr], X_label[va]
        Xo_tr, Xo_va = X_ohe[tr],   X_ohe[va]

        gi_tr = Xl_tr[:, 0]; si_tr = Xl_tr[:, 1]
        ri_tr = Xl_tr[:, 2]; ei_tr = Xl_tr[:, 3]

        pred_hlm   = fit_hlm(y_tr, gi_tr, si_tr, ri_tr, ei_tr, N_GROUPS)
        pred_lasso = fit_lasso(y_tr, Xo_tr, Xo_va)
        pred_knn   = fit_knn(y_tr, Xo_tr)
        pred_rf    = fit_rf(y_tr, Xl_tr)
        pred_xgb   = fit_xgb(y_tr, Xl_tr)

        oof[va, 0] = pred_hlm(Xl_va)
        oof[va, 1] = pred_lasso(Xo_va)
        oof[va, 2] = pred_knn(Xo_va)
        oof[va, 3] = pred_rf(Xl_va)
        oof[va, 4] = pred_xgb(Xl_va)

    stack_weights, _ = nnls(oof, y)
    stack_weights    = stack_weights / stack_weights.sum()

    print("\nStack weights (HLM / LASSO / KNN / RF / XGBoost):")
    for name, w in zip(["HLM", "LASSO", "KNN", "RF", "XGBoost"], stack_weights):
        print(f"  {name:<8} {w:.4f}")

    # retrain on full data
    print("\nRetraining on full dataset...")
    pred_hlm_full   = fit_hlm(y, g_idx, s_idx, r_idx, e_idx, N_GROUPS)
    pred_lasso_full = fit_lasso(y, X_ohe, X_ps_ohe)
    pred_knn_full   = fit_knn(y, X_ohe)
    pred_rf_full    = fit_rf(y, X_label)
    pred_xgb_full   = fit_xgb(y, X_label)

    M = np.column_stack([
        pred_hlm_full(X_ps_label),
        pred_lasso_full(X_ps_ohe),
        pred_knn_full(X_ps_ohe),
        pred_rf_full(X_ps_label),
        pred_xgb_full(X_ps_label),
    ])
    ps_frame["predicted_prob"] = np.clip(M @ stack_weights, 0, 1)

    result = (
        ps_frame
        .groupby("state_fips")
        .apply(lambda g: np.average(g["predicted_prob"], weights=g["N"]),
               include_groups=False)
        .reset_index(name="estimate")
    )
    result["state_name"] = result["state_fips"].map(STATE_FIPS_TO_NAME)

    n_resp = (
        survey.groupby("state_fips")[OUTCOME_VAR]
        .count()
        .reset_index(name="n_respondents")
    )
    result = result.merge(n_resp, on="state_fips", how="left")
    result["n_respondents"] = result["n_respondents"].fillna(0).astype(int)

    save_estimates(result, MODEL_NAME, OUTCOME_VAR)

    diag_dir  = OUTPUT_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_path = diag_dir / f"{MODEL_NAME}_{OUTCOME_VAR}_summary.txt"
    valid = result["estimate"].dropna()

    with open(diag_path, "w") as f:
        f.write(f"SRP — Diagnostic Summary\n{'='*55}\n\n")
        f.write(f"Outcome variable: {OUTCOME_VAR}\n")
        f.write(f"Base learners: HLM, LASSO, KNN, RF, XGBoost\n")
        f.write(f"CV folds: {N_FOLDS}\n")
        f.write(f"Observations: {len(y)}\n\n")
        f.write("Stack weights:\n")
        for name, w in zip(["HLM", "LASSO", "KNN", "RF", "XGBoost"], stack_weights):
            f.write(f"  {name:<8} {w:.4f}\n")
        f.write(f"\nState Estimate Summary:\n")
        f.write(f"  Mean:   {valid.mean():.4f}\n")
        f.write(f"  Median: {valid.median():.4f}\n")
        f.write(f"  Min:    {valid.min():.4f}\n")
        f.write(f"  Max:    {valid.max():.4f}\n")
        f.write(f"  Std:    {valid.std():.4f}\n\n")
        f.write(f"State-level estimates:\n")
        f.write(result[["state_fips", "state_name", "estimate"]].to_string(index=False))

    elapsed = (time.time() - start) / 60
    print(f"Diagnostics → {diag_path}")
    print(f"Elapsed: {elapsed:.1f} min")
