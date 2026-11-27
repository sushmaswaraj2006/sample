"""
ASTRA — Ablation Study
Phase 3: 5 ablation experiments using CatBoost on event-wise split
"""
import os, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ml" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

import argparse as _ap, os as _os
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--train", default=_os.environ.get("ASTRA_TRAIN_CSV", ""))
_p.add_argument("--test",  default=_os.environ.get("ASTRA_TEST_CSV",  ""))
_a, _ = _p.parse_known_args()
ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = Path(_a.train) if _a.train else ROOT / "data" / "train_data.csv"
TEST_CSV  = Path(_a.test)  if _a.test  else ROOT / "data" / "test_data.csv"
HIGH_THRESHOLD = -12
MED_THRESHOLD  = -20
RANDOM_STATE   = 42

# Load event-wise split if available, else recreate
print("Loading data...")
train_raw = pd.read_csv(TRAIN_CSV)
test_raw  = pd.read_csv(TEST_CSV)
full      = pd.concat([train_raw, test_raw], ignore_index=True)

all_events = full["event_id"].unique()
np.random.seed(RANDOM_STATE)
np.random.shuffle(all_events)
split_idx = int(len(all_events) * 0.80)
train_ew = full[full["event_id"].isin(set(all_events[:split_idx]))].copy()
test_ew  = full[full["event_id"].isin(set(all_events[split_idx:]))].copy()

EXCLUDE  = {"event_id", "mission_id", "risk"}
CAT_COLS = ["c_object_type"]
NUM_COLS = [c for c in full.columns if c not in EXCLUDE and c not in CAT_COLS]

def add_flags(df):
    df = df.copy()
    df["covariance_missing"]    = (df["t_ct_r"].isna() | df["c_ct_r"].isna()).astype(int)
    df["t_rcs_missing"]         = df["t_rcs_estimate"].isna().astype(int)
    df["c_rcs_missing"]         = df["c_rcs_estimate"].isna().astype(int)
    df["space_weather_missing"] = df["F10"].isna().astype(int)
    df["c_object_type"]         = df["c_object_type"].fillna("UNKNOWN").astype(str)
    return df

train_ew = add_flags(train_ew)
test_ew  = add_flags(test_ew)

MISS_FLAGS   = ["covariance_missing","t_rcs_missing","c_rcs_missing","space_weather_missing"]
BASE_FEATURES = NUM_COLS + MISS_FLAGS + CAT_COLS
BASE_FEATURES = [f for f in BASE_FEATURES if f in train_ew.columns]

def classify(v, high=-12, med=-20):
    if v > high:  return "High"
    if v > med:   return "Medium"
    return "Low"

def run_experiment(name, features, high_thr=-12):
    cat_idx = [features.index(c) for c in CAT_COLS if c in features]
    model = CatBoostRegressor(
        iterations=400, learning_rate=0.08, depth=7,
        random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False,
        cat_features=cat_idx,
    )
    X_tr = train_ew[features]
    X_te = test_ew[features]
    y_tr = train_ew["risk"]
    y_te = test_ew["risk"]
    model.fit(X_tr, y_tr)
    pred_reg = model.predict(X_te)
    pred_cls = [classify(v, high_thr) for v in pred_reg]
    true_cls = [classify(v, HIGH_THRESHOLD) for v in y_te]
    classes = ["Low","Medium","High"]
    wf1  = round(f1_score(true_cls, pred_cls, average="weighted", labels=classes, zero_division=0), 4)
    hrr  = round(recall_score(true_cls, pred_cls, labels=["High"], average="macro", zero_division=0), 4)
    cm   = confusion_matrix(true_cls, pred_cls, labels=classes).tolist()
    from sklearn.metrics import mean_absolute_error, r2_score
    r2   = round(r2_score(y_te, pred_reg), 4)
    mae  = round(mean_absolute_error(y_te, pred_reg), 4)
    print(f"  {name:45s} | W-F1={wf1:.4f}  High-Recall={hrr:.4f}  R²={r2:.4f}")
    return {"name": name, "weighted_f1": wf1, "high_risk_recall": hrr, "r2": r2, "mae": mae, "confusion_matrix": cm}

print("\nRunning ablation experiments...")
print(f"{'Experiment':45s} | W-F1    High-Recall  R²")
print("-" * 80)

ablation = {}

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEMATIC TARGET-LEAKAGE & FEATURE ABLATION EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────
# Exp A: Full feature set (demonstrating formula leakage: R² ~ 0.9998)
# Exp B: Without max_risk_scaling
# Exp C: Without max_risk_estimate
# Exp D: Without both max_risk_scaling & max_risk_estimate (Clean Astrodynamics Model)
# Exp E: Geometry & Covariance features only
# ─────────────────────────────────────────────────────────────────────────────

# 1. Full baseline (Demonstrates target formula leakage)
ablation["exp_A_full_features_leaky"] = run_experiment(
    "Exp A: Full feature set (Includes target leakage bounds)",
    BASE_FEATURES
)

# 2. Without max_risk_scaling only
no_scaling = [f for f in BASE_FEATURES if f != "max_risk_scaling"]
ablation["exp_B_without_max_risk_scaling"] = run_experiment(
    "Exp B: Without max_risk_scaling",
    no_scaling
)

# 3. Without max_risk_estimate only
no_estimate = [f for f in BASE_FEATURES if f != "max_risk_estimate"]
ablation["exp_C_without_max_risk_estimate"] = run_experiment(
    "Exp C: Without max_risk_estimate",
    no_estimate
)

# 4. Without both (Clean Astrodynamics Model - Exp D)
no_max = [f for f in BASE_FEATURES if f not in ("max_risk_scaling", "max_risk_estimate")]
ablation["exp_D_clean_no_risk_derived"] = run_experiment(
    "Exp D: Clean model (No risk-derived features)",
    no_max
)

# 5. Geometry and Covariance only (Exp E)
geom_cov_features = [
    f for f in no_max
    if any(k in f for k in ["miss", "relative", "sma", "inc", "ecc", "sigma", "covariance", "mahalanobis", "geocentric"])
]
ablation["exp_E_geometry_covariance_only"] = run_experiment(
    "Exp E: Geometry & Covariance features only",
    geom_cov_features
)

# 6. Without missingness indicators
no_miss = [f for f in no_max if f not in MISS_FLAGS]
ablation["without_missingness_indicators"] = run_experiment(
    "Without missingness indicators (on clean feature set)",
    no_miss
)

# 5. High threshold = -10
ablation["high_threshold_neg10"] = run_experiment("High-risk threshold at log10(Pc)=-10", BASE_FEATURES, high_thr=-10)

# 6. High threshold = -15
ablation["high_threshold_neg15"] = run_experiment("High-risk threshold at log10(Pc)=-15", BASE_FEATURES, high_thr=-15)

(RESULTS / "ablation_results.json").write_text(json.dumps(ablation, indent=2))
print(f"\nAblation results saved → {RESULTS / 'ablation_results.json'}")
