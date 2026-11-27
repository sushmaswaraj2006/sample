"""
ASTRA — Probability Calibration Analysis
Checks whether the CatBoost model's risk scores are well-calibrated
and applies isotonic regression calibration if needed.

Usage:
    python ml/experiments/calibration.py --train /path/train.csv --test /path/test.csv
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ml" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--train", default=os.environ.get("ASTRA_TRAIN_CSV", ""))
parser.add_argument("--test",  default=os.environ.get("ASTRA_TEST_CSV",  ""))
args, _ = parser.parse_known_args()

TRAIN_CSV = Path(args.train) if args.train else ROOT / "data" / "train_data.csv"
TEST_CSV  = Path(args.test)  if args.test  else ROOT / "data" / "test_data.csv"

MODEL_PATH = ROOT / "ml" / "models" / "catboost_event_wise.cbm"
if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "backend" / "model" / "catboost_risk_model.cbm"

FEATURES_PATH = ROOT / "backend" / "model" / "selected_features.json"

HIGH_THR = -12
MED_THR  = -20
RANDOM_STATE = 42

print("=" * 60)
print("ASTRA CALIBRATION ANALYSIS")
print("=" * 60)

if not TRAIN_CSV.exists() or not TEST_CSV.exists():
    print("Data files not found. Set ASTRA_TRAIN_CSV / ASTRA_TEST_CSV env vars.")
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA (event-wise split)
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading data...")
train_raw = pd.read_csv(TRAIN_CSV)
test_raw  = pd.read_csv(TEST_CSV)
full      = pd.concat([train_raw, test_raw], ignore_index=True)

all_events = full["event_id"].unique()
np.random.seed(RANDOM_STATE)
np.random.shuffle(all_events)
split_idx = int(len(all_events) * 0.80)
train_ew  = full[full["event_id"].isin(set(all_events[:split_idx]))].copy()
test_ew   = full[full["event_id"].isin(set(all_events[split_idx:]))].copy()

# Load feature schema
with open(FEATURES_PATH) as f:
    schema = json.load(f)
features    = schema["features"]
cat_cols    = schema.get("cat_features", [])

def add_flags(df):
    df = df.copy()
    df["covariance_missing"]    = (df["t_ct_r"].isna() | df["c_ct_r"].isna()).astype(int)
    df["t_rcs_missing"]         = df["t_rcs_estimate"].isna().astype(int)
    df["c_rcs_missing"]         = df["c_rcs_estimate"].isna().astype(int)
    df["space_weather_missing"] = df["F10"].isna().astype(int)
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN").astype(str)
    return df

test_ew = add_flags(test_ew)

# Ensure all features exist in test
for f in features:
    if f not in test_ew.columns:
        test_ew[f] = np.nan

print(f"Test set: {len(test_ew):,} CDMs across {test_ew['event_id'].nunique():,} events")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL AND PREDICT
# ─────────────────────────────────────────────────────────────────────────────
from catboost import CatBoostRegressor

model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))
print(f"Model loaded: {MODEL_PATH}")

X_test  = test_ew[features]
y_true  = test_ew["risk"].values
y_pred  = model.predict(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning calibration analysis...")

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Convert to binary classification (High-risk vs not) for calibration
y_true_binary = (y_true > HIGH_THR).astype(int)

# Map predicted log10(Pc) to [0,1] score for calibration check
# Using a sigmoid-style normalization
y_pred_score = 1 / (1 + np.exp(-(y_pred - HIGH_THR)))

# Calibration curve
prob_true, prob_pred = calibration_curve(y_true_binary, y_pred_score, n_bins=10)

# Expected Calibration Error
ece = float(np.mean(np.abs(prob_true - prob_pred)))

# Brier score
brier = brier_score_loss(y_true_binary, y_pred_score)

print(f"\nCalibration Results (High-risk binary, threshold = {HIGH_THR}):")
print(f"  Expected Calibration Error (ECE): {ece:.4f}")
print(f"  Brier Score:                      {brier:.4f}")
print(f"  {'Well-calibrated' if ece < 0.05 else 'Needs calibration'}")

# ─────────────────────────────────────────────────────────────────────────────
# ISOTONIC REGRESSION CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
print("\nFitting isotonic calibration on training predictions...")
train_ew_cal = add_flags(train_ew)
for f in features:
    if f not in train_ew_cal.columns:
        train_ew_cal[f] = np.nan

X_train_cal = train_ew_cal[features]
y_train_true_bin = (train_ew_cal["risk"].values > HIGH_THR).astype(int)
y_train_pred_raw = model.predict(X_train_cal)
y_train_score    = 1 / (1 + np.exp(-(y_train_pred_raw - HIGH_THR)))

from sklearn.isotonic import IsotonicRegression

ir = IsotonicRegression(out_of_bounds="clip")
ir.fit(y_train_score, y_train_true_bin)
y_pred_calibrated = ir.predict(y_pred_score)

prob_true_cal, prob_pred_cal = calibration_curve(y_true_binary, y_pred_calibrated, n_bins=10)
ece_cal   = float(np.mean(np.abs(prob_true_cal - prob_pred_cal)))
brier_cal = brier_score_loss(y_true_binary, y_pred_calibrated)

print(f"\nPost-calibration Results:")
print(f"  ECE (after isotonic):  {ece_cal:.4f} (was {ece:.4f})")
print(f"  Brier Score (after):   {brier_cal:.4f} (was {brier:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
calibration_data = {
    "pre_calibration": {
        "ece": round(ece, 6),
        "brier_score": round(brier, 6),
        "calibration_curve_prob_true": prob_true.tolist(),
        "calibration_curve_prob_pred": prob_pred.tolist(),
        "verdict": "Well-calibrated" if ece < 0.05 else "Needs calibration",
    },
    "post_calibration_isotonic": {
        "ece": round(ece_cal, 6),
        "brier_score": round(brier_cal, 6),
        "calibration_curve_prob_true": prob_true_cal.tolist(),
        "calibration_curve_prob_pred": prob_pred_cal.tolist(),
        "improvement_ece": round(ece - ece_cal, 6),
    },
    "threshold_used": HIGH_THR,
    "n_test_samples": int(len(y_true)),
}

out = RESULTS / "calibration_results.json"
out.write_text(json.dumps(calibration_data, indent=2))

print(f"\nCalibration results saved → {out}")
print("=" * 60)
