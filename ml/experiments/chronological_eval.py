"""
ASTRA — Chronological Evaluation Script
Phase 2 Tier B: Train on events with earlier TCAs, test on later ones.
Provides a temporal validity check complementary to the random event-wise split.

Usage:
    python ml/experiments/chronological_eval.py --train /path/train.csv --test /path/test.csv
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ml" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--train", default=os.environ.get("ASTRA_TRAIN_CSV", ""))
parser.add_argument("--test",  default=os.environ.get("ASTRA_TEST_CSV",  ""))
args, _ = parser.parse_known_args()

TRAIN_CSV = Path(args.train) if args.train else ROOT / "data" / "train_data.csv"
TEST_CSV  = Path(args.test)  if args.test  else ROOT / "data" / "test_data.csv"

HIGH_THRESHOLD = -12
MED_THRESHOLD  = -20
RANDOM_STATE   = 42

print("=" * 60)
print("ASTRA CHRONOLOGICAL EVALUATION")
print("=" * 60)

if not TRAIN_CSV.exists() or not TEST_CSV.exists():
    print("Data files not found.")
    raise SystemExit(1)

print("\nLoading data...")
train_raw = pd.read_csv(TRAIN_CSV)
test_raw  = pd.read_csv(TEST_CSV)
full      = pd.concat([train_raw, test_raw], ignore_index=True)

FEATURES_PATH = ROOT / "backend" / "model" / "selected_features.json"
with open(FEATURES_PATH) as f:
    schema = json.load(f)
features = schema["features"]
cat_cols = schema.get("cat_features", [])

MISS_FLAGS = ["covariance_missing", "t_rcs_missing", "c_rcs_missing", "space_weather_missing"]

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

# ─────────────────────────────────────────────────────────────────────────────
# CHRONOLOGICAL SPLIT
# Sort events by genuine absolute calendar / epoch time:
# Priority 1: 'tca' (Time of Closest Approach absolute epoch / MJD)
# Priority 2: 'epoch' or 'creation_date'
# Priority 3: 't_time_lastob_start' (earliest observation time)
# Priority 4: 'event_id' (sequential campaign event assignment)
# (Note: 'time_to_tca' represents relative lead time, NOT calendar date).
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding chronological split...")

time_col = None
for candidate in ["tca", "epoch", "creation_date", "t_time_lastob_start", "event_id"]:
    if candidate in full.columns:
        time_col = candidate
        break

print(f"  Chronological ordering key: {time_col}")

if time_col == "event_id":
    event_order = (
        full.groupby("event_id")["event_id"]
        .first()
        .reset_index(name="chrono_time")
        .sort_values("chrono_time")
    )
else:
    event_order = (
        full.groupby("event_id")[time_col]
        .min()
        .reset_index(name="chrono_time")
        .sort_values("chrono_time")
    )

n_events = len(event_order)
split_idx = int(n_events * 0.80)

train_events = set(event_order.iloc[:split_idx]["event_id"])
test_events  = set(event_order.iloc[split_idx:]["event_id"])

train_chr = full[full["event_id"].isin(train_events)].copy()
test_chr  = full[full["event_id"].isin(test_events)].copy()

print(f"  Train: {len(train_events):,} events, {len(train_chr):,} CDMs")
print(f"  Test:  {len(test_events):,} events, {len(test_chr):,} CDMs")
print(f"  Overlap: {len(train_events & test_events)} events")

train_chr = add_flags(train_chr)
test_chr  = add_flags(test_chr)

for f in features:
    if f not in train_chr.columns:
        train_chr[f] = np.nan
    if f not in test_chr.columns:
        test_chr[f] = np.nan

y_train = train_chr["risk"]
y_test  = test_chr["risk"]

X_train = train_chr[features]
X_test  = test_chr[features]

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN CATBOOST
# ─────────────────────────────────────────────────────────────────────────────
from catboost import CatBoostRegressor
from sklearn.metrics import f1_score, recall_score, mean_absolute_error, r2_score, mean_squared_error

cat_idx = [features.index(c) for c in cat_cols if c in features]
model = CatBoostRegressor(
    iterations=800, learning_rate=0.05, depth=8,
    random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False,
    cat_features=cat_idx,
)

print("\nTraining CatBoost on chronological train split...")
model.fit(X_train, y_train)

y_pred_reg = model.predict(X_test)

def classify(v):
    if v > HIGH_THRESHOLD: return "High"
    if v > MED_THRESHOLD:  return "Medium"
    return "Low"

y_pred_cls = [classify(v) for v in y_pred_reg]
y_true_cls = [classify(v) for v in y_test]
classes = ["Low", "Medium", "High"]

wf1 = f1_score(y_true_cls, y_pred_cls, average="weighted", labels=classes, zero_division=0)
hrr = recall_score(y_true_cls, y_pred_cls, labels=["High"], average="macro", zero_division=0)
r2  = r2_score(y_test, y_pred_reg)
mae = mean_absolute_error(y_test, y_pred_reg)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_reg)))

results = {
    "split_type": "chronological",
    "train_events": int(len(train_events)),
    "test_events":  int(len(test_events)),
    "train_cdms":   int(len(train_chr)),
    "test_cdms":    int(len(test_chr)),
    "weighted_f1":      round(wf1, 4),
    "high_risk_recall": round(hrr, 4),
    "r2":               round(r2,  4),
    "mae":              round(mae, 4),
    "rmse":             round(rmse, 4),
}

print(f"\n{'='*50}")
print(f"CHRONOLOGICAL EVALUATION RESULTS")
print(f"{'='*50}")
print(f"  Weighted F1:       {wf1:.4f}")
print(f"  High-Risk Recall:  {hrr:.4f}")
print(f"  R²:                {r2:.4f}")
print(f"  MAE:               {mae:.4f}")
print(f"  RMSE:              {rmse:.4f}")
print(f"\n  Compare to event-wise split in model_comparison.json")

out = RESULTS / "chronological_eval_results.json"
out.write_text(json.dumps(results, indent=2))
print(f"\nResults saved → {out}")
