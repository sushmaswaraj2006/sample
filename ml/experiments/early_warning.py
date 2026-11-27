"""
ASTRA — Early Warning Evaluation
Phase 3: Per-event CDM sequence analysis to measure early-warning performance
"""
import json, warnings, numpy as np, pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ml" / "results"
MODEL_PATH = ROOT / "backend" / "model" / "catboost_risk_model.cbm"
FEAT_PATH  = ROOT / "backend" / "model" / "selected_features.json"

import argparse as _ap, os as _os
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--train", default=_os.environ.get("ASTRA_TRAIN_CSV", ""))
_p.add_argument("--test",  default=_os.environ.get("ASTRA_TEST_CSV",  ""))
_a, _ = _p.parse_known_args()
ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = Path(_a.train) if _a.train else ROOT / "data" / "train_data.csv"
TEST_CSV  = Path(_a.test)  if _a.test  else ROOT / "data" / "test_data.csv"
HIGH_THR  = -12
RANDOM_STATE = 42

print("Loading data for early-warning evaluation...")
train_raw = pd.read_csv(TRAIN_CSV)
test_raw  = pd.read_csv(TEST_CSV)
full      = pd.concat([train_raw, test_raw], ignore_index=True)

all_events = full["event_id"].unique()
np.random.seed(RANDOM_STATE)
np.random.shuffle(all_events)
split_idx = int(len(all_events) * 0.80)
test_events = set(all_events[split_idx:])
test_df = full[full["event_id"].isin(test_events)].copy()

# Load model and features
schema = json.loads(FEAT_PATH.read_text())
features = schema["features"]
cat_features = schema.get("cat_features", ["c_object_type"])

# Add missingness flags if present in feature list
def add_flags(df):
    df = df.copy()
    if "covariance_missing" in features:
        df["covariance_missing"]    = (df["t_ct_r"].isna() | df["c_ct_r"].isna()).astype(int)
    if "t_rcs_missing" in features:
        df["t_rcs_missing"]         = df["t_rcs_estimate"].isna().astype(int)
    if "c_rcs_missing" in features:
        df["c_rcs_missing"]         = df["c_rcs_estimate"].isna().astype(int)
    if "space_weather_missing" in features:
        df["space_weather_missing"] = df["F10"].isna().astype(int)
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN").astype(str)
    return df

test_df = add_flags(test_df)

# Ensure all features exist
for f in features:
    if f not in test_df.columns:
        test_df[f] = np.nan

model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))
print(f"Model loaded. Evaluating {len(test_events)} test events...")

results = {
    "total_test_events": int(len(test_events)),
    "actual_high_risk_events": 0,
    "events_with_early_warning": 0,
    "early_warning_rate": 0.0,
    "mean_lead_cdms": 0.0,
    "median_lead_cdms": 0.0,
    "max_lead_cdms": 0,
    "false_alarms_total": 0,
    "false_alarms_per_event": 0.0,
    "event_details": []
}

lead_times = []
false_alarms_list = []
high_risk_event_count = 0

for event_id, group in test_df.groupby("event_id"):
    # Sort by time_to_tca descending = earliest CDM first (largest TCA = furthest from collision)
    group = group.sort_values("time_to_tca", ascending=False).reset_index(drop=True)
    true_risks = group["risk"].values
    
    # Find first ACTUAL high-risk CDM
    first_actual_high = None
    for i, r in enumerate(true_risks):
        if r > HIGH_THR:
            first_actual_high = i
            break
    
    if first_actual_high is None:
        # No high-risk CDM in this event - check for false alarms
        X_event = group[features].fillna(np.nan)
        preds = model.predict(X_event)
        false_alarms = int((preds > HIGH_THR).sum())
        if false_alarms > 0:
            false_alarms_list.append(false_alarms)
        continue
    
    high_risk_event_count += 1
    
    # Predict all CDMs BEFORE the first actual high-risk CDM
    if first_actual_high == 0:
        # First CDM is already high-risk — no room for early warning
        results["event_details"].append({
            "event_id": int(event_id),
            "first_actual_high_idx": 0,
            "early_warning": False,
            "lead_cdms": 0,
        })
        continue
    
    X_before = group.iloc[:first_actual_high][features].fillna(np.nan)
    preds_before = model.predict(X_before)
    
    # Did any earlier CDM predict High?
    early_hits = np.where(preds_before > HIGH_THR)[0]
    if len(early_hits) > 0:
        first_hit = early_hits[0]
        lead = first_actual_high - first_hit  # CDMs of lead time
        lead_times.append(lead)
        results["event_details"].append({
            "event_id": int(event_id),
            "first_actual_high_idx": int(first_actual_high),
            "early_warning": True,
            "first_predicted_high_idx": int(first_hit),
            "lead_cdms": int(lead),
        })
    else:
        results["event_details"].append({
            "event_id": int(event_id),
            "first_actual_high_idx": int(first_actual_high),
            "early_warning": False,
            "lead_cdms": 0,
        })

# Aggregate
results["actual_high_risk_events"]    = int(high_risk_event_count)
results["events_with_early_warning"]  = len(lead_times)
results["early_warning_rate"]         = round(len(lead_times) / max(high_risk_event_count, 1), 4)
results["mean_lead_cdms"]             = round(float(np.mean(lead_times)) if lead_times else 0.0, 2)
results["median_lead_cdms"]           = round(float(np.median(lead_times)) if lead_times else 0.0, 2)
results["max_lead_cdms"]              = int(max(lead_times)) if lead_times else 0
results["false_alarms_total"]         = int(sum(false_alarms_list))
results["false_alarms_per_event"]     = round(sum(false_alarms_list)/max(len(false_alarms_list),1), 2)

(RESULTS / "early_warning_results.json").write_text(json.dumps(results, indent=2))

print(f"\n{'='*50}")
print(f"EARLY WARNING RESULTS")
print(f"{'='*50}")
print(f"  Total test events:            {results['total_test_events']}")
print(f"  Actual high-risk events:      {results['actual_high_risk_events']}")
print(f"  Events with early warning:    {results['events_with_early_warning']}")
print(f"  Early-warning rate:           {results['early_warning_rate']*100:.2f}%")
print(f"  Mean lead time (CDMs):        {results['mean_lead_cdms']}")
print(f"  Median lead time (CDMs):      {results['median_lead_cdms']}")
print(f"  Max lead time (CDMs):         {results['max_lead_cdms']}")
print(f"  False alarms (total):         {results['false_alarms_total']}")
print(f"  False alarms per event:       {results['false_alarms_per_event']}")
print(f"\nResults saved → {RESULTS / 'early_warning_results.json'}")
