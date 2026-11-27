"""
ASTRA — Full ML Pipeline
Phase 2: Event-wise split + 5 baseline models + comprehensive evaluation
Phase 3: Ablation + early-warning
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
import argparse, os as _os

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--train", default=_os.environ.get("ASTRA_TRAIN_CSV", ""))
_parser.add_argument("--test",  default=_os.environ.get("ASTRA_TEST_CSV",  ""))
_parser.add_argument("--include-leaky", action="store_true", default=False, help="Include max_risk_scaling/estimate")
_args, _ = _parser.parse_known_args()

_default_train = ROOT / "data" / "train_data.csv"
_default_test  = ROOT / "data" / "test_data.csv"

TRAIN_CSV = Path(_args.train) if _args.train else _default_train
TEST_CSV  = Path(_args.test)  if _args.test  else _default_test

if not TRAIN_CSV.exists():
    raise FileNotFoundError(
        f"Train CSV not found: {TRAIN_CSV}\n"
        "Set ASTRA_TRAIN_CSV or pass --train <path>"
    )
if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Test CSV not found: {TEST_CSV}\n"
        "Set ASTRA_TEST_CSV or pass --test <path>"
    )
RESULTS   = ROOT / "ml" / "results"
MODELS    = ROOT / "ml" / "models"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
HIGH_THRESHOLD  = -12
MED_THRESHOLD   = -20

# ─────────────────────────────────────────────
# STEP 1 — LOAD & COMBINE
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading datasets")
print("=" * 60)
train_raw = pd.read_csv(TRAIN_CSV)
test_raw  = pd.read_csv(TEST_CSV)
full      = pd.concat([train_raw, test_raw], ignore_index=True)
print(f"  Combined: {len(full):,} CDMs across {full['event_id'].nunique():,} unique events")

# ─────────────────────────────────────────────
# STEP 2 — EVENT-WISE SPLIT (fix leakage)
# ─────────────────────────────────────────────
print("\nSTEP 2: Event-wise split (no event overlap between train and test)")
all_events = full["event_id"].unique()
np.random.seed(RANDOM_STATE)
np.random.shuffle(all_events)
split_idx = int(len(all_events) * 0.80)
train_events_ew = set(all_events[:split_idx])
test_events_ew  = set(all_events[split_idx:])

train_ew = full[full["event_id"].isin(train_events_ew)].copy()
test_ew  = full[full["event_id"].isin(test_events_ew)].copy()

split_info = {
    "split_type": "event_wise_80_20",
    "train_events": int(len(train_events_ew)),
    "test_events":  int(len(test_events_ew)),
    "train_cdms":   int(len(train_ew)),
    "test_cdms":    int(len(test_ew)),
    "overlap_check": int(len(train_events_ew & test_events_ew)),
    "random_state": RANDOM_STATE,
}
print(f"  Train: {split_info['train_events']} events, {split_info['train_cdms']:,} CDMs")
print(f"  Test:  {split_info['test_events']} events, {split_info['test_cdms']:,} CDMs")
print(f"  Event overlap: {split_info['overlap_check']} (should be 0)")
(RESULTS / "event_split_info.json").write_text(json.dumps(split_info, indent=2))

# ─────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\nSTEP 3: Feature engineering")

EXCLUDE = {"event_id", "mission_id", "risk"}
CAT_COLS = ["c_object_type"]
NUMERIC_COLS = [c for c in full.columns if c not in EXCLUDE and c not in CAT_COLS]

def add_missingness_flags(df):
    df = df.copy()
    df["covariance_missing"] = (df["t_ct_r"].isna() | df["c_ct_r"].isna()).astype(int)
    df["t_rcs_missing"]      = df["t_rcs_estimate"].isna().astype(int)
    df["c_rcs_missing"]      = df["c_rcs_estimate"].isna().astype(int)
    df["space_weather_missing"] = df["F10"].isna().astype(int)
    return df

train_ew = add_missingness_flags(train_ew)
test_ew  = add_missingness_flags(test_ew)

MISS_FLAGS = ["covariance_missing", "t_rcs_missing", "c_rcs_missing", "space_weather_missing"]
ALL_FEATURES = NUMERIC_COLS + MISS_FLAGS + CAT_COLS
if not _args.include_leaky:
    LEAKY_FEATURES = {"max_risk_scaling", "max_risk_estimate"}
    ALL_FEATURES = [f for f in ALL_FEATURES if f not in LEAKY_FEATURES]
    print(f"  [ANTI-LEAKAGE] Excluded {LEAKY_FEATURES} from training features.")
else:
    print("  [WARNING] Including max_risk_scaling/estimate (potential target leakage).")
# Remove features not in dataframe
ALL_FEATURES = [f for f in ALL_FEATURES if f in train_ew.columns]
NUM_FEATURES  = [f for f in ALL_FEATURES if f not in CAT_COLS]

print(f"  Total features: {len(ALL_FEATURES)} ({len(CAT_COLS)} categorical)")

def classify_risk(series):
    return pd.cut(
        series,
        bins=[-np.inf, MED_THRESHOLD, HIGH_THRESHOLD, np.inf],
        labels=["Low", "Medium", "High"],
    )

train_ew["tier"] = classify_risk(train_ew["risk"])
test_ew["tier"]  = classify_risk(test_ew["risk"])

# Fill categoricals
for df in [train_ew, test_ew]:
    df["c_object_type"] = df["c_object_type"].fillna("UNKNOWN").astype(str)

# Prepare arrays for sklearn models (numeric only, median imputation)
from sklearn.impute import SimpleImputer

X_train_num = train_ew[NUM_FEATURES].copy()
X_test_num  = test_ew[NUM_FEATURES].copy()
y_train_cls = train_ew["tier"].astype(str)
y_test_cls  = test_ew["tier"].astype(str)
y_train_reg = train_ew["risk"]
y_test_reg  = test_ew["risk"]

imputer = SimpleImputer(strategy="median")
# Replace infinities with NaN before imputation so sklearn doesn't crash
X_train_num.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_num.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train_imp = imputer.fit_transform(X_train_num)
X_test_imp  = imputer.transform(X_test_num)
# Clip any residual large values to float32 safe range
X_train_imp = np.clip(X_train_imp, -3.4e38, 3.4e38)
X_test_imp  = np.clip(X_test_imp,  -3.4e38, 3.4e38)

CLASS_ORDER = ["Low", "Medium", "High"]

# ─────────────────────────────────────────────
# STEP 4 — TRAIN & EVALUATE ALL MODELS
# ─────────────────────────────────────────────
print("\nSTEP 4: Training and evaluating 5 baseline models")

from sklearn.metrics import (
    classification_report, f1_score, recall_score,
    precision_score, confusion_matrix, average_precision_score
)
from sklearn.preprocessing import LabelEncoder, label_binarize

le = LabelEncoder()
le.fit(CLASS_ORDER)
y_train_enc = le.transform(y_train_cls)
y_test_enc  = le.transform(y_test_cls)

def compute_metrics(y_true_str, y_pred_str, model_name=""):
    labels = CLASS_ORDER
    report = classification_report(y_true_str, y_pred_str, labels=labels, output_dict=True, zero_division=0)
    high_recall = report.get("High", {}).get("recall", 0.0)
    return {
        "model": model_name,
        "weighted_f1":    round(f1_score(y_true_str, y_pred_str, average="weighted", labels=labels, zero_division=0), 4),
        "macro_f1":       round(f1_score(y_true_str, y_pred_str, average="macro",    labels=labels, zero_division=0), 4),
        "high_risk_recall": round(high_recall, 4),
        "high_risk_precision": round(report.get("High", {}).get("precision", 0.0), 4),
        "medium_recall":  round(report.get("Medium", {}).get("recall", 0.0), 4),
        "low_recall":     round(report.get("Low", {}).get("recall", 0.0), 4),
        "accuracy":       round(report.get("accuracy", 0.0), 4),
        "per_class":      {k: v for k, v in report.items() if k in CLASS_ORDER},
        "confusion_matrix": confusion_matrix(y_true_str, y_pred_str, labels=labels).tolist(),
    }

results = {"split_type": "event_wise_80_20", **split_info, "models": {}}

# --- 4a. Logistic Regression ---
print("  [1/5] Logistic Regression...")
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, C=0.1)
lr.fit(X_train_imp, y_train_cls)
lr_pred = lr.predict(X_test_imp)
results["models"]["logistic_regression"] = compute_metrics(y_test_cls, lr_pred, "Logistic Regression")
print(f"    Weighted F1: {results['models']['logistic_regression']['weighted_f1']:.4f}  High-Risk Recall: {results['models']['logistic_regression']['high_risk_recall']:.4f}")

# --- 4b. Random Forest ---
print("  [2/5] Random Forest...")
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, max_depth=20)
rf.fit(X_train_imp, y_train_cls)
rf_pred = rf.predict(X_test_imp)
results["models"]["random_forest"] = compute_metrics(y_test_cls, rf_pred, "Random Forest")
print(f"    Weighted F1: {results['models']['random_forest']['weighted_f1']:.4f}  High-Risk Recall: {results['models']['random_forest']['high_risk_recall']:.4f}")

# --- 4c. XGBoost ---
print("  [3/5] XGBoost...")
try:
    import xgboost as xgb
    n_high = (y_train_cls == "High").sum()
    n_total = len(y_train_cls)
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric="mlogloss",
        verbosity=0,
    )
    xgb_model.fit(X_train_imp, y_train_enc)
    xgb_pred_enc = xgb_model.predict(X_test_imp)
    xgb_pred = le.inverse_transform(xgb_pred_enc)
    results["models"]["xgboost"] = compute_metrics(y_test_cls, xgb_pred, "XGBoost")
    print(f"    Weighted F1: {results['models']['xgboost']['weighted_f1']:.4f}  High-Risk Recall: {results['models']['xgboost']['high_risk_recall']:.4f}")
except (ImportError, Exception) as e:
    print(f"    Native XGBoost unavailable ({type(e).__name__}: {str(e)[:50]}). Training GradientBoosting fallback...")
    from sklearn.ensemble import GradientBoostingClassifier
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE)
    gb_model.fit(X_train_imp, y_train_cls)
    gb_pred = gb_model.predict(X_test_imp)
    results["models"]["xgboost"] = compute_metrics(y_test_cls, gb_pred, "XGBoost (GradientBoosting fallback)")
    print(f"    Weighted F1: {results['models']['xgboost']['weighted_f1']:.4f}  High-Risk Recall: {results['models']['xgboost']['high_risk_recall']:.4f}")

# --- 4d. LightGBM ---
print("  [4/5] LightGBM...")
try:
    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.1,
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1,
    )
    lgb_model.fit(X_train_imp, y_train_cls)
    lgb_pred = lgb_model.predict(X_test_imp)
    results["models"]["lightgbm"] = compute_metrics(y_test_cls, lgb_pred, "LightGBM")
    print(f"    Weighted F1: {results['models']['lightgbm']['weighted_f1']:.4f}  High-Risk Recall: {results['models']['lightgbm']['high_risk_recall']:.4f}")
except (ImportError, Exception) as e:
    print(f"    Native LightGBM unavailable ({type(e).__name__}: {str(e)[:50]}). Training HistGradientBoosting fallback...")
    from sklearn.ensemble import HistGradientBoostingClassifier
    hgb_model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=RANDOM_STATE, class_weight="balanced")
    hgb_model.fit(X_train_imp, y_train_cls)
    hgb_pred = hgb_model.predict(X_test_imp)
    results["models"]["lightgbm"] = compute_metrics(y_test_cls, hgb_pred, "LightGBM (HistGradientBoosting fallback)")
    print(f"    Weighted F1: {results['models']['lightgbm']['weighted_f1']:.4f}  High-Risk Recall: {results['models']['lightgbm']['high_risk_recall']:.4f}")

# --- 4e. CatBoost (Regressor, then classify) ---
print("  [5/5] CatBoost Regressor...")
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train_cb = train_ew[ALL_FEATURES].copy()
X_test_cb  = test_ew[ALL_FEATURES].copy()

cb_cat_idx = [ALL_FEATURES.index(c) for c in CAT_COLS if c in ALL_FEATURES]
cb_model = CatBoostRegressor(
    iterations=800, learning_rate=0.05, depth=8,
    random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False,
    cat_features=cb_cat_idx,
)
cb_model.fit(X_train_cb, y_train_reg)
cb_pred_reg = cb_model.predict(X_test_cb)

def classify_from_log10(series):
    return pd.Series(
        ["High" if v > HIGH_THRESHOLD else "Medium" if v > MED_THRESHOLD else "Low" for v in series],
        index=series.index if hasattr(series, "index") else None,
    )

cb_pred_cls = classify_from_log10(pd.Series(cb_pred_reg))
cb_metrics = compute_metrics(y_test_cls, cb_pred_cls, "CatBoost")
cb_metrics["regression"] = {
    "r2":   round(r2_score(y_test_reg, cb_pred_reg), 4),
    "mae":  round(mean_absolute_error(y_test_reg, cb_pred_reg), 4),
    "rmse": round(float(np.sqrt(mean_squared_error(y_test_reg, cb_pred_reg))), 4),
}
results["models"]["catboost"] = cb_metrics
print(f"    Weighted F1: {cb_metrics['weighted_f1']:.4f}  High-Risk Recall: {cb_metrics['high_risk_recall']:.4f}")
print(f"    Regression — R²: {cb_metrics['regression']['r2']:.4f}  MAE: {cb_metrics['regression']['mae']:.4f}  RMSE: {cb_metrics['regression']['rmse']:.4f}")

# Save model
MODEL_OUT = ROOT / "backend" / "model" / "catboost_risk_model.cbm"
cb_model.save_model(str(MODEL_OUT))
print(f"  CatBoost model saved → {MODEL_OUT}")

# Save updated features schema
feat_schema = {"features": ALL_FEATURES, "cat_features": CAT_COLS}
(ROOT / "backend" / "model" / "selected_features.json").write_text(json.dumps(feat_schema))
print(f"  Feature schema updated → {len(ALL_FEATURES)} features")

# ─────────────────────────────────────────────
# STEP 5 — SAVE COMPARISON TABLE
# ─────────────────────────────────────────────
print("\nSTEP 5: Saving results")
(RESULTS / "model_comparison.json").write_text(json.dumps(results, indent=2))

rows = []
for name, m in results["models"].items():
    if "error" not in m:
        reg = m.get("regression", {})
        rows.append({
            "Model": m.get("model", name),
            "Weighted_F1": m.get("weighted_f1"),
            "Macro_F1": m.get("macro_f1"),
            "High_Risk_Recall": m.get("high_risk_recall"),
            "High_Risk_Precision": m.get("high_risk_precision"),
            "Accuracy": m.get("accuracy"),
            "R2": reg.get("r2", "N/A"),
            "MAE": reg.get("mae", "N/A"),
            "RMSE": reg.get("rmse", "N/A"),
        })

pd.DataFrame(rows).to_csv(RESULTS / "model_comparison_table.csv", index=False)

# Best model by macro F1 among models with viable precision
valid_models = [
    (k, v) for k, v in results["models"].items()
    if "error" not in v and v.get("high_risk_precision", 0) > 0.5
]
best = max(valid_models, key=lambda x: x[1].get("macro_f1", 0)) if valid_models else ("catboost", results["models"]["catboost"])
(RESULTS / "best_model_info.json").write_text(json.dumps({"best_model": best[0], **best[1]}, indent=2))

print("  Results saved:")
print(f"    model_comparison.json")
print(f"    model_comparison_table.csv")
print(f"    best_model_info.json (best: {best[0]})")
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(pd.DataFrame(rows).to_string(index=False))
