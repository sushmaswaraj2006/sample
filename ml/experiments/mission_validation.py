"""
ASTRA — Orbital-Regime Generalization Validation
(Complementary to Mission-Wise Cross-Validation)

SCIENTIFIC NOTE ON MISSION IDENTIFIERS:
In the public ESA Kelvins SDC-II dataset, spacecraft mission names and satellite IDs
were anonymized by ESA to protect sensitive commercial and governmental space assets.
When explicit 'mission_id' is absent, this experiment groups satellites by physical
orbital regimes based on semi-major axis (altitude) and orbital inclination.

This provides an honest out-of-distribution generalization test across:
1. LEO Sun-Synchronous Orbit (LEO-SSO: sma < 7200 km, inc > 80°)
2. LEO Mid/Low-Inclination Orbit (LEO-MidInc: sma < 7200 km, inc <= 80°)
3. High Orbit (High LEO / MEO / GEO: sma >= 7200 km)
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import f1_score, recall_score, precision_score, r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "backend" / "model" / "selected_features.json"
RESULTS_DIR = ROOT / "ml" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HIGH_THRESHOLD = -12.0
MED_THRESHOLD = -20.0


def run_mission_evaluation(data_csv: Path) -> Dict[str, Any]:
    """
    Performs mission-wise validation: groups events by target satellite altitude / inclination regime
    or mission ID if present, testing out-of-mission domain transfer.
    """
    if not data_csv.exists():
        raise FileNotFoundError(f"Data CSV not found: {data_csv}")

    print(f"Loading data from {data_csv}...")
    df = pd.read_csv(data_csv)

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    features = schema["features"]
    cat_cols = set(schema.get("cat_features", []))

    # Determine mission clustering field
    # In ESA SDC-II, if 'mission_id' is not present, we partition by orbital regime (LEO-SSO, LEO-Equatorial, MEO/GEO)
    if "mission_id" in df.columns:
        df["mission_group"] = df["mission_id"].astype(str)
    else:
        # Physical mission regimes based on semi-major axis (sma) and inclination (inc)
        sma = df["t_j2k_sma"]
        inc = df["t_j2k_inc"]
        conditions = [
            (sma < 7200) & (inc > 80),   # Polar / Sun-Synchronous LEO
            (sma < 7200) & (inc <= 80),  # Mid/Low-inclination LEO
            (sma >= 7200),               # High LEO / MEO / GEO
        ]
        choices = ["LEO_SunSync", "LEO_MidInc", "HighOrbit_Regime"]
        df["mission_group"] = np.select(conditions, choices, default="LEO_General")

    unique_missions = df["mission_group"].value_counts()
    print(f"Discovered mission regimes:\n{unique_missions}")

    # Add missingness flags
    if "covariance_missing" in features:
        df["covariance_missing"] = (df["t_ct_r"].isna() | df["c_ct_r"].isna()).astype(int)
    if "t_rcs_missing" in features:
        df["t_rcs_missing"] = df["t_rcs_estimate"].isna().astype(int)
    if "c_rcs_missing" in features:
        df["c_rcs_missing"] = df["c_rcs_estimate"].isna().astype(int)
    if "space_weather_missing" in features:
        df["space_weather_missing"] = df["F10"].isna().astype(int)
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN").astype(str)

    mission_results = {}
    
    # Run Leave-One-Mission-Group-Out or Train-on-Dominant, Test-on-Others
    for mission in unique_missions.index:
        if unique_missions[mission] < 100:
            continue
        print(f"\nEvaluating Holdout Mission Group: {mission}...")
        test_mask = df["mission_group"] == mission
        train_mask = ~test_mask

        # Ensure event-wise split integrity
        train_events = set(df.loc[train_mask, "event_id"].unique())
        test_events = set(df.loc[test_mask, "event_id"].unique())
        overlap = train_events & test_events
        if overlap:
            train_mask = train_mask & (~df["event_id"].isin(overlap))

        X_train = df.loc[train_mask, features]
        y_train = df.loc[train_mask, "risk"]
        X_test = df.loc[test_mask, features]
        y_test = df.loc[test_mask, "risk"]

        cat_idx = [features.index(c) for c in cat_cols if c in features]
        cb = CatBoostRegressor(
            iterations=400,
            learning_rate=0.08,
            depth=6,
            random_seed=42,
            verbose=0,
            cat_features=cat_idx,
            allow_writing_files=False
        )
        cb.fit(X_train, y_train)
        preds = cb.predict(X_test)

        # Classification metrics
        true_tiers = ["High" if r > HIGH_THRESHOLD else "Medium" if r > MED_THRESHOLD else "Low" for r in y_test]
        pred_tiers = ["High" if p > HIGH_THRESHOLD else "Medium" if p > MED_THRESHOLD else "Low" for p in preds]

        classes = ["Low", "Medium", "High"]
        wf1 = round(f1_score(true_tiers, pred_tiers, average="weighted", labels=classes, zero_division=0), 4)
        hr_rec = round(recall_score(true_tiers, pred_tiers, labels=["High"], average="macro", zero_division=0), 4)
        hr_prec = round(precision_score(true_tiers, pred_tiers, labels=["High"], average="macro", zero_division=0), 4)
        r2 = round(r2_score(y_test, preds), 4)
        mae = round(mean_absolute_error(y_test, preds), 4)

        mission_results[mission] = {
            "test_cdms": int(len(X_test)),
            "weighted_f1": wf1,
            "high_risk_recall": hr_rec,
            "high_risk_precision": hr_prec,
            "r2_score": r2,
            "mae": mae,
        }
        print(f"  {mission} -> W-F1: {wf1:.4f}, HR-Recall: {hr_rec:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")

    output_payload = {
        "validation_type": "orbital_regime_generalization",
        "description": "Cross-regime holdout evaluation across Sun-Synchronous LEO, Mid-Inclination LEO, and High-Orbit regimes",
        "regimes": mission_results
    }
    summary_file = RESULTS_DIR / "orbital_regime_validation_results.json"
    summary_file.write_text(json.dumps(output_payload, indent=2))
    # Keep backwards compatibility
    (RESULTS_DIR / "mission_validation_results.json").write_text(json.dumps(output_payload, indent=2))
    print(f"\nOrbital regime generalization results saved to {summary_file}")
    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=ROOT / "data" / "train_data.csv")
    args = parser.parse_args()
    res = run_mission_evaluation(Path(args.data))
    print(json.dumps(res, indent=2))
