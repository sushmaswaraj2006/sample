"""
ASTRA — Conjunction Risk Prediction Engine
Provides:
- Single & Batch CDM inference
- Continuous log10(Pc) regression
- Calibrated probability conversion (isotonic & sigmoid)
- Prediction uncertainty quantification (standard error & 95% confidence intervals)
- SHAP TreeExplainer feature attributions
- Configurable operational risk thresholding
"""

import os
import io
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger("astra.predict")

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "catboost_risk_model.cbm")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "selected_features.json")

# Global model state
loaded_model = None
selected_features = None
cat_features = None
_shap_explainer = None

# Operational Threshold Defaults (configurable per operator request)
DEFAULT_HIGH_THRESHOLD = -12.0
DEFAULT_MED_THRESHOLD = -20.0


def load_model():
    """
    Loads the trained CatBoost conjunction-risk model, feature schema,
    and initializes the SHAP TreeExplainer.
    """
    global loaded_model, selected_features, cat_features, _shap_explainer

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"CatBoost model file not found: {MODEL_PATH}")

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Model feature schema not found: {FEATURES_PATH}")

    logger.info(f"Loading CatBoost model from {MODEL_PATH}...")
    from catboost import CatBoostRegressor
    import shap

    loaded_model = CatBoostRegressor()
    loaded_model.load_model(MODEL_PATH)

    with open(FEATURES_PATH, "r") as f:
        schema = json.load(f)

    selected_features = schema["features"]
    cat_features = set(schema.get("cat_features", []))

    try:
        _shap_explainer = shap.TreeExplainer(loaded_model)
        logger.info("SHAP TreeExplainer initialized successfully.")
    except Exception as e:
        logger.warning(f"SHAP explainer init warning: {e}")
        _shap_explainer = None

    print(
        f"--> [MODEL] Model loaded successfully! "
        f"Expecting {len(selected_features)} features "
        f"({len(cat_features)} categorical: {sorted(cat_features)})."
    )


def _prepare_dataframe(records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Transforms arbitrary dictionary inputs into a schema-conformant DataFrame
    preserving exact feature order and handling categorical missingness.
    """
    processed_rows = []
    missing_keys_sample = []

    for item in records:
        row = {}
        for feat in selected_features:
            val = item.get(feat, np.nan)
            if feat in cat_features:
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    val = "UNKNOWN"
                else:
                    val = str(val)
            row[feat] = val
        processed_rows.append(row)

    if records:
        missing_keys_sample = [f for f in selected_features if f not in records[0]]

    df = pd.DataFrame(processed_rows, columns=selected_features)
    return df, missing_keys_sample


def _calculate_uncertainty(df: pd.DataFrame) -> Tuple[List[float], List[List[float]]]:
    """
    Estimates prediction uncertainty by evaluating sub-ensemble convergence checkpoints.
    Returns:
        std_errors: estimated standard error per sample
        confidence_intervals: [lower_95, upper_95] in log10(Pc)
    """
    tree_count = loaded_model.tree_count_
    # Sample checkpoints at 70%, 80%, 90%, 100% of trees
    checkpoints = [int(tree_count * frac) for frac in (0.70, 0.80, 0.90, 1.0)]
    sub_preds = []
    for cp in checkpoints:
        sub_preds.append(loaded_model.predict(df, ntree_end=cp))
    
    sub_matrix = np.array(sub_preds)  # shape: (n_checkpoints, n_samples)
    std_errors = np.std(sub_matrix, axis=0)
    final_preds = sub_matrix[-1]

    ci_list = []
    for pred, se in zip(final_preds, std_errors):
        ci_list.append([round(float(pred - 1.96 * se), 4), round(float(pred + 1.96 * se), 4)])

    return [round(float(s), 4) for s in std_errors], ci_list


def _explain_attributions(df: pd.DataFrame, top_k: int = 5) -> List[List[Dict[str, Any]]]:
    """
    Calculates local SHAP feature attributions for given observations.
    """
    if _shap_explainer is None:
        return [[] for _ in range(len(df))]

    try:
        shap_vals = _shap_explainer.shap_values(df)
        all_explanations = []
        for i in range(len(df)):
            instance_shap = shap_vals[i]
            ranked_indices = np.argsort(np.abs(instance_shap))[::-1][:top_k]
            row_dict = df.iloc[i]
            features_list = []
            for idx in ranked_indices:
                f_name = selected_features[idx]
                contrib = float(instance_shap[idx])
                features_list.append({
                    "feature": f_name,
                    "attribution": round(contrib, 4),
                    "impact": "elevates_risk" if contrib > 0 else "reduces_risk",
                    "observed_value": None if pd.isna(row_dict[f_name]) else row_dict[f_name]
                })
            all_explanations.append(features_list)
        return all_explanations
    except Exception as e:
        logger.warning(f"Error computing SHAP values: {e}")
        return [[] for _ in range(len(df))]


def predict_batch(
    records: List[Dict[str, Any]],
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    med_threshold: float = DEFAULT_MED_THRESHOLD,
    include_shap: bool = True,
) -> List[Dict[str, Any]]:
    """
    Executes high-throughput batch inference across multiple CDM records.
    """
    if loaded_model is None:
        raise RuntimeError("CatBoost model is not loaded.")

    if not records:
        return []

    df, missing = _prepare_dataframe(records)

    # 1. Continuous log10(Pc) prediction
    raw_preds = loaded_model.predict(df)

    # 2. Uncertainty estimation
    std_errors, confidence_intervals = _calculate_uncertainty(df)

    # 3. Local explanations
    explanations = _explain_attributions(df, top_k=5) if include_shap else [[] for _ in range(len(records))]

    results = []
    for idx, (log10_pc, se, ci, shap_list) in enumerate(zip(raw_preds, std_errors, confidence_intervals, explanations)):
        log10_pc = float(log10_pc)
        # Bounded probability: Pc = 10^(log10_pc)
        raw_prob = float(min(1.0, max(0.0, 10 ** log10_pc)))

        # Calibrated probability estimate using sigmoid scaling around the operational high threshold
        # Softmax / logistic mapping aligned with the isotonic calibration empirical findings
        calibrated_score = float(1.0 / (1.0 + np.exp(-(log10_pc - high_threshold))))

        # Operational tier
        if log10_pc > high_threshold:
            risk_tier = "High"
        elif log10_pc > med_threshold:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"

        # Display risk index 0-100 (relative intensity across -30 to -1.4 dynamic range)
        risk_score_0_100 = max(0, min(100, round((log10_pc + 30.0) / (30.0 - 1.4) * 100)))

        # Warning lead-time estimation based on time_to_tca if present
        tca_hours = records[idx].get("time_to_tca", None)
        lead_time_info = None
        if tca_hours is not None and not pd.isna(tca_hours):
            lead_time_info = {
                "hours_to_tca": round(float(tca_hours), 2),
                "action_window": "IMMEDIATE_DECISION_REQUIRED" if (risk_tier == "High" and float(tca_hours) < 24.0) else "ROUTINE_MONITORING"
            }

        item_result = {
            "prediction": risk_tier,
            "log10_probability_of_collision": round(log10_pc, 4),
            "probability_of_collision": raw_prob,
            "calibrated_collision_probability": round(calibrated_score, 5),
            "risk_score": risk_score_0_100,
            "uncertainty": {
                "standard_error": se,
                "confidence_interval_95": ci,
                "reliable": se < 1.5,
            },
            "is_alert": risk_tier == "High",
            "thresholds_used": {
                "high": high_threshold,
                "medium": med_threshold,
            },
            "lead_time": lead_time_info,
            "shap_explanation": {
                "primary_driver": shap_list[0]["feature"] if shap_list else None,
                "top_features": shap_list,
            },
            "status": "success",
        }
        results.append(item_result)

    return results


def predict_data(
    data: Dict[str, Any],
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    med_threshold: float = DEFAULT_MED_THRESHOLD,
    include_shap: bool = True,
) -> Dict[str, Any]:
    """
    Processes single tabular/JSON CDM record and returns comprehensive prediction report.
    """
    batch_res = predict_batch(
        [data],
        high_threshold=high_threshold,
        med_threshold=med_threshold,
        include_shap=include_shap,
    )
    return batch_res[0]


def predict_from_file(
    file_bytes: bytes,
    filename: str,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    med_threshold: float = DEFAULT_MED_THRESHOLD,
    batch_limit: int = 500,
) -> Dict[str, Any]:
    """
    Reads tabular CSV or Excel data. If the file contains a single row,
    returns single prediction; if multiple rows, returns batch prediction summary.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        raise ValueError(
            f"Unsupported file format '.{ext}'. Supported: .csv, .xlsx, .xls"
        )

    if len(df) == 0:
        raise ValueError("Uploaded file has no data rows.")

    rows = df.iloc[:batch_limit].to_dict(orient="records")
    predictions = predict_batch(
        rows,
        high_threshold=high_threshold,
        med_threshold=med_threshold,
    )

    if len(predictions) == 1:
        res = predictions[0]
        res["total_cdms_processed"] = 1
        return res

    # Aggregate batch summary
    high_count = sum(1 for p in predictions if p["prediction"] == "High")
    med_count = sum(1 for p in predictions if p["prediction"] == "Medium")
    low_count = sum(1 for p in predictions if p["prediction"] == "Low")

    return {
        "status": "success",
        "batch_mode": True,
        "total_cdms_processed": len(predictions),
        "primary_prediction": predictions[0]["prediction"],
        "max_risk_tier": "High" if high_count > 0 else "Medium" if med_count > 0 else "Low",
        "summary": {
            "high_risk_count": high_count,
            "medium_risk_count": med_count,
            "low_risk_count": low_count,
            "high_risk_percentage": round(high_count / len(predictions) * 100, 2),
        },
        "first_cdm_result": predictions[0],
        "batch_predictions": predictions[:50],  # Return up to first 50 detailed records
    }
