"""
ASTRA — Explainability & Feature Importance Module (SHAP)
Computes global and local SHAP feature importances using CatBoost's TreeExplainer.
Exports summary plots, importance tables, and local feature attribution dictionaries.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import shap
import catboost

logger = logging.getLogger("astra.shap")

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "backend" / "model" / "catboost_risk_model.cbm"
SCHEMA_PATH = ROOT / "backend" / "model" / "selected_features.json"
RESULTS_DIR = ROOT / "ml" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_explainer = None
_model = None
_features = None
_cat_features = None


def get_explainer():
    """
    Lazily loads the CatBoost model and initializes the TreeExplainer.
    Cached for high-throughput local explanation queries.
    """
    global _explainer, _model, _features, _cat_features
    if _explainer is not None:
        return _explainer, _features, _cat_features

    if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Model or schema missing: {MODEL_PATH}, {SCHEMA_PATH}")

    _model = catboost.CatBoostRegressor()
    _model.load_model(str(MODEL_PATH))

    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    _features = schema["features"]
    _cat_features = set(schema.get("cat_features", []))

    _explainer = shap.TreeExplainer(_model)
    return _explainer, _features, _cat_features


def explain_single_instance(data_dict: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    """
    Computes local SHAP attributions for a single CDM feature dictionary.
    Returns the top-K driving features and their positive/negative impact on risk.
    """
    explainer, features, cat_feats = get_explainer()

    row = {}
    for feat in features:
        val = data_dict.get(feat, np.nan)
        if feat in cat_feats:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                val = "UNKNOWN"
            else:
                val = str(val)
        row[feat] = val

    df = pd.DataFrame([row], columns=features)
    shap_vals = explainer.shap_values(df)[0]
    base_val = float(explainer.expected_value)

    # Sort by absolute contribution
    ranked_indices = np.argsort(np.abs(shap_vals))[::-1]

    top_features = []
    for idx in ranked_indices[:top_k]:
        f_name = features[idx]
        val = row[f_name]
        contrib = float(shap_vals[idx])
        top_features.append({
            "feature": f_name,
            "value": None if pd.isna(val) else val,
            "shap_attribution": round(contrib, 4),
            "effect": "increases_risk" if contrib > 0 else "decreases_risk",
            "direction": "+" if contrib > 0 else "-",
        })

    return {
        "base_value_log10_pc": round(base_val, 4),
        "top_contributing_features": top_features,
        "primary_risk_driver": top_features[0]["feature"] if top_features else None,
        "explanation_summary": (
            f"Risk primarily driven by {top_features[0]['feature']} "
            f"({'+' if top_features[0]['shap_attribution'] > 0 else ''}"
            f"{top_features[0]['shap_attribution']} log10(Pc))."
            if top_features else "No explanation available."
        ),
    }


def compute_global_feature_importance(
    sample_df: Optional[pd.DataFrame] = None,
    n_samples: int = 1000,
    save_outputs: bool = True
) -> List[Dict[str, Any]]:
    """
    Computes global mean |SHAP| feature importance across representative background data.
    """
    explainer, features, cat_feats = get_explainer()

    if sample_df is None:
        # Load sample from test data if present, otherwise fallback to model feature importances
        test_path = ROOT / "data" / "test_data.csv"
        if test_path.exists():
            sample_df = pd.read_csv(test_path, nrows=n_samples)
        else:
            # Fallback to CatBoost's native feature importances
            native_fi = _model.get_feature_importance()
            ranked = np.argsort(native_fi)[::-1]
            return [
                {"feature": features[i], "importance_score": round(float(native_fi[i]), 4), "rank": rank + 1}
                for rank, i in enumerate(ranked)
            ]

    # Process features in sample_df
    X = sample_df.reindex(columns=features)
    for c in cat_feats:
        if c in X.columns:
            X[c] = X[c].fillna("UNKNOWN").astype(str)

    shap_matrix = explainer.shap_values(X)
    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    ranked = np.argsort(mean_abs)[::-1]

    results = []
    for rank, idx in enumerate(ranked):
        results.append({
            "feature": features[idx],
            "mean_abs_shap": round(float(mean_abs[idx]), 5),
            "rank": rank + 1
        })

    if save_outputs:
        out_path = RESULTS_DIR / "shap_feature_importance.json"
        out_path.write_text(json.dumps(results[:30], indent=2))
        logger.info(f"Saved top 30 SHAP features to {out_path}")

    return results


if __name__ == "__main__":
    print("Testing local SHAP explanation on dummy baseline observation...")
    sample_cdm = {
        "time_to_tca": 2.5,
        "miss_distance": 182.4,
        "relative_speed": 14200.0,
        "mahalanobis_distance": 1.45,
        "c_sigma_t": 120.0,
        "c_object_type": "DEBRIS"
    }
    explanation = explain_single_instance(sample_cdm)
    print(json.dumps(explanation, indent=2))
