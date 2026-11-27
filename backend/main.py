"""
ASTRA — Production FastAPI Backend Server
Provides:
- Single & Batch CDM Inference endpoints
- Explanations (SHAP) & Uncertainty Quantification
- Configurable Operational Risk Thresholds
- JWT Authentication (Bearer Token)
- Conjunction Event & Prediction Storage
- Live Health & Telemetry Metrics
"""

import time
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, Dict, List, Optional

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Depends,
    Request,
    status,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

import predict
from auth import (
    Token,
    authenticate_user,
    create_access_token,
    get_current_user,
    require_auth,
    require_admin,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from cache import cache_get, cache_set, cache_delete, cache_key
from database import (
    save_prediction,
    get_predictions,
    get_events,
    get_event,
    get_alerts,
    acknowledge_alert,
    get_summary_stats,
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("astra.api")


# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--> Initializing ASTRA Satellite Conjunction Backend Server...")
    predict.load_model()
    # Try non-blocking database and cache initializations
    try:
        from database import get_db
        get_db()
    except Exception as e:
        logger.warning(f"Database ping warning: {e}")
    try:
        from cache import _get_client
        _get_client()
    except Exception as e:
        logger.warning(f"Cache ping warning: {e}")
    logger.info("--> ASTRA Backend initialization complete.")
    yield
    logger.info("--> ASTRA Backend shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# APP INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ASTRA Satellite Conjunction API",
    description=(
        "Production-grade ML early-warning and conjunction risk assessment platform. "
        "Predicts continuous log10(Pc), calibrated collision probability, and SHAP explainability."
    ),
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing & tracing middleware
@app.middleware("http")
async def add_process_time_and_logging(request: Request, call_next):
    req_id = uuid.uuid4().hex[:8]
    start_time = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
    logger.info(f"[{req_id}] {request.method} {request.url.path} -> {response.status_code} ({elapsed_ms}ms)")
    return response


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
# Schema for JSON & Form login
class LoginRequest(BaseModel):
    username: str
    password: str


class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    event_id: Optional[str] = None
    filename: Optional[str] = None
    high_threshold: Optional[float] = Field(default=-12.0, description="Custom high-risk threshold in log10(Pc)")
    med_threshold: Optional[float] = Field(default=-20.0, description="Custom medium-risk threshold in log10(Pc)")
    include_shap: Optional[bool] = Field(default=True, description="Compute local SHAP feature attributions")


class BatchPredictionRequest(BaseModel):
    items: List[Dict[str, Any]]
    high_threshold: Optional[float] = -12.0
    med_threshold: Optional[float] = -20.0
    include_shap: Optional[bool] = False


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH & METRICS ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check():
    """Service health and model readiness probe."""
    from database import get_db
    from cache import _get_client
    return {
        "project": "ASTRA SDC-II",
        "version": "2.1.0",
        "status": "Online",
        "model_loaded": predict.loaded_model is not None,
        "features_loaded": len(predict.selected_features or []),
        "mode": "Live Clean CatBoost (102 Features)" if predict.loaded_model is not None else "Model Not Loaded",
        "mongodb": "connected" if get_db() is not None else "degraded_offline",
        "redis": "connected" if _get_client() is not None else "degraded_offline",
    }


# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/auth/login", response_model=Token, tags=["Authentication"])
async def login_for_access_token(payload: LoginRequest):
    """Authenticates credentials and returns a signed JWT access token."""
    user = authenticate_user(payload.username, payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        username=user["username"],
        role=user["role"],
    )


@app.get("/auth/me", tags=["Authentication"])
async def read_users_me(current_user=Depends(require_auth)):
    """Returns current active authenticated user identity."""
    return {
        "username": current_user.username,
        "full_name": current_user.full_name,
        "role": current_user.role,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict/json", tags=["Prediction"])
async def predict_json(
    payload: PredictionRequest,
    current_user=Depends(get_current_user),
):
    """
    Predicts conjunction risk from a JSON tabular CDM payload.
    Returns: continuous log10(Pc), calibrated Pc, uncertainty interval, and SHAP drivers.
    """
    try:
        result = predict.predict_data(
            payload.features,
            high_threshold=payload.high_threshold or -12.0,
            med_threshold=payload.med_threshold or -20.0,
            include_shap=payload.include_shap if payload.include_shap is not None else True,
        )
        event_id = payload.event_id or f"CONJ-{uuid.uuid4().hex[:6].upper()}"
        result["event_id"] = event_id

        # Async/Background DB save
        save_prediction(
            event_id=event_id,
            filename=payload.filename or "json_stream",
            result=result,
            username=current_user.username if current_user else "anonymous",
        )
        return result
    except Exception as e:
        logger.error(f"Inference error on /predict/json: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict/tabular", tags=["Prediction"])
async def predict_tabular(
    request: Request,
    high_threshold: float = Query(default=-12.0),
    med_threshold: float = Query(default=-20.0),
    current_user=Depends(get_current_user),
):
    """
    Ingests tabular CSV or Excel CDMs.
    Processes multipart file upload natively with zero dependency on python-multipart.
    """
    content_type = request.headers.get("content-type", "")
    body = await request.body()

    filename = "uploaded_cdm.csv"
    file_bytes = b""

    if "multipart/form-data" in content_type:
        import re
        m = re.search(r'boundary=([^;]+)', content_type)
        if m:
            boundary = m.group(1).strip('"').encode()
            parts = body.split(b"--" + boundary)
            for part in parts:
                if b'filename=' in part:
                    header_end = part.find(b"\r\n\r\n")
                    offset = 4
                    if header_end == -1:
                        header_end = part.find(b"\n\n")
                        offset = 2
                    headers = part[:header_end].decode(errors="ignore")
                    fn_match = re.search(r'filename="([^"]+)"', headers)
                    if fn_match:
                        filename = fn_match.group(1)
                    file_bytes = part[header_end + offset:].rstrip(b"\r\n--")
                    break
        if not file_bytes:
            file_bytes = body
    else:
        # Direct raw binary stream upload
        file_bytes = body
        raw_fn = request.headers.get("x-filename")
        if raw_fn:
            filename = raw_fn

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predict.predict_from_file(
            file_bytes,
            filename,
            high_threshold=high_threshold,
            med_threshold=med_threshold,
        )
        event_id = f"FILE-{uuid.uuid4().hex[:6].upper()}"
        result["event_id"] = event_id

        save_prediction(
            event_id=event_id,
            filename=filename,
            result=result if not result.get("batch_mode") else result.get("first_cdm_result", result),
            username=current_user.username if current_user else "anonymous",
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error on tabular upload {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch_endpoint(
    payload: BatchPredictionRequest,
    current_user=Depends(get_current_user),
):
    """
    High-performance batch prediction endpoint for multiple CDMs.
    """
    try:
        predictions = predict.predict_batch(
            payload.items,
            high_threshold=payload.high_threshold or -12.0,
            med_threshold=payload.med_threshold or -20.0,
            include_shap=payload.include_shap or False,
        )
        return {
            "total_processed": len(predictions),
            "predictions": predictions,
        }
    except Exception as e:
        logger.error(f"Error on /predict/batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY & MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """
    Returns verified model architecture, features, and validated benchmarks.
    """
    ck = cache_key("model", "info")
    cached = cache_get(ck)
    if cached:
        return cached

    info = {
        "model_type": "CatBoostRegressor",
        "leakage_free": True,
        "feature_count": len(predict.selected_features or []),
        "features": predict.selected_features or [],
        "categorical_features": list(predict.cat_features or []),
        "target": "log10(Pc)",
        "default_thresholds": {
            "High": "log10(Pc) > -12",
            "Medium": "-20 < log10(Pc) <= -12",
            "Low": "log10(Pc) <= -20",
        },
        "verified_event_wise_metrics": {
            "r2": 0.8868,
            "mae": 2.2261,
            "rmse": 3.3790,
            "high_risk_precision": 0.9330,
            "high_risk_recall": 0.7966,
            "weighted_f1": 0.8608,
            "validation_split": "Event-Wise 80/20 (Zero Event Leakage)",
        },
        "probability_calibration": {
            "method": "Isotonic Regression",
            "expected_calibration_error": 0.0882,
            "brier_score": 0.0013,
            "status": "Well-Calibrated",
        },
    }
    cache_set(ck, info, ttl=300)
    return info


@app.get("/model/explain/global", tags=["Model"])
async def get_global_explanations(top_k: int = 15):
    """
    Returns top global features ranked by mean |SHAP| impact on collision risk.
    """
    from experiments.explainability_shap import compute_global_feature_importance
    try:
        import sys
        sys.path.insert(0, "ml")
        results = compute_global_feature_importance()
        return {"top_features": results[:top_k]}
    except Exception as e:
        # Fallback to model native feature importances
        if predict.loaded_model is not None:
            fi = predict.loaded_model.get_feature_importance()
            feats = predict.selected_features
            sorted_idx = fi.argsort()[::-1][:top_k]
            return {
                "top_features": [
                    {"feature": feats[i], "importance_score": round(float(fi[i]), 4), "rank": r + 1}
                    for r, i in enumerate(sorted_idx)
                ]
            }
        return {"top_features": []}


# ─────────────────────────────────────────────────────────────────────────────
# EVENT & ALERT MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/events", tags=["Events"])
async def list_conjunction_events(limit: int = 50, skip: int = 0):
    events = get_events(limit=limit, skip=skip)
    return {"events": events, "count": len(events)}


@app.get("/events/{event_id}", tags=["Events"])
async def get_single_event(event_id: str):
    ev = get_event(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found.")
    return ev


@app.get("/alerts", tags=["Alerts"])
async def get_alerts_list(limit: int = 50, status: Optional[str] = None):
    alerts = get_alerts(limit=limit, status_filter=status)
    return {"alerts": alerts, "count": len(alerts)}


@app.patch("/alerts/{event_id}/acknowledge", tags=["Alerts"])
async def acknowledge_event_alert(event_id: str, current_user=Depends(require_auth)):
    success = acknowledge_alert(event_id, current_user.username)
    if not success:
        raise HTTPException(status_code=404, detail="Alert already acknowledged or not found.")
    cache_delete(cache_key("alerts"))
    return {"success": True, "event_id": event_id, "acknowledged_by": current_user.username}


@app.get("/analytics/summary", tags=["Analytics"])
async def get_dashboard_summary():
    """Summary statistics for mission operations dashboard."""
    stats = get_summary_stats()
    return stats
