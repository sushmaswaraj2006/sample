"""
ASTRA — MongoDB Database Layer
Provides async connection and document helpers for events, predictions, and alerts.
"""

import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

# MongoDB is optional — the system degrades gracefully when not configured.
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB   = os.environ.get("MONGO_DB", "astra")

_client = None
_db     = None


def get_db():
    """
    Returns the MongoDB database handle.
    Lazily initialises the connection on first call.
    Returns None if pymongo is not installed or MongoDB is not reachable.
    """
    global _client, _db
    if _db is not None:
        return _db
    try:
        from pymongo import MongoClient
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        # Trigger a ping to verify the connection
        _client.admin.command("ping")
        _db = _client[MONGO_DB]
        print(f"--> [DB] MongoDB connected: {MONGO_URI}/{MONGO_DB}")
        _ensure_indexes()
        return _db
    except Exception as e:
        print(f"--> [DB] MongoDB not available ({e}). Running without persistence.")
        return None


def _ensure_indexes():
    """Create indexes on first connect for fast queries."""
    if _db is None:
        return
    try:
        _db["events"].create_index("event_id", unique=True)
        _db["events"].create_index("created_at")
        _db["predictions"].create_index("event_id")
        _db["predictions"].create_index("created_at")
        _db["alerts"].create_index("event_id")
        _db["alerts"].create_index([("priority", -1), ("created_at", -1)])
    except Exception as e:
        print(f"--> [DB] Index creation warning: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# EVENTS
# ─────────────────────────────────────────────────────────────────────────────
def save_event(event_id: str, cdm_data: Dict[str, Any]) -> bool:
    db = get_db()
    if db is None:
        return False
    try:
        db["events"].update_one(
            {"event_id": event_id},
            {"$set": {**cdm_data, "event_id": event_id, "updated_at": _now()}},
            upsert=True,
        )
        return True
    except Exception as e:
        print(f"--> [DB] save_event error: {e}")
        return False


def get_events(limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
    db = get_db()
    if db is None:
        return []
    try:
        cursor = db["events"].find(
            {}, {"_id": 0}
        ).sort("created_at", -1).skip(skip).limit(limit)
        return list(cursor)
    except Exception as e:
        print(f"--> [DB] get_events error: {e}")
        return []


def get_event(event_id: str) -> Optional[Dict[str, Any]]:
    db = get_db()
    if db is None:
        return None
    try:
        return db["events"].find_one({"event_id": event_id}, {"_id": 0})
    except Exception as e:
        print(f"--> [DB] get_event error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
def save_prediction(
    event_id: str,
    filename: str,
    result: Dict[str, Any],
    username: Optional[str] = None,
) -> bool:
    db = get_db()
    if db is None:
        return False
    try:
        doc = {
            "event_id": event_id,
            "filename": filename,
            "result": result,
            "username": username,
            "created_at": _now(),
        }
        db["predictions"].insert_one(doc)
        # Auto-generate alert if High risk
        if result.get("prediction") == "High":
            _create_alert(event_id, result)
        return True
    except Exception as e:
        print(f"--> [DB] save_prediction error: {e}")
        return False


def get_predictions(limit: int = 100) -> List[Dict[str, Any]]:
    db = get_db()
    if db is None:
        return []
    try:
        cursor = db["predictions"].find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
        return list(cursor)
    except Exception as e:
        print(f"--> [DB] get_predictions error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────────────────────────────────────
def _create_alert(event_id: str, prediction: Dict[str, Any]):
    db = get_db()
    if db is None:
        return
    try:
        risk_level = prediction.get("prediction", "Low")
        priority = {"High": "high", "Medium": "medium", "Low": "low"}.get(risk_level, "low")
        alert = {
            "event_id": event_id,
            "title": f"{risk_level} future-risk conjunction detected",
            "priority": priority,
            "log10_pc": prediction.get("log10_probability_of_collision"),
            "risk_score": prediction.get("risk_score"),
            "status": "New",
            "is_early_warning": prediction.get("is_early_warning", False),
            "created_at": _now(),
            "acknowledged_at": None,
            "acknowledged_by": None,
        }
        db["alerts"].insert_one(alert)
    except Exception as e:
        print(f"--> [DB] _create_alert error: {e}")


def get_alerts(limit: int = 50, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    db = get_db()
    if db is None:
        return []
    try:
        query = {}
        if status_filter:
            query["status"] = status_filter
        cursor = (
            db["alerts"]
            .find(query, {"_id": 0})
            .sort([("priority", -1), ("created_at", -1)])
            .limit(limit)
        )
        return list(cursor)
    except Exception as e:
        print(f"--> [DB] get_alerts error: {e}")
        return []


def acknowledge_alert(event_id: str, username: str) -> bool:
    db = get_db()
    if db is None:
        return False
    try:
        result = db["alerts"].update_one(
            {"event_id": event_id, "status": {"$ne": "Acknowledged"}},
            {"$set": {"status": "Acknowledged", "acknowledged_at": _now(), "acknowledged_by": username}},
        )
        return result.modified_count > 0
    except Exception as e:
        print(f"--> [DB] acknowledge_alert error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def get_summary_stats() -> Dict[str, Any]:
    db = get_db()
    if db is None:
        return {"available": False}
    try:
        total_predictions = db["predictions"].count_documents({})
        high_risk = db["predictions"].count_documents({"result.prediction": "High"})
        medium_risk = db["predictions"].count_documents({"result.prediction": "Medium"})
        low_risk = db["predictions"].count_documents({"result.prediction": "Low"})
        active_alerts = db["alerts"].count_documents({"status": {"$ne": "Acknowledged"}})
        return {
            "available": True,
            "total_predictions": total_predictions,
            "high_risk": high_risk,
            "medium_risk": medium_risk,
            "low_risk": low_risk,
            "active_alerts": active_alerts,
        }
    except Exception as e:
        print(f"--> [DB] get_summary_stats error: {e}")
        return {"available": False}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
