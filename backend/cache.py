"""
ASTRA — Redis Cache Layer
Provides simple get/set/delete wrappers with JSON serialisation.
Degrades gracefully when Redis is not available.
"""

import json
import os
from typing import Any, Optional

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "300"))  # 5 minutes default

_redis_client = None


def _get_client():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
        client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        _redis_client = client
        print(f"--> [CACHE] Redis connected: {REDIS_URL}")
        return _redis_client
    except Exception as e:
        print(f"--> [CACHE] Redis not available ({e}). Running without cache.")
        return None


def cache_get(key: str) -> Optional[Any]:
    """Retrieve a cached value. Returns None on miss or error."""
    client = _get_client()
    if client is None:
        return None
    try:
        raw = client.get(key)
        return json.loads(raw) if raw is not None else None
    except Exception:
        return None


def cache_set(key: str, value: Any, ttl: int = CACHE_TTL_SECONDS) -> bool:
    """Store a value in cache. Returns False on error."""
    client = _get_client()
    if client is None:
        return False
    try:
        client.setex(key, ttl, json.dumps(value, default=str))
        return True
    except Exception:
        return False


def cache_delete(key: str) -> bool:
    """Remove a key from cache."""
    client = _get_client()
    if client is None:
        return False
    try:
        client.delete(key)
        return True
    except Exception:
        return False


def cache_key(*parts: str) -> str:
    """Build a namespaced cache key."""
    return "astra:" + ":".join(str(p) for p in parts)


def invalidate_prefix(prefix: str) -> int:
    """Delete all keys matching a prefix pattern. Returns count deleted."""
    client = _get_client()
    if client is None:
        return 0
    try:
        keys = client.keys(f"astra:{prefix}*")
        if keys:
            return client.delete(*keys)
        return 0
    except Exception:
        return 0
