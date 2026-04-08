"""
Simple TTL-based job result cache.
In-memory dict with optional JSON file persistence.
"""
import os
import json
import time
import hashlib
from src.config import CACHE_TTL_SECONDS, CACHE_FILE
from src.logger import get_logger

log = get_logger("cache")

_cache: dict = {}


def _make_key(query: str, location: str, sources: list) -> str:
    raw = f"{query.lower().strip()}|{location.lower().strip()}|{','.join(sorted(sources))}"
    return hashlib.md5(raw.encode()).hexdigest()


def get(query: str, location: str, sources: list) -> list | None:
    """Return cached results if fresh, else None."""
    key = _make_key(query, location, sources)
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL_SECONDS:
        log.info("Cache HIT for '%s' in '%s'", query, location)
        return entry["data"]
    return None


def put(query: str, location: str, sources: list, data: list):
    """Store results in cache."""
    key = _make_key(query, location, sources)
    _cache[key] = {"ts": time.time(), "data": data}
    log.info("Cached %d jobs for '%s' in '%s'", len(data), query, location)
    _persist()


def clear():
    """Flush the entire cache."""
    _cache.clear()
    log.info("Cache cleared")


def _persist():
    """Best-effort save to disk."""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_cache, f, default=str)
    except Exception as exc:
        log.warning("Cache persist failed: %s", exc)


def load_from_disk():
    """Load cache from disk on startup."""
    global _cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _cache = json.load(f)
            log.info("Loaded %d cache entries from disk", len(_cache))
    except Exception as exc:
        log.warning("Cache load failed: %s", exc)
        _cache = {}
