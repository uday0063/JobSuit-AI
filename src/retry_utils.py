"""
Retry decorator with exponential backoff.
"""
import time
import functools
from src.logger import get_logger

log = get_logger("retry")


def retry(max_attempts: int = 3, base_delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator that retries a function on exception.

    Args:
        max_attempts: Maximum number of tries (including the first).
        base_delay:   Initial delay in seconds before first retry.
        backoff:      Multiplier applied to delay after each failure.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        log.warning(
                            "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                            fn.__name__, attempt, max_attempts, exc, delay,
                        )
                        time.sleep(delay)
                        delay *= backoff
                    else:
                        log.error(
                            "%s failed after %d attempts: %s",
                            fn.__name__, max_attempts, exc,
                        )
            raise last_exc
        return wrapper
    return decorator
