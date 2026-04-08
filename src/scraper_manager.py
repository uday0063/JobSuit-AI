"""
Scraper Manager — orchestrates parallel scraping with retry + caching.
Used by the web API as a higher-level wrapper around scraper.py.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.scraper import scrape_linkedin_jobs, generate_static_fallback
from src.retry_utils import retry
from src import cache
from src.config import THREAD_POOL_SIZE, SCRAPE_TIMEOUT, MAX_JOBS_PER_SOURCE
from src.logger import get_logger

log = get_logger("scraper_manager")


@retry(max_attempts=2, base_delay=2.0)
def _run_scrape(api_token, query, location, max_jobs, sources, chat_client, profile):
    """Single scrape attempt (retried on failure)."""
    return scrape_linkedin_jobs(
        api_token=api_token,
        search_query=query,
        location=location,
        max_jobs=max_jobs,
        sources=sources,
        chat_client=chat_client,
        profile=profile,
    )


def discover_jobs(
    api_token: str,
    query: str,
    location: str,
    sources: list,
    chat_client=None,
    profile=None,
    max_jobs: int = MAX_JOBS_PER_SOURCE,
    use_cache: bool = True,
) -> list[dict]:
    """
    High-level entry point for job discovery.
    Checks cache first, then scrapes with retry, then falls back.
    """
    # 1. Cache check
    if use_cache:
        cached = cache.get(query, location, sources)
        if cached:
            return cached

    # 2. Scrape with retry
    try:
        results = _run_scrape(
            api_token, query, location, max_jobs, sources, chat_client, profile
        )
    except Exception as exc:
        log.error("All scrape attempts failed: %s — using static fallback", exc)
        results = generate_static_fallback(query, location, profile)

    # 3. Cache results
    if results and use_cache:
        cache.put(query, location, sources, results)

    log.info("discover_jobs returning %d results for '%s'", len(results), query)
    return results
