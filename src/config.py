"""
Centralized configuration constants for the pipeline.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")

# --- Scraping ---
DEFAULT_SOURCES = ["linkedin", "indeed", "glassdoor", "naukri", "internshala", "wellfound", "yc", "remoteok", "wwr", "jobspresso", "remotive"]
MAX_JOBS_PER_SOURCE = 24
SCRAPE_TIMEOUT = 40
THREAD_POOL_SIZE = 5

# --- Pipeline ---
FAKE_THRESHOLD = 0.3
MATCH_THRESHOLD = 0.6
MAX_APPLICATIONS_PER_RUN = 15
MODEL_DIR = "models"

# --- Cache ---
CACHE_TTL_SECONDS = 900  # 15 minutes
CACHE_FILE = "cache/job_cache.json"

# --- Logging ---
LOG_DIR = "logs"
LOG_FILE = "logs/pipeline.log"
LOG_LEVEL = "INFO"

# --- Strictness Mapping ---
STRICTNESS_MAP = {
    0: 0.8,   # Loose
    1: 0.5,   # Balanced
    2: 0.3,   # Strict
}
