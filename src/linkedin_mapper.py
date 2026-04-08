"""
=============================================================
  src/linkedin_mapper.py — Schema Mapping Logic
=============================================================
  Translates the raw JSON output from LinkedIn/Apify into 
  the specific 13-feature schema your XGBoost model expects.
"""

import urllib.parse

def map_linkedin_to_schema(raw_job: dict) -> dict:
    """
    Field-by-field mapping to the "Final Boss" model schema.

    Args:
        raw_job  : Dict from Apify linkedin scraper

    Returns:
        Structured dict for the fake job detector
    """
    location = raw_job.get("location", "").strip().lower()

    company = raw_job.get("company", "") or raw_job.get("companyName", "Confidential")
    if str(company).lower() in ['nan', 'none', 'undefined', '']:
        company = "Confidential Employer"

    raw_url = raw_job.get("applyUrl", "") or raw_job.get("url", "") or raw_job.get("job_url", "")
    title = raw_job.get("title", "Job")

    if not raw_url or str(raw_url).lower() in ['nan', 'none', 'undefined', '']:
        encoded_title = urllib.parse.quote_plus(str(title))
        encoded_loc = urllib.parse.quote_plus(str(raw_job.get("location", "India")))
        final_url = f"https://www.linkedin.com/jobs/search/?keywords={encoded_title}&location={encoded_loc}"
    else:
        final_url = raw_url

    # 🏁 Universal Mapping (handles multiple actor schemas)
    mapped = {
        "title":           title,
        # Look for both 'company' and 'companyName'
        "company_name":    company,
        "location":        raw_job.get("location", ""),
        "description":     raw_job.get("description", ""),
        "requirements":    raw_job.get("description", ""),
        "benefits":        "",
        "salary_range":    raw_job.get("salary", ""),
        "employment_type": raw_job.get("employmentType", "Full-time"),

        # Binary flags
        "telecommuting":   1 if "remote" in location else 0,
        "has_company_logo": 1 if (raw_job.get("companyLogo") or raw_job.get("companyLogoUrl")) else 0,

        "apply_url":       final_url,

        # Preserve metadata for UI
        "company":         company,
        "url":             final_url,
        "source":          raw_job.get("source", "Unknown"),
    }

    return mapped
