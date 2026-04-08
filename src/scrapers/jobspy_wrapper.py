import pandas as pd
import time
from jobspy import scrape_jobs
from src.scrapers.base import BaseScraper

class JobSpyScraper(BaseScraper):
    """
    Optimized JobSpy wrapper with retry logic for various job boards.
    """
    def __init__(self, sites=["indeed", "linkedin", "glassdoor"]):
        self.sites = sites

    def scrape(self, query: str, location: str, max_results: int = 20) -> list[dict]:
        retries = 2
        for attempt in range(retries + 1):
            try:
                print(f"🚀 [JobSpy] Attempt {attempt+1}: Searching {self.sites} for '{query}'...")
                
                # Broaden query internally for Fresher/Intern exposure if needed
                search_term = query
                if attempt > 0:
                    search_term = f"{query} OR Intern OR Fresher OR Entry Level"

                # Dynamic country detection for Indeed/Glassdoor
                target_country = 'india' if 'india' in location.lower() or 'delhi' in location.lower() or 'mumbai' in location.lower() else None

                kwargs = {
                    "site_name": self.sites,
                    "search_term": search_term,
                    "location": location,
                    "distance": 15,
                    "results_wanted": max_results,
                    "hours_old": 336,
                    "proxy": None
                }
                if target_country:
                    kwargs["country_indeed"] = target_country

                jobs_df = scrape_jobs(**kwargs)
                
                if jobs_df.empty:
                    print(f"  ⚠️ [JobSpy] {self.sites} returned 0 results.")
                    if attempt < retries: continue
                    return []

                # 🛠️ Standardized Output Mapping
                results = []
                for _, row in jobs_df.iterrows():
                    # Extreme fallback for URL
                    job_url = row.get("job_url") or row.get("url") or row.get("link", "")
                    job_url = str(job_url).strip()
                    if job_url.lower() in ['nan', 'none', 'undefined', '']:
                        job_url = ""
                    
                    # Fix relative/malformed URLs
                    if job_url and not job_url.startswith(('http://', 'https://')):
                        job_url = "https://" + job_url
                    
                    # Prevent 'undefined' in other fields
                    def get_val(key, fallback):
                        val = str(row.get(key, fallback)).strip()
                        return val if val.lower() not in ['nan', 'none', 'undefined', ''] else fallback

                    results.append({
                        "title": get_val("title", "Discovery Role"),
                        "company": get_val("company", "Strategic Organization"),
                        "location": get_val("location", location),
                        "description": get_val("description", "No description provided."),
                        "url": job_url,
                        "source": get_val("site", "Job Boards")
                    })
                
                print(f"  ✅ [JobSpy] {self.sites} success: Found {len(results)} jobs.")
                return results

            except Exception as e:
                print(f"  ❌ [JobSpy] Error (Attempt {attempt+1}): {str(e)}")
                if attempt < retries:
                    time.sleep(2) # Brief cooling period
                    continue
        
        return [] # Always return list, never None
