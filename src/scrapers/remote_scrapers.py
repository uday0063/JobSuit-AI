import requests
import re
from src.scrapers.base import BaseScraper

class RemoteWWRScraper(BaseScraper):
    """
    Fetcher for We Work Remotely via their programming RSS/public list.
    """
    def scrape(self, query: str, location: str, max_results: int = 10) -> list[dict]:
        print(f"🚀  RemoteWWR: Searching for '{query}'...")
        url = "https://weworkremotely.com/categories/remote-programming-jobs.rss"
        try:
            r = requests.get(url, timeout=10)
            # Simple regex parser for XML/RSS to avoid extra dependencies
            items = re.findall(r'<item>(.*?)</item>', r.text, re.DOTALL)
            
            results = []
            for item in items[:max_results]:
                title = re.search(r'<title>(.*?)</title>', item).group(1)
                link = re.search(r'<link>(.*?)</link>', item).group(1)
                desc = re.search(r'<description>(.*?)</description>', item, re.DOTALL).group(1)
                
                # Filter by query locally since RSS returns all programming jobs
                if query.lower() in title.lower() or query.lower() in desc.lower():
                    results.append({
                        "title": title,
                        "company": "Remote Company",
                        "location": "Remote",
                        "description": re.sub(r'<[^>]+>', ' ', desc),
                        "url": link,
                        "source": "We Work Remotely"
                    })
            print(f"  ✔  RemoteWWR returned {len(results)} matches.")
            return results
        except Exception as e:
            print(f"  ❌  RemoteWWR Scraper Error: {str(e)}")
            return []

class RemoteOKScraper(BaseScraper):
    """
    Fetcher for Remote OK via their public API.
    """
    def scrape(self, query: str, location: str, max_results: int = 10) -> list[dict]:
        print(f"🚀  RemoteOK: Searching for '{query}'...")
        url = f"https://remoteok.com/api?tag={query.replace(' ', '-').lower()}"
        try:
            # Remote OK API requires a unique User-Agent
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            data = r.json()
            
            results = []
            # Skip the first item (it's the legal notice)
            for job in data[1:max_results+1]:
                results.append({
                    "title": job.get("position", ""),
                    "company": job.get("company", ""),
                    "location": "Remote",
                    "description": re.sub(r'<[^>]+>', ' ', job.get("description", "")),
                    "url": job.get("url", ""),
                    "source": "Remote OK"
                })
            print(f"  ✔  RemoteOK returned {len(results)} matches.")
            return results
        except Exception as e:
            print(f"  ❌  RemoteOK Scraper Error: {str(e)}")
            return []
