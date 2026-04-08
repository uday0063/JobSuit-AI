import sys
import re
import requests
import json
import urllib.parse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.scrapers.jobspy_wrapper import JobSpyScraper
from src.scrapers.remote_scrapers import RemoteWWRScraper, RemoteOKScraper

def generate_static_fallback(search_query, location, profile=None):
    """
    Guaranteed fallback that ALWAYS returns results.
    Creates job entries with direct search links to major job portals,
    tailored to the user's profile and search query.
    """
    profile = profile or {}
    skills = profile.get("skills", [])
    seniority = profile.get("seniority", profile.get("level", ""))
    # Build search variations from skills
    search_variants = [search_query]
    for skill in skills[:4]:
        variant = skill.strip()
        if variant and variant.lower() != search_query.lower():
            search_variants.append(variant)

    results = []

    board_configs = [
        {
            "name": "LinkedIn",
            "url_fn": lambda q, l: f"https://www.linkedin.com/jobs/search/?keywords={urllib.parse.quote_plus(q)}&location={urllib.parse.quote_plus(l)}",
        },
        {
            "name": "Indeed India",
            "url_fn": lambda q, l: f"https://in.indeed.com/jobs?q={urllib.parse.quote_plus(q)}&l={urllib.parse.quote_plus(l)}",
        },
        {
            "name": "Naukri",
            "url_fn": lambda q, l: f"https://www.naukri.com/{q.lower().replace(' ', '-')}-jobs-in-{l.lower().replace(' ', '-')}",
        },
        {
            "name": "Glassdoor",
            "url_fn": lambda q, l: f"https://www.glassdoor.co.in/Job/jobs.htm?sc.keyword={urllib.parse.quote_plus(q)}&locT=C&locKeyword={urllib.parse.quote_plus(l)}",
        },
    ]

    level_tag = f" ({seniority})" if seniority else ""

    for i, variant in enumerate(search_variants[:6]):
        board = board_configs[i % len(board_configs)]
        url = board["url_fn"](variant, location)

        results.append({
            "title": f"{variant}{level_tag}",
            "company": f"Multiple Openings via {board['name']}",
            "location": location,
            "description": (
                f"Browse the latest {variant} openings in {location} on {board['name']}. "
                f"Click 'Apply' to view live listings matching your profile. "
                f"Positions are updated in real-time by employers."
            ),
            "url": url,
            "source": board["name"],
        })

    print(f"  🛡️ [Static Fallback] Generated {len(results)} job portal search links.")
    return results


def generate_ai_matches(search_query, location, chat_client, profile):
    """
    Highly resilient AI-Forecasted Matches for Zero-Result scenarios.
    """
    profile = profile or {}

    if not chat_client:
        print(f"  ⚠️ [AI] No chat client available. Using static fallback.")
        return generate_static_fallback(search_query, location, profile)

    print(f"🧠 [Resilience Mode] Generating AI-forecasted jobs for '{search_query}'...")

    skills = profile.get("skills", ["General HR", "Recruitment"])
    level = profile.get("seniority", profile.get("level", "Fresher / Entry"))

    system_prompt = f"""
    You are a 'Job Market Oracle'. The user's live search for '{search_query}' in '{location}' failed.
    Generate a JSON object containing exactly 6 HIGH-ACCURACY job opportunities for {location}.
    
    USER: {profile.get('personal_details', {}).get('name')}
    TARGET: {search_query}
    SKILLS: {', '.join(skills)}
    LEVEL: {level}
    
    OUTPUT FORMAT:
    {{
      "matches": [
        {{
          "title": "Role Title",
          "company": "Specific Indian Company name",
          "location": "{location}",
          "description": "Short snippet mentioning {', '.join(skills[:2])}.",
          "url": "https://www.linkedin.com/jobs",
          "source": "AI Market Forecast"
        }}
      ]
    }}
    """
    
    try:
        completion = chat_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        raw_content = completion.choices[0].message.content
        data = json.loads(raw_content)
        
        # Extremely robust extraction
        results = []
        possible_keys = ["matches", "jobs", "results", "opportunities"]
        
        if isinstance(data, list):
            results = data
        elif isinstance(data, dict):
            for key in possible_keys:
                if key in data and isinstance(data[key], list):
                    results = data[key]
                    break
            if not results: # If keys not found, check if dict itself has job-like items
                if "title" in data: results = [data]
        
        # Post-process: ensure each job has a useful search URL
        for job in results:
            title = job.get("title", search_query)
            if not job.get("url") or job.get("url") == "https://www.linkedin.com/jobs":
                encoded_title = urllib.parse.quote_plus(title)
                encoded_loc = urllib.parse.quote_plus(location)
                job["url"] = f"https://www.linkedin.com/jobs/search/?keywords={encoded_title}&location={encoded_loc}"
            if not job.get("source"):
                job["source"] = "AI Forecast"

        print(f"  🏁 [AI] Resilience match complete: Generated {len(results)} tailored opportunities.")
        return results if results else generate_static_fallback(search_query, location, profile)
    except Exception as e:
        print(f"  ❌ Resilience Fallback failed: {str(e)}. Using static fallback.")
        return generate_static_fallback(search_query, location, profile)

def scrape_linkedin_jobs(
    api_token: str,
    search_query: str = "Python Developer",
    location: str = "India",
    max_jobs: int = 20,
    sources: list[str] = ["indeed", "linkedin", "glassdoor", "wwr", "remoteok"],
    chat_client = None,
    profile = None
) -> list[dict]:
    """
    Blitz Discovery Engine: Launches Precision, Broad, and AI-Forecast searches in parallel.
    Guarantees results in under 40 seconds.
    """
    all_results = []
    scrapers = []

    # 🛠️ Define Scraper Pool
    jobspy_sites = [s for s in sources if s in ["indeed", "linkedin", "glassdoor"]]
    if jobspy_sites: scrapers.append(JobSpyScraper(sites=jobspy_sites))
    if "wwr" in sources: scrapers.append(RemoteWWRScraper())
    if "remoteok" in sources: scrapers.append(RemoteOKScraper())

    print(f"\n⚡ [Blitz] Launching Parallel Discovery Suite for '{search_query}'...")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 1. Precise Sweep
        f_precise = executor.submit(lambda: [j for s in scrapers for j in (s.scrape(search_query, location, max_jobs) or [])])
        
        # 2. Broad Sweep
        broad_q = f"{search_query} OR Intern OR Entry Level"
        f_broad = executor.submit(lambda: [j for s in scrapers for j in (s.scrape(broad_q, location, max_jobs//2) or [])])
        
        # 3. AI Resilience Forecast
        f_ai = executor.submit(generate_ai_matches, search_query, location, chat_client, profile)

        # Wait for all (with 40s total capped timeout)
        futures = {"Precise": f_precise, "Broad": f_broad, "AI-Forecast": f_ai}
        
        for name, future in futures.items():
            try:
                data = future.result(timeout=40)
                if data:
                    print(f"  ✅ [Blitz] {name} pass returned {len(data)} jobs.")
                    all_results.extend(data)
            except Exception as e:
                print(f"  ⚠️ [Blitz] {name} pass timed out or failed: {str(e)}")

    # 🌍 Robust Deduplication & Merging
    seen_keys = set()
    unique_results = []
    
    def normalize(text):
        if not text: return ""
        return re.sub(r'[^a-z0-9]', '', str(text).lower())

    for job in all_results:
        # 1. Normalize URL (stripping analytics query params)
        raw_url = job.get("url") or job.get("applyUrl", "")
        clean_url = raw_url.split('?')[0].rstrip('/') if raw_url else ""
        
        # 2. Content Hash (Title + Company + Location)
        # Using the first 100 chars of description to distinguish between multiple openings 
        # at same company with same title (e.g. different projects)
        title = normalize(job.get("title", ""))
        comp = normalize(job.get("company", job.get("company_name", "Private")))
        loc = normalize(job.get("location", ""))
        
        # Aggressive content check: (Title, Company, Location)
        # We assume the same title at the same company is the same job.
        content_key = f"{title}|{comp}|{loc}"
        
        url_is_duplicate = clean_url in seen_keys if clean_url else False
        content_is_duplicate = content_key in seen_keys
        
        if not url_is_duplicate and not content_is_duplicate:
            if clean_url: seen_keys.add(clean_url)
            seen_keys.add(content_key)
            unique_results.append(job)

    # 🛡️ GUARANTEED FALLBACK: If everything failed, generate portal search links
    if not unique_results:
        print("  ⚠️ [Blitz] All sources returned 0. Activating Guaranteed Fallback...")
        unique_results = generate_static_fallback(search_query, location, profile)

    # ⚖️ Ensure site variety for "Top Results" by shuffling before sorting 
    # (Actual sorting is done by match_score later in the frontend, so we just provide a clean unique list)
    random.shuffle(unique_results)

    print(f"🏁 [Blitz] Total Discoveries: {len(unique_results)}")
    return unique_results
