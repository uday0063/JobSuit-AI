import re
from src.pipeline import load_inference_pipeline, predict_single
from src.linkedin_mapper import map_linkedin_to_schema

# 🔍 Global Tuning
USE_AI_FILTER = True # Toggle to False to disable fraud detection

INDUSTRY_SYNONYMS = {
    "hr": ["human resources", "recruitment", "talent acquisition", "hr generalist", "hr executive"],
    "sde": ["software developer", "software engineer", "coder", "programmer"],
    "ml": ["machine learning", "ai", "artificial intelligence", "data scientist"],
    "frontend": ["react", "vue", "angular", "javascript", "ui developer"],
    "qa": ["tester", "quality assurance", "automation engineer"]
}

def get_synonyms(word):
    word = word.lower()
    for key, syns in INDUSTRY_SYNONYMS.items():
        if word == key or word in syns:
            return set([key] + syns)
    return {word}

from concurrent.futures import ThreadPoolExecutor

def filter_jobs(raw_jobs, extractor, classifier, threshold=0.8, user_profile=None):
    """
    Ultra-Lenient Discovery: Prioritizes showing jobs over aggressive filtering.
    """
    all_visible_jobs = []
    fake_jobs = [] # Kept for API signature
    
    initial_count = len(raw_jobs)
    print(f"\n🕵️ [Max-Discovery] Analyzing {initial_count} discoveries with Relaxed Threshold (0.8)")

    def process_one_job(raw_job):
        desc_lower = raw_job.get("description", "").lower()
        title_lower = raw_job.get("title", "").lower()
        if not title_lower: return None

        match_score = 100
        reasons = []

        # We've removed Seniority/Title blocks to ensure max results
        # Only subtle penalties for severe mismatches remain
        
        # 1. Broad Skill Match (Max Penalty -20)
        personal_details = user_profile.get("personal_details", {}) if user_profile else {}
        must_haves = [s.lower() for s in personal_details.get("must_have_skills", [])]
        if must_haves:
            matches = [s for s in must_haves if any(syn in desc_lower for syn in get_synonyms(s))]
            if len(matches) < 1: # Only penalized if ZERO target skills found
                match_score -= 20
                reasons.append("Requires different skill focus")

        # 2. Experience Check (Max Penalty -20)
        max_exp = int(personal_details.get("max_experience_years", 3))
        match = re.search(r"(\d+)\+?\s*years?", desc_lower)
        if match and int(match.group(1)) > (max_exp + 2): # Very generous buffer
            match_score -= 20
            reasons.append(f"Senior level role ({match.group(1)}y+ exp)")

        # 3. Aggressive Location Filter (Regional Geofence)
        target_loc = personal_details.get("preferred_location", "").lower()
        job_loc = raw_job.get("location", "").lower()
        
        # Geofencing: Detect if job is in US/OH/etc while user is local
        far_tags = [", us", ",usa", " oh", " ohio", " ny", " nj", " ca", " tx", " mi", " cincinnati"]
        local_tags = ["delhi", "ghaziabad", "india", "ncr", "noida", "gurgaon"]
        is_far = any(tag in job_loc for tag in far_tags)
        is_local_search = any(tag in target_loc for tag in local_tags)

        if target_loc and target_loc not in job_loc and not any(w in job_loc for w in target_loc.replace('new', '').split() if len(w)>3):
            if is_local_search and is_far:
                match_score -= 100 # Complete regional mismatch
                reasons.append(f"Geofenced (Distant Region): {raw_job.get('location', 'Unknown')}")
            elif "remote" not in job_loc:
                match_score -= 100 # Very Strict Geofence
                reasons.append(f"Region mismatch: {raw_job.get('location', 'Unknown')}")
            else:
                # Far Remote - Hide these too unless specifically searching for remote worldwide
                if "remote" not in target_loc:
                    match_score -= 60
                    reasons.append(f"Distant Remote: {raw_job.get('location', '')}")
                else:
                    match_score -= 20
                    reasons.append(f"Remote Role: {raw_job.get('location', '')}")

        # Mapping & Multi-Level Trust Scoring
        mapped_job = map_linkedin_to_schema(raw_job)
        mapped_job["match_score"] = int(max(10, match_score))
        mapped_job["match_reasons"] = reasons

        # AI Trust Check (RELAXED THRESHOLD)
        if USE_AI_FILTER:
            try:
                res = predict_single(mapped_job, extractor, classifier)
                prob_fake = res["probability_fake"]
                mapped_job["probability_fake"] = prob_fake
                
                # We now mark fakes as "Suspicious" but KEEP them visible 
                # unless they are 80%+ likely to be scams.
                if prob_fake > 0.8:
                    mapped_job["is_suspicious"] = True
                    mapped_job["match_reasons"].append("⚠️ High risk of scam")
                    return mapped_job, False
                return mapped_job, True
            except:
                return mapped_job, True
        else:
            mapped_job["probability_fake"] = 0
            return mapped_job, True

    # parallel analysis
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_one_job, raw_jobs))

    for res in results:
        if res is None: continue
        job, is_safe = res
        # Only show jobs that have a minimum match relevance (e.g. > 45)
        # This effectively hides jobs from completely different regions/nations.
        if (job.get("match_score", 0) > 45):
            all_visible_jobs.append(job) 

    print(f"🕵️ [Max-Discovery] Process complete. Delivering {len(all_visible_jobs)} job cards.")
    return all_visible_jobs, []
