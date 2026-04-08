"""
=============================================================
  src/resume_parser.py — Deep Semantic Profiler
=============================================================
  Performs deep analysis of resume content to extract
  competencies, seniority, and search-optimized queries.
"""

from pypdf import PdfReader
import re
import json
from collections import Counter

# 🧠 Comprehensive Skill Matrix
SKILL_CATEGORIES = {
    "AI/ML": ["python", "machine learning", "ml", "artificial intelligence", "nlp", "computer vision", "cv", "deep learning", "neural networks", "stats", "statistics"],
    "Frameworks": ["pytorch", "tensorflow", "keras", "scikit-learn", "xgboost", "nltk", "spacy", "huggingface", "fastai", "opencv", "yolo"],
    "Data": ["sql", "pandas", "numpy", "matplotlib", "seaborn", "tableau", "pyspark", "bigquery", "etl", "data engineering", "feature engineering"],
    "Cloud/Ops": ["aws", "azure", "gcp", "docker", "kubernetes", "mlops", "fastapi", "flask", "git", "ci/cd", "deployment", "databricks"],
    "Tools": ["jupyter", "vscode", "colab", "excel", "powerbi"]
}

# 🚩 Seniority Keywords
SENIORITY_MARKERS = {
    "Fresher": ["intern", "graduate", "fresher", "junior", "student", "trainee"],
    "Managerial": ["lead", "manager", "head", "architect", "senior", "principal"]
}

def analyze_resume_deeply(file_path: str):
    """
    Exhaustive analysis of the PDF to build a career profile.
    """
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += (page.extract_text() or "") + " "
    except Exception as e:
        return {"error": str(e)}

    text_lower = text.lower()
    
    # 1. Extraction of Skills (Categorized)
    found_skills = {}
    all_matched = []
    for cat, skills in SKILL_CATEGORIES.items():
        matches = [s for s in skills if re.search(rf"\b{re.escape(s)}\b", text_lower)]
        if matches:
            found_skills[cat] = matches
            all_matched.extend(matches)

    # 2. Seniority Assessment
    level = "Junior / Professional"
    
    fresher_count = sum(text_lower.count(s) for s in SENIORITY_MARKERS["Fresher"])
    senior_count = sum(text_lower.count(s) for s in SENIORITY_MARKERS["Managerial"])
    
    # If explicit fresher keywords appear, heavily lean towards entry level
    if fresher_count > 0 and fresher_count >= senior_count:
        level = "Fresher / Entry Level"
    elif senior_count > 2:
        level = "Senior / Lead"
    # 3. Domain Identification (Primary Strength)
    domain_scores = Counter()
    if any(s in text_lower for s in ["nlp", "text", "transformer", "bert", "gpt", "nltk"]): domain_scores["NLP"] += 2
    if any(s in text_lower for s in ["cv", "vision", "image", "yolo", "cnn", "opencv"]): domain_scores["Computer Vision"] += 2
    if any(s in text_lower for s in ["tabular", "xgboost", "random forest", "regression"]): domain_scores["Tabular ML"] += 2
    
    primary_domain = domain_scores.most_common(1)[0][0] if domain_scores else "General AI/ML"

    # 4. Generate Optimized Search Query
    # Picks the top 2 unique tools + primary domain
    top_tools = all_matched[:2]
    search_query = f"{primary_domain} {' '.join(top_tools)}" if top_tools else f"{primary_domain} Engineer"

    return {
        "role": f"{primary_domain} Engineer",
        "level": level,
        "skills": all_matched,
        "skill_map": found_skills,
        "search_query": search_query,
        "full_text": text
    }

def summarize_job_fit(job_desc: str, profile: dict) -> str:
    """
    Hyper-personalized career rationale inspired by elite job boards.
    """
    desc_lower = job_desc.lower()
    skills = profile.get("skills", [])
    name = profile.get("name", "").split()[0] if profile.get("name") else "Nishant"
    role = profile.get("role_category", "Professional")
    
    # Identify overlaps
    overlaps = [s for s in skills if s.lower() in desc_lower]
    
    # Domain matching
    is_fintech = any(w in desc_lower for w in ["bank", "fintech", "finance", "payment", "ledger"])
    is_ai = any(w in desc_lower for w in ["ai", "ml", "intelligence", "llm", "neural"])
    is_bigtech = any(w in desc_lower for w in ["google", "amazon", "microsoft", "meta", "apple", "netflix"])
    
    if len(overlaps) >= 3:
        reason = f"Excellent strategic fit. Your {overlaps[0]} and {overlaps[1]} expertise aligns perfectly with their {role} requirements."
        if is_fintech: reason = f"Direct fit for this Fintech role. Your background in {overlaps[0]} bridges their technical gaps perfectly."
        if is_ai: reason = f"AI-First fit! Your deep competence in {overlaps[0]} and {overlaps[1]} makes you a high-probability candidate."
        return reason
    elif len(overlaps) > 0:
        return f"Consistent with your {role} career path. Specific overlap found in {overlaps[0]}."
    else:
        return f"Strategic stretch role. Aligns with your broader {role} trajectory."

def tailor_profile_with_ai(pdf_text: str, client):
    """
    Uses LLM to transform raw resume text into a search-optimized career profile.
    """
    if not client:
        return analyze_resume_deeply_fallback(pdf_text)

    system_prompt = """
    You are a 'Hyper-Specialized Career Architect'. Your job is to extract 
    'Career DNA' from a resume and format it for a job-matching JSON response.
    
    CRITICAL: YOU MUST RETURN ONLY A PURE JSON OBJECT. NO TEXT BEFORE OR AFTER.
    
    JSON STRUCTURE:
    {
      "name": "Full Name",
      "email": "Email Address",
      "phone": "Phone Number",
      "search_query": "Best 3-word query for LinkedIn/Indeed (e.g., 'HR Executive', 'Python Developer')",
      "preferred_location": "Likely city based on experience (default to 'Delhi')",
      "max_experience_years": Number (total years of exp),
      "skills": ["Skill1", "Skill2", "Skill3", "Skill4", "Skill5", "Skill6"],
      "role_category": "Main Domain (e.g. HR, AI/ML, SDE, Marketing)",
      "seniority": "Junior", "Mid", or "Senior"
    }
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract Career DNA from this resume text:\n\n{pdf_text[:8000]}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        profile = json.loads(completion.choices[0].message.content)
        return profile
    except Exception as e:
        print(f"❌ AI Tailoring Error: {str(e)}")
        return analyze_resume_deeply_fallback(pdf_text)

def analyze_resume_deeply_fallback(text: str):
    return {
        "name": "Extracted Name",
        "search_query": "Career Professional",
        "skills": ["General"],
        "max_experience_years": 2,
        "role_category": "General"
    }
