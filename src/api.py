import os
import werkzeug
import json
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from groq import Groq

from src.pipeline import load_inference_pipeline
from src.resume_parser import analyze_resume_deeply, summarize_job_fit, tailor_profile_with_ai
from src.scraper_manager import discover_jobs
from src.filter import filter_jobs
from src.auto_apply import trigger_auto_apply_async
from src import cache as job_cache
from src.config import (
    GROQ_API_KEY, APIFY_TOKEN, MODEL_DIR,
    STRICTNESS_MAP, DEFAULT_SOURCES,
)
from src.logger import get_logger

load_dotenv()
log = get_logger("api")


def create_app(model_dir=MODEL_DIR):
    app = Flask(__name__, template_folder="../templates", static_folder="../static")

    # Load AI pipeline
    log.info("Loading AI pipeline from '%s'...", model_dir)
    extractor, classifier = load_inference_pipeline(model_dir)
    app.extractor = extractor
    app.classifier = classifier

    # Groq client
    if GROQ_API_KEY and "your_groq_api_key" not in GROQ_API_KEY:
        app.chat_client = Groq(api_key=GROQ_API_KEY)
    else:
        app.chat_client = None
        log.warning("GROQ_API_KEY not set — chatbot disabled")

    # Warm cache from disk
    job_cache.load_from_disk()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route('/upload_resume', methods=['POST'])
    def upload_resume():
        if 'resume' not in request.files:
            return jsonify({"error": "No file"}), 400

        file = request.files['resume']
        UPLOAD_FOLDER = 'uploads'
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        filename = werkzeug.utils.secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        raw_res = analyze_resume_deeply(path)
        text = raw_res.get("full_text", "")
        profile = tailor_profile_with_ai(text, app.chat_client)

        return jsonify({"status": "success", "profile": profile})

    @app.route('/find_matches', methods=['POST'])
    def find_matches():
        profile = request.get_json()
        query = profile.get("search_query", "AI/ML Engineer")
        location = profile.get("personal_details", {}).get("preferred_location", "Delhi")
        sources = profile.get("sources", DEFAULT_SOURCES)
        strictness = profile.get("strictness", 1)
        threshold = STRICTNESS_MAP.get(strictness, 0.5)

        refresh = profile.get("refresh", False)
        
        # If refreshing, rotate the query and expand location to discover MORE jobs
        if refresh:
            original_query = query
            skills = profile.get("skills", [])
            import random
            
            # 🆕 Geo-Expansion: If first search was local, broaden to Remote/India on refresh
            original_loc = location
            expansions = ["Remote", "India", "Global"]
            location = random.choice(expansions)
            log.info("Refreshing with expanded location: '%s' (was '%s')", location, original_loc)

            # 🆕 Query Diversification: Use specific skill combinations to find niche openings
            if skills:
                # Pick a random skill as the PRIMARY search term to avoid search fatigue
                main_skill = random.choice(skills)
                query = f"{main_skill} {original_query}"
                log.info("Refreshing with rotated query: '%s'", query)

        log.info("find_matches: query='%s', location='%s', strictness=%s, refresh=%s", query, location, strictness, refresh)

        raw_jobs = discover_jobs(
            api_token=APIFY_TOKEN,
            query=query,
            location=location,
            sources=sources,
            chat_client=app.chat_client,
            profile=profile,
            use_cache=not refresh
        )
        log.info("Total raw jobs discovered: %d", len(raw_jobs))

        legit, fake = filter_jobs(
            raw_jobs,
            app.extractor,
            app.classifier,
            threshold=threshold,
            user_profile=profile,
        )
        log.info("Filter results: %d legit, %d fake", len(legit), len(fake))

        for job in legit:
            job["ai_fit_summary"] = summarize_job_fit(job.get("description", ""), profile)

        # Phase 4: Autonomous Apply
        if profile.get("auto_apply", False):
            log.info("Auto-Apply is enabled. Triggering background task...")
            # We assume the uploaded resume is saved locally
            # In a real app we'd track the user's latest resume path
            trigger_auto_apply_async(legit, "uploads/resume.pdf", profile)

        return jsonify({"results": legit, "raw_count": len(raw_jobs)})

    @app.route('/chat', methods=['POST'])
    def chat():
        if not app.chat_client:
            return jsonify({"error": "AI Chatbot is not configured. Please add GROQ_API_KEY to .env"}), 503

        data = request.get_json()
        messages = data.get("messages", [])
        job_context = data.get("job_context", {})
        user_profile = data.get("user_profile", {})

        system_msg = f"""
        # ROLE: LEAD CAREER STRATEGIST & APPLICATION EXPERT
        You are the 'AI Career Strategist' for {user_profile.get('personal_details', {}).get('name', 'the user')}.
        Your goal is to provide elite application strategies and craft high-impact application content.

        ## STRATEGIC CONTEXT
        - TARGET JOB: {job_context.get('title')} at {job_context.get('company')}
        - SKILL VECTORS: {', '.join(user_profile.get('skills', []))}
        - SENIORITY LEVEL: {user_profile.get('seniority', 'Professional')}

        ## MANDATORY OUTPUT STRUCTURE:
        1. **### APPLICATION STRATEGY**
           - A bulleted list of tactical advice on why the user is a strong fit.
        2. **### STRATEGIC CONTENT (COVER LETTER / RESUME TAILORING)**
           - If tailoring a resume, provide a high-impact **RESUME SUMMARY** and a **WORK EXPERIENCE REWRITE** for their top 2-3 bullet points.
           - If crafting a cover letter, provide a professional, persuasive letter that bridges the gap between their background and this job.

        ## TONE & STYLE:
        - Elite, tactical, and encouraging.
        - Use Markdown headers, bold text, and lists for high readability.
        - Avoid generic fluff. Focus on ROI.
        """

        try:
            completion = app.chat_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system_msg}] + messages,
                temperature=0.7,
                max_tokens=1024,
            )
            return jsonify({"response": completion.choices[0].message.content})
        except Exception as e:
            log.error("Chat error: %s", e)
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
