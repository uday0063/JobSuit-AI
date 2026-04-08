# JobSuit AI 🚀  
### Intelligent Job Matching & Auto-Apply Platform

JobSuit AI is an AI-powered job intelligence system designed to optimize the job search process through intelligent profile-based matching, filtering, and automation.

It analyzes user skills and preferences to surface high-relevance opportunities, eliminate low-quality or fraudulent listings, and streamline the application workflow.

---

## ✨ Key Features

- 🔍 AI-based job matching using profile analysis  
- 🧠 Intelligent filtering (removes irrelevant/fraud jobs)  
- 📊 Match scoring system  
- ⚡ Automated job application workflow  
- 🎯 Clean dashboard UI  

---

## 🛠 Tech Stack

- React + Tailwind CSS  
- Python + FastAPI  
- Groq API  

---

## ⚙️ Setup

Create a `.env` file:








# 🔍 Fake Job Posting Detector


> **NLP + XGBoost Hybrid Classifier** — Detect fraudulent job postings using text signals and structured metadata.

---

## 📁 Project Structure

```
fake_job_detector/
├── fake_job_detector.py        ← Main entry point (CLI router)
├── requirements.txt
├── data/
│   └── jobs.csv                ← Your training dataset (CSV)
├── models/                     ← Saved model artifacts (auto-created)
│   ├── xgb_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── pipeline_metadata.json
└── src/
    ├── __init__.py
    ├── preprocessor.py         ← Text cleaning + structured feature engineering
    ├── features.py             ← TF-IDF vectorisation + sparse hstack
    ├── model.py                ← XGBoost classifier wrapper
    ├── pipeline.py             ← End-to-end train & inference orchestrator
    ├── cli.py                  ← Interactive terminal prediction UI
    ├── api.py                  ← Flask REST API
    ├── demo.py                 ← Self-contained demo (no dataset needed)
    └── data_generator.py       ← Synthetic dataset generator
```

---

## ⚙️ Setup

```bash
# 1. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Mode 1 — Demo (no dataset required)
Trains on built-in synthetic data and runs sample predictions immediately:
```bash
python fake_job_detector.py --mode demo
```

### Mode 2 — Train on your own dataset
```bash
# Option A: Use the synthetic dataset generator
python -m src.data_generator --rows 2000 --fake-ratio 0.15 --out data/jobs.csv

# Option B: Use the real EMSCAD dataset (recommended)
#   Download from: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
#   Save as: data/jobs.csv

python fake_job_detector.py --mode train --data data/jobs.csv
```

### Mode 3 — Interactive CLI Prediction
After training, analyse custom job postings interactively:
```bash
python fake_job_detector.py --mode predict
```

### Mode 4 — Flask REST API
```bash
python fake_job_detector.py --mode api --port 5000
```

---

## 🌐 Flask API Endpoints

| Method | Endpoint    | Description                        |
|--------|-------------|------------------------------------|
| `GET`  | `/health`   | API health check                   |
| `POST` | `/predict`  | Classify a job posting             |
| `GET`  | `/features` | Top-N most important features      |

### Example `/predict` Request
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "title": "Work From Home - Earn $500/Day!",
           "description": "No experience needed! Amazing opportunity! Click here now! Guaranteed income!!!",
           "company_name": "",
           "salary_range": "",
           "requirements": "None",
           "location": "",
           "employment_type": "Part-time",
           "telecommuting": 1,
           "has_company_logo": 0
         }'
```

### Example Response
```json
{
  "label": 1,
  "label_str": "FAKE",
  "probability_fake": 0.9741,
  "probability_real": 0.0259,
  "suspicious_words": {
    "found": ["work from home", "no experience", "guaranteed", "click here", "earn from home"],
    "score": 0.2632
  },
  "latency_ms": 12.4
}
```

---

## 🧠 How It Works

```
Raw Job Posting
      │
      ▼
┌─────────────────────────────────┐
│  Text Preprocessing             │
│  • Lowercase + strip HTML/tags  │
│  • Replace URLs with token      │
│  • Combine title+desc+req       │
└────────────┬────────────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
 TF-IDF (5000   Structured Features
  bigram feats)  • has_company
                 • has_salary
                 • desc_length
                 • exclamation_count
                 • link_count
                 • has_logo
                 • telecommuting
                 • employment_type
      │             │
      └──────┬───────┘
             ▼
      scipy.sparse.hstack
      (5000 + 10 features)
             │
             ▼
      XGBoost Classifier
      (scale_pos_weight for
       class imbalance)
             │
             ▼
   Fake (1) / Real (0)
   + Probability Score
   + Suspicious Words
```

---

## 📊 Features Engineered

### Text Features (TF-IDF)
- **5,000 unigram + bigram** features extracted from combined `title + description + requirements`
- `sublinear_tf=True` — log-normalised term frequencies reduce the impact of very frequent terms

### Structured Features

| Feature | Description |
|---|---|
| `has_company` | 1 if `company_name` is non-empty |
| `has_salary` | 1 if `salary_range` is non-empty |
| `has_location` | 1 if `location` is non-empty |
| `has_logo` | Value of `has_company_logo` field |
| `telecommuting` | Value of `telecommuting` field |
| `desc_length` | Character length of description |
| `req_length` | Character length of requirements |
| `exclamation_count` | Number of `!` in description |
| `link_count` | Number of URLs (`http/www`) in description |
| `employment_type_enc` | Ordinal encoding of employment type |

### 🚩 Suspicious Word List (30 red-flag phrases)
`urgent`, `work from home`, `no experience`, `easy money`, `guaranteed`, `click here`, `unlimited earning`, `passive income`, `wire transfer`, `make money fast`, `get rich`, `daily payment`, `no interview`, `whatsapp`, `free training`, `be your own boss` … and more.

---

## 📈 Model

| Parameter | Value |
|---|---|
| Algorithm | `XGBClassifier` |
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| Class imbalance | `scale_pos_weight = neg/pos` |
| Train/test split | 80 / 20 (stratified) |

---

## 📦 Dataset

For best results, use the **EMSCAD (Employment Scam Aegean Corpus)** dataset:

> 📥 **Download:** [Kaggle — Real or Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

Save it as `data/jobs.csv`. The required columns are:
`title`, `company_name`, `location`, `salary_range`, `description`, `requirements`, `benefits`, `employment_type`, `telecommuting`, `has_company_logo`, `fraudulent`

Alternatively, generate synthetic data for testing:
```bash
python -m src.data_generator --rows 5000 --fake-ratio 0.15 --out data/jobs.csv
```

---

## 🛡️ Safety Tips

This tool highlights these red flags common in fraudulent job postings:
- ❌ No company name or logo
- ❌ Vague or missing salary / location
- ❌ Excessive exclamation marks (`!!!`)
- ❌ Suspicious phrases: *"no experience needed"*, *"guaranteed income"*, *"work from home"* combined with other signals
- ❌ Requests for payment, wire transfer, or WhatsApp contact

---

*Built with Python · scikit-learn · XGBoost · Flask*
