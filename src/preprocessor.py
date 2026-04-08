"""
=============================================================
  src/preprocessor.py — Data Preprocessing & Feature Engineering
=============================================================
  Handles:
    - Missing value imputation
    - Text cleaning & normalization
    - Structured feature extraction
    - Text field combination for NLP
"""

import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants — columns used for text & structured features
# ---------------------------------------------------------------------------
TEXT_COLS = ["title", "description", "requirements", "benefits"]
STRUCTURED_COLS = [
    "has_company",
    "has_salary",
    "has_location",
    "desc_length",
    "req_length",
    "exclamation_count",
    "link_count",
    "has_logo",
    "telecommuting",
    "employment_type_enc",
    "has_phone",
    "has_personal_email",
    "high_caps_ratio",
    "suspicious_score",
]

SUSPICIOUS_WORDS = [
    # General & Pressure
    "urgent", "work from home", "no experience", "easy money", "guaranteed",
    "click here", "apply now", "unlimited earning", "passive income", "wire transfer",
    "make money fast", "get rich", "bonus", "earn from home", "daily payment",
    "no interview", "part time", "data entry", "home based", "whatsapp",
    "free training", "100%", "amazing opportunity", "be your own boss",
    "hurry", "limited seats", "slots left", "immediate joining",
    "no skills", "start today", "earn while you sleep",
    # Payment & Fees
    "registration fee", "refundable", "activation fee", "security deposit",
    "processing fee", "payment required", "earn lakhs", "easy cash", "₹",
    "upi", "gpay", "paytm", "phonepe", "bank details", "payout",
    # Micro-Payment & Booking Scams
    "zoom call", "book your seat", "pay 49", "micro payment", "/-",
    "slot booking", "paid orientation", "interview fee",
    "pay now", "qr", "qr code", "scanner", "payment link", "transaction",
    # MLM & Jargon
    "network marketing", "direct selling", "join now", "be a partner",
    "side hustle", "income source", "fast cash",
]

EMPLOYMENT_TYPE_MAP = {
    "Full-time": 0,
    "Part-time": 1,
    "Contract": 2,
    "Temporary": 3,
    "Internship": 4,
    "Other": 5,
    "": -1,
}


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Lowercase, strip HTML tags, remove special characters,
    but PRESERVE currency symbols and exclamation marks.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # remove HTML tags
    text = re.sub(r"http\S+|www\S+", " url ", text)  # replace URLs with token
    
    # Keep letters, digits, spaces, and critical symbols like ₹, $, /, -
    text = re.sub(r"[^a-z0-9\s!$₹?%/-]", " ", text)
    
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Structured Feature Extraction
# ---------------------------------------------------------------------------
def extract_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create hand-crafted features from the raw DataFrame.
    Returns a DataFrame with only the structured feature columns.
    """
    feats = pd.DataFrame(index=df.index)

    # Core metadata presence
    feats["has_company"] = df["company_name"].apply(
        lambda x: 1 if isinstance(x, str) and x.strip() != "" else 0
    )
    feats["has_salary"] = df["salary_range"].apply(
        lambda x: 1 if isinstance(x, str) and x.strip() != "" else 0
    )
    feats["has_location"] = df["location"].apply(
        lambda x: 1 if isinstance(x, str) and x.strip() != "" else 0
    )
    feats["has_logo"] = df["has_company_logo"].fillna(0).astype(int)
    feats["telecommuting"] = df["telecommuting"].fillna(0).astype(int)

    # Text length & punctuation
    feats["desc_length"] = df["description"].apply(
        lambda x: len(str(x)) if isinstance(x, str) else 0
    )
    feats["req_length"] = df["requirements"].apply(
        lambda x: len(str(x)) if isinstance(x, str) else 0
    )
    feats["exclamation_count"] = df["description"].apply(
        lambda x: str(x).count("!") if isinstance(x, str) else 0
    )
    feats["link_count"] = df["description"].apply(
        lambda x: len(re.findall(r"http\S+|www\S+", str(x))) if isinstance(x, str) else 0
    )

    # 🚩 ADVANCED REGEX FEATURES
    # 1. Phone number presence (Indian & global formats)
    phone_pattern = r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+91[-.\s]?\d{10}|\b\d{10}\b"
    feats["has_phone"] = df["description"].apply(
        lambda x: 1 if re.search(phone_pattern, str(x)) else 0
    )

    # 2. Personal email presence (Scammers use @gmail.com)
    personal_email_pattern = r"\b[A-Za-z0-9._%+-]+@(gmail|yahoo|outlook|hotmail|rediff|aol)\.com\b"
    feats["has_personal_email"] = df["description"].apply(
        lambda x: 1 if re.search(personal_email_pattern, str(x), re.IGNORECASE) else 0
    )

    # 4. Suspicious Word Count (direct numeric feature)
    def count_red_flags(text):
        t = str(text).lower()
        return sum(1 for word in SUSPICIOUS_WORDS if word in t)
    
    feats["suspicious_score"] = df["description"].apply(count_red_flags)

    # Encode employment type
    feats["employment_type_enc"] = (
        df["employment_type"]
        .fillna("")
        .map(lambda x: EMPLOYMENT_TYPE_MAP.get(str(x).strip(), 5))
    )

    return feats


# ---------------------------------------------------------------------------
# Combined Text Column
# ---------------------------------------------------------------------------
def build_combined_text(df: pd.DataFrame) -> pd.Series:
    """
    Combine title + description + requirements into one cleaned text string.
    This is the input for TF-IDF vectorization.
    """
    combined = (
        df["title"].fillna("").apply(clean_text)
        + " "
        + df["description"].fillna("").apply(clean_text)
        + " "
        + df["requirements"].fillna("").apply(clean_text)
    )
    return combined.str.strip()


# ---------------------------------------------------------------------------
# Full Preprocessing Pipeline
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Main preprocessing function.

    Args:
        df: Raw input DataFrame (must match the expected schema).

    Returns:
        combined_text : pd.Series — cleaned text for TF-IDF
        struct_feats  : pd.DataFrame — structured/numeric features
    """
    # Ensure required columns exist; fill missing with defaults
    required_text_cols = ["title", "description", "requirements", "benefits",
                          "company_name", "salary_range", "location"]
    for col in required_text_cols:
        if col not in df.columns:
            df[col] = ""

    for col in ["has_company_logo", "telecommuting"]:
        if col not in df.columns:
            df[col] = 0

    if "employment_type" not in df.columns:
        df["employment_type"] = ""

    combined_text = build_combined_text(df)
    struct_feats = extract_structured_features(df)

    return combined_text, struct_feats


# ---------------------------------------------------------------------------
# Suspicious Word Highlighter (for predictions)
# ---------------------------------------------------------------------------
def highlight_suspicious_words(text: str) -> dict:
    """
    Scan input text for known suspicious / red-flag patterns.

    Returns:
        dict with keys:
          'found'   : list of matched suspicious phrases
          'score'   : suspicion score (0-1), proportional to matches
    """
    text_lower = text.lower()
    found = [word for word in SUSPICIOUS_WORDS if word in text_lower]
    score = min(len(found) / max(len(SUSPICIOUS_WORDS), 1), 1.0)
    return {"found": found, "score": round(score, 4)}
