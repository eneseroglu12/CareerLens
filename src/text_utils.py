from __future__ import annotations

import re
from collections import Counter

TR_EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "from", "is", "are", "be", "been", "this", "that", "these", "those", "you", "your", "we",
    "our", "will", "can", "should", "must", "have", "has", "had", "it", "its", "into", "about",
    "bir", "ve", "veya", "ile", "için", "bu", "şu", "o", "de", "da", "den", "dan", "olan",
    "olarak", "gibi", "çok", "daha", "en", "mi", "mı", "mu", "mü", "ise", "hem", "her",
    "iş", "pozisyon", "aday", "candidate", "role", "position", "job", "work", "experience",
}


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9][A-Za-zÇĞİÖŞÜçğıöşü0-9+#.\-/]{1,}", text.lower())
    return [token.strip(".-/") for token in tokens if token.strip(".-/") not in TR_EN_STOPWORDS]


def keyword_overlap(cv_text: str, job_text: str, limit: int = 18) -> dict:
    cv_tokens = Counter(tokenize_keywords(cv_text))
    job_tokens = Counter(tokenize_keywords(job_text))

    if not job_tokens:
        return {"coverage": 0.0, "matched": [], "missing": []}

    ranked_job_terms = [term for term, _ in job_tokens.most_common(60)]
    matched = [term for term in ranked_job_terms if term in cv_tokens]
    missing = [term for term in ranked_job_terms if term not in cv_tokens]

    coverage = len(matched) / max(len(ranked_job_terms), 1)
    return {
        "coverage": coverage,
        "matched": matched[:limit],
        "missing": missing[:limit],
    }
