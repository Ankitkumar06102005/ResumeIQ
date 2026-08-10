"""
preprocessing.py — Text cleaning and normalization pipeline.

Handles: lowercasing, special-char removal, stopword filtering, lemmatization.
Used by every downstream phase so it lives here once and is imported everywhere.
"""

import re
import string
import os

# Keep the service fast and offline-safe. Importing NLTK loads a large module
# graph, so its corpus is used only when explicitly requested.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "of", "on", "or", "that", "the", "to", "with", "this",
    "these", "those", "it", "its", "was", "were", "will", "can", "have",
    "has", "had", "your", "our", "you", "we", "their", "they", "not",
}


def _get_stopwords() -> set[str]:
    """Optionally use the larger NLTK list without making it a startup cost."""
    if os.getenv("RESUMEIQ_USE_NLTK_STOPWORDS", "").lower() not in {"1", "true", "yes"}:
        return _STOPWORDS
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except (ImportError, LookupError):
        return _STOPWORDS

# Load spaCy's small English model for lemmatization.
# Run `python -m spacy download en_core_web_sm` once before using this module.
_nlp = None


def _get_nlp():
    """Load spaCy only when explicitly enabled for training-quality lemmatization."""
    global _nlp
    if _nlp is None and os.getenv("RESUMEIQ_USE_SPACY", "").lower() in {"1", "true", "yes"}:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            _nlp.max_length = 2_000_000
        except (ImportError, OSError):
            _nlp = False
    return _nlp if _nlp is not False else None
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str, max_words: int = 400) -> str:
    """
    Full cleaning pipeline:
      1. Lowercase
      2. Remove URLs, emails, phone numbers
      3. Strip punctuation and digits
      4. Collapse whitespace
      5. Truncate to max_words (speeds up lemmatization on long resumes)
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove phone numbers (basic patterns)
    text = re.sub(r"\+?\d[\d\s\-().]{7,}\d", " ", text)
    # Remove punctuation
    text = text.translate(_PUNCT_TABLE)
    # Remove standalone digits (years/numbers that add noise)
    text = re.sub(r"\b\d+\b", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate — first 400 words carry most of the signal
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])

    return text


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Filter out English stopwords from a token list."""
    stopwords = _get_stopwords()
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def lemmatize(tokens: list[str]) -> list[str]:
    """
    Lemmatize a list of tokens using spaCy.
    Joins tokens → spaCy doc → extracts lemmas to preserve spaCy's context.
    """
    nlp = _get_nlp()
    if nlp is None:
        return tokens
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc if not token.is_space]


def preprocess(text: str) -> str:
    """
    Full pipeline: clean → tokenize → remove stopwords → lemmatize → rejoin.
    Returns a single cleaned string suitable for TF-IDF or embedding input.
    """
    cleaned = clean_text(text)
    tokens = cleaned.split()
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def preprocess_batch(texts: list[str], batch_size: int = 256) -> list[str]:
    """
    Batch version using spaCy's nlp.pipe() — significantly faster than
    calling preprocess() one-at-a-time on large corpora.
    """
    cleaned = [clean_text(t) for t in texts]
    nlp = _get_nlp()
    if nlp is None:
        return [" ".join(remove_stopwords(text.split())) for text in cleaned]
    results = []
    for doc in nlp.pipe(cleaned, batch_size=batch_size):
        tokens = [
            token.lemma_ for token in doc
            if not token.is_space and token.text not in _get_stopwords() and len(token.text) > 1
        ]
        results.append(" ".join(tokens))
    return results
