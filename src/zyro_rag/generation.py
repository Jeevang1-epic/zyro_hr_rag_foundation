from __future__ import annotations

import re

from .retrieval import RetrievalResult


STOP_WORDS = {
    "about",
    "after",
    "does",
    "employee",
    "employees",
    "from",
    "have",
    "that",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def split_sentences(text: str) -> list[str]:
    body = re.sub(r"(?m)^#{1,6}\s+.*(?:\n+|$)", "", text.strip()).strip()
    pieces = re.split(r"(?<=[.!?])\s+", body)
    return [piece.strip() for piece in pieces if piece.strip()]


def keyword_set(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower())
    return {word for word in words if len(word) > 2 and word not in STOP_WORDS}


def extractive_answer(question: str, results: list[RetrievalResult], max_chars: int = 700) -> str:
    fallback = "The available HR policy documents do not contain enough information to answer this."
    if not results:
        return fallback

    best_text = results[0].text.strip()
    if not best_text:
        return fallback

    question_terms = keyword_set(question)
    sentences = split_sentences(best_text)
    if not sentences:
        return best_text[:max_chars].strip() or fallback

    scored: list[tuple[int, int, str]] = []
    for sentence in sentences:
        overlap = len(question_terms & keyword_set(sentence))
        scored.append((overlap, len(sentence), sentence))

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    chosen = [sentence for overlap, _, sentence in scored[:4] if overlap > 0]
    if not chosen:
        chosen = sentences[:3]

    answer = " ".join(chosen).strip()
    return answer[:max_chars].strip() or best_text[:max_chars].strip() or fallback
