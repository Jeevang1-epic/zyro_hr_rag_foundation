from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    source: str
    text: str
    score: float
    rank: int


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


class HybridTfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Cannot build retriever with zero chunks")

        self.chunks = chunks
        self.texts = [chunk.text for chunk in chunks]
        self.word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.word_matrix = self.word_vectorizer.fit_transform(self.texts)
        self.char_matrix = self.char_vectorizer.fit_transform(self.texts)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            return []

        normalized = normalize_query(query)
        word_query = self.word_vectorizer.transform([normalized])
        char_query = self.char_vectorizer.transform([normalized])

        word_scores = cosine_similarity(word_query, self.word_matrix).ravel()
        char_scores = cosine_similarity(char_query, self.char_matrix).ravel()
        blended_scores = (0.60 * word_scores) + (0.40 * char_scores)

        limit = min(top_k, len(self.chunks))
        top_indices = np.argsort(blended_scores)[::-1][:limit]

        results: list[RetrievalResult] = []
        for rank, index in enumerate(top_indices, start=1):
            idx = int(index)
            chunk = self.chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    text=chunk.text,
                    score=float(blended_scores[idx]),
                    rank=rank,
                )
            )
        return results
