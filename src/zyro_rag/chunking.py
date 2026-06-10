from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        start = end - chunk_overlap
    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_index, doc in enumerate(documents):
        source = str(doc.get("source", f"document_{doc_index}"))
        text = str(doc.get("text", ""))
        for chunk_index, chunk in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"doc{doc_index:04d}_chunk{chunk_index:04d}",
                    source=source,
                    text=chunk,
                )
            )
    return chunks
