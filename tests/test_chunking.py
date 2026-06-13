import pytest

from zyro_rag.chunking import chunk_text


def test_chunk_text_overlap():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert chunks[0][-50:] == chunks[1][:50]


def test_chunk_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, chunk_overlap=100)
