# Zyro Dynamics HR Help Desk RAG Foundation

This repository is a mock-data-only foundation for the Zyro Dynamics HR Help Desk RAG Challenge. It prepares the local package structure, retrieval pipeline, baseline answer generation, submission creation, and validation before the real Kaggle dataset is available.

## Current Scope

- Load mock HR policy documents from `sample_mock/input/policies`.
- Load mock test questions and a mock sample submission.
- Chunk policy text with overlap.
- Retrieve relevant chunks with local TF-IDF only.
- Generate extractive grounded answers from retrieved chunks.
- Write and validate local output files under `outputs/`.
- Run unit tests without external APIs or real Kaggle data.

## Out of Scope for Prompt 01

- Real Kaggle data.
- External LLM APIs.
- Streamlit or other UI work.
- Prompt 02 or Prompt 03 migration/scoring work.

## Local Setup

```bash
pip install -r requirements.txt
python scripts/run_local_smoke.py
pytest -q
```

The smoke run creates:

```text
outputs/submission.csv
outputs/retrieval_debug.csv
outputs/run_log.json
```

`outputs/` is intentionally ignored by git.

## Kaggle Migration Later

When the real competition dataset opens, inspect `/kaggle/input`, detect the real sample submission and question files, adapt paths if needed, and write the final submission to `/kaggle/working/submission.csv`.
