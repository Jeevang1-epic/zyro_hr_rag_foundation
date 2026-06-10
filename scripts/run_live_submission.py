from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

import sys

sys.path.insert(0, str(SRC_DIR))

from zyro_rag.live_pipeline import (  # noqa: E402
    LANGSMITH_DRAFT_LINK,
    STREAMLIT_DRAFT_LINK,
    answer_questions,
    chunk_pdf_documents,
    decode_questions,
    detect_dataset_folder,
    generate_submission,
    load_pdf_documents,
    read_sample_submission,
    validate_live_submission,
    write_pdf_debug,
)


def sample_question_enc_by_id(sample: pd.DataFrame) -> tuple[dict[str, str] | None, str]:
    required = {"question_id", "question_enc"}
    if not required.issubset(sample.columns):
        return None, "not_available"

    question_enc = sample["question_enc"].astype(str).str.strip()
    if not question_enc.str.len().gt(0).all():
        return None, "sample_empty"

    unique_count = question_enc.nunique(dropna=False)
    if unique_count != len(sample):
        return None, "sample_placeholder_or_repeated_values"

    return dict(zip(sample["question_id"].astype(str), question_enc)), "sample_submission"


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_folder = detect_dataset_folder(PROJECT_ROOT / "datasets")
    pdf_paths = sorted(dataset_folder.glob("*.pdf"))
    sample_paths = sorted(dataset_folder.glob("sample_submission.*"))
    notebook_path = dataset_folder / "Starter_Notebook.ipynb"

    if not sample_paths:
        raise FileNotFoundError("sample_submission.* not found in detected dataset folder.")
    sample_path = sample_paths[0]
    sample = read_sample_submission(sample_path)
    question_enc_by_id, question_enc_source = sample_question_enc_by_id(sample)

    fernet, questions, encrypted_question_pairs = decode_questions(notebook_path)
    documents = load_pdf_documents(dataset_folder)
    failures = [doc for doc in documents if doc.error or not doc.text.strip()]
    write_pdf_debug(documents, output_dir / "pdf_extraction_debug.md")
    if failures:
        details = ", ".join(f"{doc.source}: {doc.error or 'empty extraction'}" for doc in failures)
        raise RuntimeError(f"PDF extraction failed: {details}")

    chunks = chunk_pdf_documents(documents, chunk_size=900, chunk_overlap=150)
    answers = answer_questions(questions, chunks, top_k=6)

    submission = generate_submission(fernet, answers, question_enc_by_id=question_enc_by_id)
    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    retrieval_rows: list[dict] = []
    answer_rows: list[dict] = []
    for answer in answers:
        top_sources = []
        for result in answer.results:
            if result.source not in top_sources:
                top_sources.append(result.source)
            retrieval_rows.append(
                {
                    "question_id": answer.question_id,
                    "question_preview_or_decoded_if_available": answer.question,
                    "rank": result.rank,
                    "score": result.score,
                    "source": result.source,
                    "chunk_preview": result.text[:500],
                }
            )

        answer_rows.append(
            {
                "question_id": answer.question_id,
                "question_preview_or_decoded_if_available": answer.question,
                "answer_preview": answer.answer,
                "top_sources": "; ".join(top_sources[:4]),
                "is_refusal": answer.is_refusal,
                "needs_review": answer.needs_review,
            }
        )

    pd.DataFrame(retrieval_rows).to_csv(output_dir / "retrieval_debug_live.csv", index=False)
    pd.DataFrame(answer_rows).to_csv(output_dir / "answers_preview_live.csv", index=False)

    read_back = pd.read_csv(submission_path)
    validation = validate_live_submission(read_back)

    run_log = {
        "status": "PASS" if validation["ready_for_submission"] else "DRAFT_ONLY",
        "dataset_folder": str(dataset_folder),
        "pdf_count": len(pdf_paths),
        "pdf_files": [path.name for path in pdf_paths],
        "sample_submission_path": str(sample_path),
        "sample_shape": list(sample.shape),
        "sample_columns": sample.columns.tolist(),
        "question_enc_source": question_enc_source,
        "starter_notebook_path": str(notebook_path),
        "questions_loaded": len(questions),
        "encrypted_question_pairs_loaded": len(encrypted_question_pairs),
        "pdfs_loaded": len(documents),
        "chunks_created": len(chunks),
        "chunk_size": 900,
        "chunk_overlap": 150,
        "optional_next_chunk_config": {"chunk_size": 1200, "chunk_overlap": 200},
        "retrieval_method": "hybrid TF-IDF word + char n-gram cosine with source/title boosting",
        "top_k": 6,
        "llm_used": False,
        "fallback_used": True,
        "langsmith_tracing_enabled": bool(
            os.getenv("LANGCHAIN_TRACING_V2") == "true"
            and (os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY"))
        ),
        "streamlit_link_value": submission["streamlit_link"].iloc[0],
        "langsmith_link_value": submission["langsmith_link"].iloc[0],
        "draft_link_placeholders": {
            "streamlit": STREAMLIT_DRAFT_LINK,
            "langsmith": LANGSMITH_DRAFT_LINK,
        },
        "validation": validation,
    }
    (output_dir / "run_log_live.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")

    print("Live local dataset run complete.")
    print(f"dataset_folder={dataset_folder}")
    print(f"pdfs_loaded={len(documents)}")
    print(f"chunks_created={len(chunks)}")
    print(f"questions_loaded={len(questions)}")
    print(f"submission={submission_path}")
    print(f"status={run_log['status']}")
    print(f"question_enc_source={question_enc_source}")
    print(f"streamlit_ready={validation['streamlit_link_ready']}")
    print(f"langsmith_ready={validation['langsmith_link_ready']}")
    if validation["ready_for_submission"]:
        print("SUBMISSION READY FOR KAGGLE UPLOAD.")
    if not validation["ready_for_submission"]:
        print("DRAFT_ONLY - DO NOT SUBMIT YET. STREAMLIT/LANGSMITH LINKS REQUIRED.")


if __name__ == "__main__":
    main()
