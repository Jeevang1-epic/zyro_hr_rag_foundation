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


def ready_question_enc_by_id(path: Path) -> tuple[dict[str, str] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        submission = pd.read_csv(path)
    except Exception:
        return None, None

    required = {"question_id", "question_enc"}
    expected_ids = [f"Q{i:02d}" for i in range(1, 16)]
    if not required.issubset(submission.columns):
        return None, None
    if submission["question_id"].astype(str).tolist() != expected_ids:
        return None, None

    question_enc = submission["question_enc"].astype(str).str.strip()
    if not question_enc.str.len().gt(0).all():
        return None, None
    if question_enc.nunique(dropna=False) != len(submission):
        return None, None

    return dict(zip(submission["question_id"].astype(str), question_enc)), str(path)


def resolve_question_enc_by_id(output_dir: Path, sample: pd.DataFrame) -> tuple[dict[str, str] | None, str]:
    for path in [
        output_dir / "submission.csv",
        output_dir / "submission_score_92_71_backup.csv",
        output_dir / "submission_score_91_72_backup.csv",
    ]:
        question_enc_by_id, source = ready_question_enc_by_id(path)
        if question_enc_by_id:
            return question_enc_by_id, source or "existing_submission"

    return sample_question_enc_by_id(sample)


def read_ready_links(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    try:
        submission = pd.read_csv(path)
    except Exception:
        return None, None

    required = {"streamlit_link", "langsmith_link"}
    if not required.issubset(submission.columns) or submission.empty:
        return None, None

    streamlit_link = str(submission["streamlit_link"].iloc[0]).strip()
    langsmith_link = str(submission["langsmith_link"].iloc[0]).strip()
    if streamlit_link.startswith("https://") and langsmith_link.startswith("https://"):
        return streamlit_link, langsmith_link
    return None, None


def resolve_submission_links(output_dir: Path) -> tuple[str | None, str | None, str]:
    streamlit_link = os.getenv("STREAMLIT_APP_URL")
    langsmith_link = os.getenv("LANGSMITH_TRACE_URL")
    if streamlit_link and langsmith_link:
        return streamlit_link.strip(), langsmith_link.strip(), "environment"

    for source, path in [
        ("existing_submission", output_dir / "submission.csv"),
        ("score_92_71_backup", output_dir / "submission_score_92_71_backup.csv"),
        ("score_91_72_backup", output_dir / "submission_score_91_72_backup.csv"),
    ]:
        streamlit_link, langsmith_link = read_ready_links(path)
        if streamlit_link and langsmith_link:
            return streamlit_link, langsmith_link, source

    return None, None, "draft_placeholders"


def langsmith_tracing_enabled() -> bool:
    tracing_flag = (
        os.getenv("LANGSMITH_TRACING", "").lower() == "true"
        or os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    )
    has_key = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
    return tracing_flag and has_key


def build_answer_preview_rows(answers) -> list[dict]:
    answer_rows: list[dict] = []
    for answer in answers:
        top_sources: list[str] = []
        for result in answer.results:
            if result.source not in top_sources:
                top_sources.append(result.source)

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
    return answer_rows


def main() -> None:
    from generate_score_recovery_candidates import generate_score_recovery_candidates

    generate_score_recovery_candidates(selected_candidate="E")
    return

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
    question_enc_by_id, question_enc_source = resolve_question_enc_by_id(output_dir, sample)

    fernet, questions, encrypted_question_pairs = decode_questions(notebook_path)
    documents = load_pdf_documents(dataset_folder)
    failures = [doc for doc in documents if doc.error or not doc.text.strip()]
    write_pdf_debug(documents, output_dir / "pdf_extraction_debug.md")
    if failures:
        details = ", ".join(f"{doc.source}: {doc.error or 'empty extraction'}" for doc in failures)
        raise RuntimeError(f"PDF extraction failed: {details}")

    chunks = chunk_pdf_documents(documents, chunk_size=1200, chunk_overlap=250)
    candidate_answers_by_variant = {}
    for variant in ["A", "B", "C"]:
        candidate_answers = answer_questions(questions, chunks, top_k=8, answer_variant=variant)
        candidate_answers_by_variant[variant] = candidate_answers
        pd.DataFrame(build_answer_preview_rows(candidate_answers)).to_csv(
            output_dir / f"candidate_{variant}_answers_preview.csv",
            index=False,
        )

    selected_answer_variant = "C"
    answers = candidate_answers_by_variant[selected_answer_variant]

    streamlit_link, langsmith_link, link_source = resolve_submission_links(output_dir)
    submission = generate_submission(
        fernet,
        answers,
        streamlit_link=streamlit_link,
        langsmith_link=langsmith_link,
        question_enc_by_id=question_enc_by_id,
    )
    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    retrieval_rows: list[dict] = []
    for answer in answers:
        for result in answer.results:
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

    pd.DataFrame(retrieval_rows).to_csv(output_dir / "retrieval_debug_live.csv", index=False)
    pd.DataFrame(build_answer_preview_rows(answers)).to_csv(output_dir / "answers_preview_live.csv", index=False)

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
        "chunk_size": 1200,
        "chunk_overlap": 250,
        "previous_chunk_config": {"chunk_size": 900, "chunk_overlap": 150},
        "retrieval_method": "hybrid TF-IDF word + char n-gram cosine with source/title boosting",
        "top_k": 8,
        "candidate_answer_variants_generated": ["A", "B", "C"],
        "selected_answer_variant": selected_answer_variant,
        "llm_used": False,
        "fallback_used": True,
        "langsmith_tracing_enabled": langsmith_tracing_enabled(),
        "streamlit_link_value": submission["streamlit_link"].iloc[0],
        "langsmith_link_value": submission["langsmith_link"].iloc[0],
        "link_source": link_source,
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
    print(f"link_source={link_source}")
    print(f"streamlit_ready={validation['streamlit_link_ready']}")
    print(f"langsmith_ready={validation['langsmith_link_ready']}")
    if validation["ready_for_submission"]:
        print("SUBMISSION READY FOR KAGGLE UPLOAD.")
    if not validation["ready_for_submission"]:
        print("DRAFT_ONLY - DO NOT SUBMIT YET. STREAMLIT/LANGSMITH LINKS REQUIRED.")


if __name__ == "__main__":
    main()
