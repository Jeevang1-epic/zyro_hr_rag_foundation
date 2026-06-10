from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .chunking import chunk_documents
from .config import RagConfig
from .generation import extractive_answer
from .loaders import load_documents
from .retrieval import HybridTfidfRetriever
from .schema import detect_question_file, detect_sample_submission, detect_schema
from .submission import build_submission, save_submission
from .validate_submission import validate_submission


def run_pipeline(config: RagConfig) -> dict:
    input_dir = config.input_dir
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(input_dir.rglob("*.csv"))
    sample_path = detect_sample_submission(csv_paths)
    question_path = detect_question_file(csv_paths, sample_path=sample_path)

    if sample_path is None:
        raise FileNotFoundError("Could not detect sample submission CSV.")
    if question_path is None:
        raise FileNotFoundError("Could not detect question/test CSV.")

    sample_df = pd.read_csv(sample_path)
    question_df = pd.read_csv(question_path)
    schema = detect_schema(sample_df, question_df)

    documents = load_documents(input_dir)
    if not documents:
        raise FileNotFoundError("No policy documents were loaded from the input directory.")

    chunks = chunk_documents(documents, config.chunk_size, config.chunk_overlap)
    if not chunks:
        raise ValueError("Policy documents loaded, but chunking produced zero chunks.")

    retriever = HybridTfidfRetriever(chunks)

    answers: list[str] = []
    debug_rows: list[dict] = []
    for row_index, row in question_df.iterrows():
        question = str(row[schema["question_col"]])
        results = retriever.search(question, top_k=config.top_k)
        answer = extractive_answer(question, results, max_chars=config.max_answer_chars)
        answers.append(answer)

        question_id = row[schema["id_col"]] if schema["id_col"] in question_df.columns else row_index
        for result in results:
            debug_rows.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "rank": result.rank,
                    "score": result.score,
                    "source": result.source,
                    "chunk_id": result.chunk_id,
                    "chunk_preview": result.text[:500],
                }
            )

    submission = build_submission(sample_df, question_df, answers, schema["answer_col"])
    submission_path = output_dir / "submission.csv"
    save_submission(submission, submission_path)

    validation = validate_submission(sample_df, submission, schema["answer_col"], submission_path)

    retrieval_debug_path = output_dir / "retrieval_debug.csv"
    pd.DataFrame(debug_rows).to_csv(retrieval_debug_path, index=False)

    run_log_path = output_dir / "run_log.json"
    run_log = {
        "sample_submission_path": str(sample_path),
        "test_questions_path": str(question_path),
        "output_dir": str(output_dir),
        "submission_path": str(submission_path),
        "retrieval_debug_path": str(retrieval_debug_path),
        "run_log_path": str(run_log_path),
        "num_questions": int(len(question_df)),
        "num_documents": int(len(documents)),
        "num_chunks": int(len(chunks)),
        "chunk_size": int(config.chunk_size),
        "chunk_overlap": int(config.chunk_overlap),
        "top_k": int(config.top_k),
        **schema,
        "validation": validation,
    }
    run_log_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")

    if not validation["ok"]:
        raise AssertionError(validation["errors"])

    return run_log
