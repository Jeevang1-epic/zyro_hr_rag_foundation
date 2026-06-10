from __future__ import annotations

from pathlib import Path

import pandas as pd


ID_CANDIDATES = ["id", "question_id", "qid", "row_id", "case_id", "ticket_id"]
QUESTION_CANDIDATES = ["question", "query", "prompt", "employee_question", "ask", "input"]
ANSWER_CANDIDATES = ["answer", "response", "prediction", "output", "generated_answer"]


def find_first_column(columns, candidates: list[str]):
    lower_map = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for col in columns:
        col_lower = str(col).lower()
        if any(candidate.lower() in col_lower for candidate in candidates):
            return col
    return None


def detect_sample_submission(csv_paths: list[Path]) -> Path | None:
    for path in csv_paths:
        name = path.name.lower()
        if "sample" in name and "submission" in name:
            return path

    scored: list[tuple[int, Path]] = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path, nrows=5)
        except Exception:
            continue

        score = 0
        name = path.name.lower()
        if "submission" in name:
            score += 10
        if find_first_column(df.columns, ANSWER_CANDIDATES) is not None:
            score += 6
        if find_first_column(df.columns, ID_CANDIDATES) is not None:
            score += 2
        if find_first_column(df.columns, QUESTION_CANDIDATES) is not None:
            score -= 5
        scored.append((score, path))

    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return scored[0][1] if scored[0][0] > 0 else None


def detect_question_file(csv_paths: list[Path], sample_path: Path | None = None) -> Path | None:
    scored: list[tuple[int, Path]] = []
    for path in csv_paths:
        if sample_path is not None and path.resolve() == sample_path.resolve():
            continue
        try:
            df = pd.read_csv(path, nrows=5)
        except Exception:
            continue

        score = 0
        name = path.name.lower()
        if any(key in name for key in ["test", "question", "queries"]):
            score += 5
        if find_first_column(df.columns, QUESTION_CANDIDATES) is not None:
            score += 10
        if find_first_column(df.columns, ID_CANDIDATES) is not None:
            score += 2
        if find_first_column(df.columns, ANSWER_CANDIDATES) is not None:
            score -= 3
        scored.append((score, path))

    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return scored[0][1] if scored[0][0] > 0 else None


def detect_schema(sample_df: pd.DataFrame, question_df: pd.DataFrame) -> dict:
    id_col = find_first_column(question_df.columns, ID_CANDIDATES)
    question_col = find_first_column(question_df.columns, QUESTION_CANDIDATES)
    answer_col = find_first_column(sample_df.columns, ANSWER_CANDIDATES)

    if answer_col is None:
        non_id_cols = [
            col
            for col in sample_df.columns
            if str(col).lower() not in {candidate.lower() for candidate in ID_CANDIDATES}
        ]
        answer_col = non_id_cols[-1] if non_id_cols else sample_df.columns[-1]

    if question_col is None:
        text_cols = [col for col in question_df.columns if question_df[col].dtype == "object"]
        question_col = text_cols[0] if text_cols else question_df.columns[-1]

    return {"id_col": id_col, "question_col": question_col, "answer_col": answer_col}
