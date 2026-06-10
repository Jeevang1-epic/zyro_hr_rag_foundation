from __future__ import annotations

from pathlib import Path

import pandas as pd


def _non_empty(series: pd.Series) -> bool:
    return bool(series.notna().all() and series.astype(str).str.strip().str.len().gt(0).all())


def validate_submission(
    sample_submission: pd.DataFrame,
    submission: pd.DataFrame,
    answer_col: str,
    output_path: Path | None = None,
) -> dict:
    columns_match = list(submission.columns) == list(sample_submission.columns)
    row_count_matches = submission.shape[0] == sample_submission.shape[0]
    no_unnamed_0 = "Unnamed: 0" not in submission.columns
    answer_col_present = answer_col in submission.columns
    no_empty_answers = answer_col_present and _non_empty(submission[answer_col])
    read_back_validation = output_path is None

    errors: list[str] = []
    if not columns_match:
        errors.append("Submission columns do not exactly match sample submission columns.")
    if not row_count_matches:
        errors.append("Submission row count does not match sample submission row count.")
    if not no_unnamed_0:
        errors.append("Submission contains accidental index column 'Unnamed: 0'.")
    if not answer_col_present:
        errors.append(f"Answer column '{answer_col}' not found.")
    elif not no_empty_answers:
        errors.append("Answer column contains null or empty values.")

    if output_path is not None:
        if not output_path.exists():
            errors.append(f"Output file does not exist: {output_path}")
            read_back_validation = False
        else:
            try:
                reread = pd.read_csv(output_path)
                read_back_validation = (
                    list(reread.columns) == list(sample_submission.columns)
                    and reread.shape[0] == sample_submission.shape[0]
                    and "Unnamed: 0" not in reread.columns
                    and answer_col in reread.columns
                    and _non_empty(reread[answer_col])
                )
                if not read_back_validation:
                    errors.append("Read-back CSV validation failed.")
            except Exception as exc:
                errors.append(f"Could not read output CSV: {exc}")
                read_back_validation = False

    return {
        "ok": not errors,
        "errors": errors,
        "columns_match_sample": columns_match,
        "row_count_matches_sample": row_count_matches,
        "no_empty_answers": no_empty_answers,
        "no_unnamed_0": no_unnamed_0,
        "read_back_validation": read_back_validation,
    }
