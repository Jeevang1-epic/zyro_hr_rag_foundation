from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_submission(
    sample_submission: pd.DataFrame,
    question_df: pd.DataFrame,
    answers: list[str],
    answer_col: str,
) -> pd.DataFrame:
    if len(question_df) != len(sample_submission):
        raise ValueError(
            f"Question row count {len(question_df)} does not match sample rows {len(sample_submission)}"
        )
    if len(answers) != len(sample_submission):
        raise ValueError(f"Answer count {len(answers)} does not match sample rows {len(sample_submission)}")
    if answer_col not in sample_submission.columns:
        raise ValueError(f"Answer column '{answer_col}' is not present in sample submission")

    submission = sample_submission.copy()
    submission[answer_col] = [str(answer).strip() for answer in answers]
    return submission


def save_submission(submission: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
