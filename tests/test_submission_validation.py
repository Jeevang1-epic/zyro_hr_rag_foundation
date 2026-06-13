from pathlib import Path

import pandas as pd

from zyro_rag.submission import build_submission, save_submission
from zyro_rag.validate_submission import validate_submission


def test_submission_validation_ok(tmp_path: Path):
    sample = pd.DataFrame({"id": [1, 2], "answer": ["", ""]})
    questions = pd.DataFrame({"id": [1, 2], "question": ["q1", "q2"]})
    submission = build_submission(sample, questions, ["a1", "a2"], "answer")
    output_path = tmp_path / "submission.csv"
    save_submission(submission, output_path)

    result = validate_submission(sample, submission, "answer", output_path)

    assert result["ok"]
    assert result["columns_match_sample"]
    assert result["row_count_matches_sample"]
    assert result["no_empty_answers"]
    assert result["no_unnamed_0"]
    assert result["read_back_validation"]


def test_submission_validation_rejects_empty_answers():
    sample = pd.DataFrame({"id": [1], "answer": [""]})
    submission = pd.DataFrame({"id": [1], "answer": [" "]})

    result = validate_submission(sample, submission, "answer")

    assert not result["ok"]
    assert not result["no_empty_answers"]
