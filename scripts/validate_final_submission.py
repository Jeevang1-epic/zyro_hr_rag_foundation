from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLS = [
    "question_id",
    "question_enc",
    "answer_enc",
    "streamlit_link",
    "langsmith_link",
]


def main() -> None:
    path = Path("outputs/submission.csv")
    assert path.exists(), "outputs/submission.csv missing"

    submission = pd.read_csv(path)

    assert list(submission.columns) == REQUIRED_COLS, "Wrong columns"
    assert submission.shape[0] == 15, "Wrong row count"
    assert submission["question_id"].tolist() == [f"Q{i:02d}" for i in range(1, 16)], "Wrong question IDs"
    assert submission["question_enc"].astype(str).str.strip().str.len().gt(0).all(), "Empty question_enc"
    assert submission["answer_enc"].astype(str).str.strip().str.len().gt(0).all(), "Empty answer_enc"
    assert submission["streamlit_link"].astype(str).str.startswith("https://").all(), "Streamlit link missing"
    assert submission["langsmith_link"].astype(str).str.startswith("https://").all(), "LangSmith link missing"
    assert "Unnamed: 0" not in submission.columns, "Accidental index column"

    print("FINAL SUBMISSION VALIDATION: PASS")


if __name__ == "__main__":
    main()
