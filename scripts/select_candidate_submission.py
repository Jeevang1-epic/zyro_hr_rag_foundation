from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generate_score_recovery_candidates import (  # noqa: E402
    CANDIDATE_SUBMISSION_FILES,
    normalize_candidate,
    select_candidate,
)
from zyro_rag.live_pipeline import validate_live_submission  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        allowed = ", ".join(CANDIDATE_SUBMISSION_FILES)
        raise SystemExit(f"Usage: python scripts/select_candidate_submission.py <{allowed}>")

    candidate = normalize_candidate(sys.argv[1])
    candidate_path = PROJECT_ROOT / CANDIDATE_SUBMISSION_FILES[candidate]
    if not candidate_path.exists():
        raise SystemExit(f"Candidate file does not exist: {candidate_path}")

    destination = select_candidate(candidate)
    validation = validate_live_submission(pd.read_csv(destination))
    if not validation["ready_for_submission"]:
        raise SystemExit(f"Selected candidate failed validation: {validation}")

    print(f"Selected candidate {candidate}")
    print(f"submission={destination}")
    print("FINAL SUBMISSION VALIDATION: PASS")


if __name__ == "__main__":
    main()
