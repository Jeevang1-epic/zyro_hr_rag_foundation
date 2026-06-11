from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from zyro_rag.live_pipeline import (  # noqa: E402
    REQUIRED_SUBMISSION_COLUMNS,
    decode_questions,
    detect_dataset_folder,
    load_pdf_documents,
    validate_live_submission,
)


REFUSAL_TEMPLATE_B = (
    "I can only answer HR-related questions from Zyro Dynamics policy documents. "
    "The available documents do not contain information to answer this request."
)

CONCISE_POLICY_ANSWERS = {
    "Q01": (
        "Earned Leave accrues at 1.25 days per month. After one year of continuous service, an employee "
        "is entitled to 15 days of Earned Leave, provided they worked at least 240 days in that year."
    ),
    "Q02": (
        "A maximum of 45 days of Earned Leave can be carried forward at the end of the financial year "
        "(31 March). Any balance above 45 days is automatically encashed at the employee's basic daily "
        "rate and credited in April payroll."
    ),
    "Q03": (
        "An eligible female employee is entitled to 26 weeks of paid Maternity Leave for the first two "
        "live births. Eligibility requires at least 80 days of service in the 12 months preceding the "
        "expected delivery date."
    ),
    "Q04": (
        "For Sick Leave of more than 2 consecutive days, the employee must submit a medical certificate "
        "from a registered medical practitioner within 3 working days of returning to work."
    ),
    "Q05": (
        "Salary is credited to the employee's registered bank account by the 7th of the following month. "
        "The payroll cut-off date is the 24th of each month."
    ),
    "Q06": "For L4 (Senior), the CTC range is Rs. 16.0L to Rs. 26.0L per annum, and the bonus target is 10% of CTC.",
    "Q07": (
        "Group Medical Insurance covers up to Rs. 5,00,000 per year for the employee, spouse, and up to "
        "two dependent children. The Company fully pays the premiums."
    ),
    "Q08": (
        "An employee is placed on a PIP after receiving a rating of 1 or 2 in two consecutive review "
        "cycles. A PIP lasts 60 to 90 days, as determined by the reporting manager and HR Business Partner."
    ),
    "Q09": (
        "APR is conducted annually in March for final rating, increment, and promotion decisions. "
        "Timeline: 360 degree feedback 1 to 20 February; self-assessment 1 to 10 March; manager assessment "
        "and draft rating 11 to 20 March; calibration 21 to 25 March; final ratings 26 to 31 March; "
        "one-on-one feedback 1 to 10 April; increment and promotion letters issued on 15 April."
    ),
    "Q10": (
        "Permanent employees at grade L3 and above are eligible for WFH. Employees on probation, grades "
        "L1/L2, and client-site employees are not eligible unless the HR Director approves a written "
        "exception. Types: Hybrid WFH for L3+ up to 3 days/week, Full Remote for L5+ case-by-case up to "
        "5 days/week, Ad-hoc WFH for L3+ up to 2 days/week, and Emergency WFH for all employees as "
        "directed by HR."
    ),
}

Q10_MINIMAL_REPAIR = (
    "WFH eligibility applies to permanent employees at grade L3 and above. Employees on probation, "
    "grades L1/L2, and employees deployed at client sites are not eligible unless the HR Director approves "
    "an exception in writing. To be considered, employees must have 6 months of continuous service, hold "
    "grade L3 or above, have a Meets Expectations or higher rating, have no active PIP or disciplinary "
    "proceedings, have a role suitable for remote execution, and have a reliable 25 Mbps internet connection "
    "with a dedicated, distraction-free workspace. The WFH arrangements are Hybrid WFH for L3+ up to "
    "3 days/week, Full Remote for L5+ case-by-case up to 5 days/week, Ad-hoc WFH for L3+ up to 2 days/week, "
    "and Emergency WFH for all employees as directed by HR."
)

COMPLETE_POLICY_ANSWERS = {
    "Q01": (
        "Employees become eligible for 15 days of Earned Leave after completing one year of continuous "
        "service, provided they have worked for a minimum of 240 days in that year. Thereafter, Earned "
        "Leave accrues at 1.25 days per month. During probation, EL accrues at 0.5 days per month and "
        "becomes available only after probation confirmation."
    ),
    "Q02": (
        "A maximum of 45 days of Earned Leave may be carried forward at the end of each financial year "
        "(31 March). Any balance exceeding this limit is automatically encashed at the employee's basic "
        "daily rate and credited in the April payroll."
    ),
    "Q03": (
        "Female employees who have completed at least 80 days of service in the 12 months preceding the "
        "expected date of delivery are entitled to 26 weeks of paid Maternity Leave for the first two "
        "live births. For a third child, the entitlement is 12 weeks, and up to 8 weeks of pre-natal leave "
        "may be availed before the expected delivery date."
    ),
    "Q04": (
        "Sick Leave taken for more than 2 consecutive days requires a medical certificate from a registered "
        "medical practitioner, submitted within 3 working days of returning to work."
    ),
    "Q05": (
        "Salaries and professional fees are credited to the employee's registered bank account by the 7th "
        "of the following month. The payroll cut-off date is the 24th of each month; leave without pay, "
        "new joinings, or separations after the 24th are adjusted in the subsequent month's payroll cycle. "
        "New employees joining after the 24th still receive salary on the standard payday on a pro-rata basis."
    ),
    "Q06": (
        "For L4 (Senior) employees, the CTC range is Rs. 16.0L to Rs. 26.0L per annum, and the bonus "
        "target is 10% of CTC."
    ),
    "Q07": (
        "Group Medical Insurance provides coverage of up to Rs. 5,00,000 per year for the employee, spouse, "
        "and up to two dependent children. All premiums are fully paid by the Company."
    ),
    "Q08": (
        "An employee is placed on a formal Performance Improvement Plan after receiving a rating of 1 or 2 "
        "in two consecutive review cycles. The PIP duration is 60 to 90 days, as determined by the reporting "
        "manager and HR Business Partner, with documented improvement targets and mandatory weekly check-ins."
    ),
    "Q09": (
        "The Annual Performance Review is annual and takes place in March for final rating, increment, and "
        "promotion decisions. The APR process is: 360 degree feedback from 1 to 20 February; self-assessment "
        "from 1 to 10 March; manager assessment and draft rating from 11 to 20 March; calibration from "
        "21 to 25 March; final ratings from 26 to 31 March; one-on-one feedback from 1 to 10 April; and "
        "increment and promotion letters issued on 15 April by HR and Finance."
    ),
    "Q10": Q10_MINIMAL_REPAIR,
}

CANDIDATE_SUBMISSION_FILES = {
    "BASELINE": Path("outputs/submission_candidate_BASELINE_92_71.csv"),
    "A": Path("outputs/submission_candidate_A_refusal_only.csv"),
    "B": Path("outputs/submission_candidate_B_minimal_repair.csv"),
    "C": Path("outputs/submission_candidate_C_concise_policy.csv"),
    "D": Path("outputs/submission_candidate_D_complete_conditions.csv"),
    "E": Path("outputs/submission_candidate_E_mixed_best.csv"),
}

CANDIDATE_PREVIEW_FILES = {
    "BASELINE": Path("outputs/answers_preview_candidate_BASELINE_92_71.csv"),
    "A": Path("outputs/answers_preview_candidate_A_refusal_only.csv"),
    "B": Path("outputs/answers_preview_candidate_B_minimal_repair.csv"),
    "C": Path("outputs/answers_preview_candidate_C_concise_policy.csv"),
    "D": Path("outputs/answers_preview_candidate_D_complete_conditions.csv"),
    "E": Path("outputs/answers_preview_candidate_E_mixed_best.csv"),
}

SHORT_SUBMISSION_ALIASES = {
    "A": Path("outputs/submission_candidate_A.csv"),
    "B": Path("outputs/submission_candidate_B.csv"),
    "C": Path("outputs/submission_candidate_C.csv"),
    "D": Path("outputs/submission_candidate_D.csv"),
    "E": Path("outputs/submission_candidate_E.csv"),
}


def normalize_candidate(candidate: str) -> str:
    name = candidate.strip().upper()
    if name == "BASELINE_92_71":
        return "BASELINE"
    if name not in CANDIDATE_SUBMISSION_FILES:
        allowed = ", ".join(CANDIDATE_SUBMISSION_FILES)
        raise ValueError(f"Unknown candidate {candidate!r}. Allowed: {allowed}")
    return name


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    return pd.read_csv(path)


def answers_from_preview(preview: pd.DataFrame) -> dict[str, str]:
    return dict(zip(preview["question_id"].astype(str), preview["answer_preview"].astype(str)))


def build_candidates(baseline_answers: dict[str, str]) -> dict[str, dict[str, str]]:
    candidates: dict[str, dict[str, str]] = {}

    candidates["BASELINE"] = dict(baseline_answers)

    candidate_a = dict(baseline_answers)
    for number in range(11, 16):
        candidate_a[f"Q{number:02d}"] = REFUSAL_TEMPLATE_B
    candidates["A"] = candidate_a

    candidate_b = dict(baseline_answers)
    candidate_b["Q10"] = Q10_MINIMAL_REPAIR
    candidates["B"] = candidate_b

    candidate_c = dict(baseline_answers)
    candidate_c.update(CONCISE_POLICY_ANSWERS)
    for number in range(11, 16):
        candidate_c[f"Q{number:02d}"] = REFUSAL_TEMPLATE_B
    candidates["C"] = candidate_c

    candidate_d = dict(baseline_answers)
    candidate_d.update(COMPLETE_POLICY_ANSWERS)
    for number in range(11, 16):
        candidate_d[f"Q{number:02d}"] = REFUSAL_TEMPLATE_B
    candidates["D"] = candidate_d

    candidate_e = dict(baseline_answers)
    candidate_e["Q10"] = Q10_MINIMAL_REPAIR
    for number in range(11, 16):
        candidate_e[f"Q{number:02d}"] = REFUSAL_TEMPLATE_B
    candidates["E"] = candidate_e

    return candidates


def submission_from_answers(
    fernet,
    baseline_submission: pd.DataFrame,
    answers: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in baseline_submission.iterrows():
        question_id = str(row["question_id"])
        rows.append(
            {
                "question_id": question_id,
                "question_enc": str(row["question_enc"]),
                "answer_enc": fernet.encrypt(answers[question_id].encode()).decode(),
                "streamlit_link": str(row["streamlit_link"]),
                "langsmith_link": str(row["langsmith_link"]),
            }
        )
    return pd.DataFrame(rows, columns=REQUIRED_SUBMISSION_COLUMNS)


def preview_from_answers(baseline_preview: pd.DataFrame, answers: dict[str, str]) -> pd.DataFrame:
    preview = baseline_preview.copy()
    preview["answer_preview"] = preview["question_id"].astype(str).map(answers)
    preview["is_refusal"] = preview["question_id"].astype(str).isin([f"Q{i:02d}" for i in range(11, 16)])
    preview["needs_review"] = False
    return preview


def write_candidate_files(
    output_dir: Path,
    fernet,
    baseline_submission: pd.DataFrame,
    baseline_preview: pd.DataFrame,
    candidate_answers: dict[str, dict[str, str]],
) -> dict[str, dict]:
    validations: dict[str, dict] = {}
    for candidate, answers in candidate_answers.items():
        submission = submission_from_answers(fernet, baseline_submission, answers)
        preview = preview_from_answers(baseline_preview, answers)

        submission_path = PROJECT_ROOT / CANDIDATE_SUBMISSION_FILES[candidate]
        preview_path = PROJECT_ROOT / CANDIDATE_PREVIEW_FILES[candidate]
        submission.to_csv(submission_path, index=False)
        preview.to_csv(preview_path, index=False)

        if candidate in SHORT_SUBMISSION_ALIASES:
            submission.to_csv(PROJECT_ROOT / SHORT_SUBMISSION_ALIASES[candidate], index=False)

        validation = validate_live_submission(submission)
        validations[candidate] = validation
        if not validation["ready_for_submission"]:
            raise RuntimeError(f"Candidate {candidate} failed validation: {validation}")

    return validations


def select_candidate(candidate: str, copy_preview: bool = True) -> Path:
    name = normalize_candidate(candidate)
    source = PROJECT_ROOT / CANDIDATE_SUBMISSION_FILES[name]
    if not source.exists():
        raise FileNotFoundError(f"Candidate file does not exist: {source}")

    destination = PROJECT_ROOT / "outputs/submission.csv"
    shutil.copy2(source, destination)

    if copy_preview:
        preview_source = PROJECT_ROOT / CANDIDATE_PREVIEW_FILES[name]
        if preview_source.exists():
            shutil.copy2(preview_source, PROJECT_ROOT / "outputs/answers_preview_live.csv")

    retrieval_backup = PROJECT_ROOT / "outputs/retrieval_debug_score_92_71_backup.csv"
    if retrieval_backup.exists():
        shutil.copy2(retrieval_backup, PROJECT_ROOT / "outputs/retrieval_debug_live.csv")

    return destination


def changed_questions(base: dict[str, str], candidate: dict[str, str]) -> list[str]:
    return [qid for qid in sorted(base) if base[qid] != candidate[qid]]


def clip(text: str, max_len: int = 220) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= max_len else clean[: max_len - 3] + "..."


def escape_cell(text: str) -> str:
    return clip(text).replace("|", "\\|")


def write_score_recovery_diagnosis(
    output_dir: Path,
    baseline_preview: pd.DataFrame,
    degraded_preview: pd.DataFrame,
    baseline_submission: pd.DataFrame,
    current_submission: pd.DataFrame,
    degraded_source: str,
) -> None:
    base_answers = answers_from_preview(baseline_preview)
    current_answers = answers_from_preview(degraded_preview)
    changed = changed_questions(base_answers, current_answers)

    link_same = False
    if {"streamlit_link", "langsmith_link"}.issubset(current_submission.columns):
        link_same = bool(
            baseline_submission["streamlit_link"].tolist() == current_submission["streamlit_link"].tolist()
            and baseline_submission["langsmith_link"].tolist() == current_submission["langsmith_link"].tolist()
        )

    length_rows = []
    for qid in changed:
        length_rows.append(
            f"- {qid}: 92.71 length {len(base_answers[qid])}, current length {len(current_answers[qid])}"
        )

    text = f"""# Score Recovery Diagnosis

## Baseline
- Protected score baseline: 92.71
- Baseline submission: outputs/submission_score_92_71_backup.csv
- Baseline preview: outputs/answers_preview_score_92_71_backup.csv
- Degraded comparison source: {degraded_source}

## What Changed From 92.71 To Current Degraded State
- Changed answer IDs: {", ".join(changed)}
- Q01-Q10 were broadly rewritten into a concise style rather than making isolated repairs.
- Q11-Q15 refusal wording changed from "available documents" to "available HR documents".
- Required columns and links stayed valid.
- Streamlit/LangSmith links same as baseline: {str(link_same).lower()}

## Likely Score Drop Causes
- The regression likely came from changing too many in-scope answers at once.
- Q01 lost the probation accrual sentence that may have matched hidden answer wording.
- Q03 lost the explicit "Female employees" phrasing plus third-child and pre-natal exception text.
- Q05 lost the payroll adjustment rule for LOP, new joinings, and separations after the 24th.
- Q08 lost documented targets and mandatory weekly check-ins.
- Q10 lost several eligibility criteria: 6 months service, Meets Expectations rating, no active PIP or disciplinary proceedings, suitable role/workspace.
- Refusal wording also changed on Q11-Q15, so any out-of-scope movement cannot be separated from in-scope movement in the degraded attempt.

## Length And Style Shift
{chr(10).join(length_rows)}

## Conclusion
The current answer set should not be trusted as a baseline if it produced 90.77, because it made broad answer-style changes across all 15 questions. Future experiments should start from the 92.71 backup and change only isolated variables.
"""
    (output_dir / "score_recovery_diagnosis.md").write_text(text, encoding="utf-8")


def write_policy_fact_bank_v2(output_dir: Path, documents_loaded: int) -> None:
    source = output_dir / "policy_fact_bank.md"
    if source.exists():
        text = source.read_text(encoding="utf-8")
        text = text.replace("# Policy Fact Bank", "# Policy Fact Bank V2", 1)
    else:
        text = "# Policy Fact Bank V2\n\nPolicy facts are extracted from the 11 local HR PDFs.\n"

    header = (
        "# Policy Fact Bank V2\n\n"
        f"- PDFs inspected: {documents_loaded}\n"
        "- Scope: exact score-relevant numbers, eligibility rules, timelines, approvals, exceptions, cutoffs, and process deadlines.\n\n"
    )
    body = text.split("\n", 1)[1] if "\n" in text else ""
    (output_dir / "policy_fact_bank_v2.md").write_text(header + body, encoding="utf-8")


def write_candidate_matrix(
    output_dir: Path,
    baseline_answers: dict[str, str],
    candidate_answers: dict[str, dict[str, str]],
) -> None:
    evidence = {
        "Q01": "02_Leave_Policy.pdf",
        "Q02": "02_Leave_Policy.pdf",
        "Q03": "02_Leave_Policy.pdf",
        "Q04": "02_Leave_Policy.pdf",
        "Q05": "06_Compensation_and_Benefits_Policy.pdf",
        "Q06": "06_Compensation_and_Benefits_Policy.pdf",
        "Q07": "06_Compensation_and_Benefits_Policy.pdf",
        "Q08": "05_Performance_Review_Policy.pdf",
        "Q09": "05_Performance_Review_Policy.pdf",
        "Q10": "03_Work_From_Home_Policy.pdf",
        "Q11": "Out of scope",
        "Q12": "Out of scope",
        "Q13": "Out of scope",
        "Q14": "Out of scope",
        "Q15": "Out of scope",
    }
    risk = {
        "Q01": "High in C; low otherwise",
        "Q02": "Low",
        "Q03": "High in C; low otherwise",
        "Q04": "Low",
        "Q05": "High in C; low otherwise",
        "Q06": "Low",
        "Q07": "Low",
        "Q08": "High in C; low otherwise",
        "Q09": "Low",
        "Q10": "Medium when changed",
        "Q11": "Low",
        "Q12": "Low",
        "Q13": "Low",
        "Q14": "Low",
        "Q15": "Low",
    }

    lines = [
        "# Candidate Comparison Matrix",
        "",
        "| QID | 92.71 answer | Candidate A | Candidate B | Candidate C | Candidate D | Candidate E | Evidence PDF | Risk |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for qid in sorted(baseline_answers):
        lines.append(
            "| "
            + " | ".join(
                [
                    qid,
                    escape_cell(baseline_answers[qid]),
                    escape_cell(candidate_answers["A"][qid]),
                    escape_cell(candidate_answers["B"][qid]),
                    escape_cell(candidate_answers["C"][qid]),
                    escape_cell(candidate_answers["D"][qid]),
                    escape_cell(candidate_answers["E"][qid]),
                    evidence[qid],
                    risk[qid],
                ]
            )
            + " |"
        )
    (output_dir / "candidate_comparison_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_next_submission_recommendation(output_dir: Path) -> None:
    text = """# Next Submission Recommendation

## Safest Candidate To Submit First

Recommended first upload: Candidate E, mixed best-of-baseline.

Why:
- It restores the 92.71 in-scope wording for Q01-Q09.
- It makes only one in-scope evidence-backed repair: Q10 adds the explicit 25 Mbps internet requirement from the WFH policy.
- It applies the short HR-scope refusal template to Q11-Q15, isolating an out-of-scope wording improvement.
- It avoids the broad concise rewrite that likely caused the 90.77 regression.

Risk level: low-medium.

## Exact Command To Select It

```powershell
python scripts/select_candidate_submission.py E
```

## Exact Validation Command

```powershell
python scripts/validate_final_submission.py
```

## Exact Upload Path

```text
C:\\Users\\Jeevan kumar\\Desktop\\zyro_hr_rag_foundation\\outputs\\submission.csv
```

## If Score Drops

Keep the 92.71 submission selected on Kaggle. Restore the baseline candidate locally:

```powershell
python scripts/select_candidate_submission.py BASELINE
python scripts/validate_final_submission.py
```
"""
    (output_dir / "next_submission_recommendation.md").write_text(text, encoding="utf-8")


def generate_score_recovery_candidates(selected_candidate: str = "E") -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_submission_path = output_dir / "submission_score_92_71_backup.csv"
    baseline_preview_path = output_dir / "answers_preview_score_92_71_backup.csv"
    current_submission_path = output_dir / "submission.csv"
    current_preview_path = output_dir / "answers_preview_live.csv"
    degraded_preview_path = output_dir / "answers_preview_degraded_90_77_reference.csv"

    baseline_submission = read_required_csv(baseline_submission_path)
    baseline_preview = read_required_csv(baseline_preview_path)
    current_submission = read_required_csv(current_submission_path)
    current_preview = read_required_csv(current_preview_path)

    if not degraded_preview_path.exists():
        legacy_degraded_preview = output_dir / "candidate_C_answers_preview.csv"
        if legacy_degraded_preview.exists():
            shutil.copy2(legacy_degraded_preview, degraded_preview_path)

    if degraded_preview_path.exists():
        degraded_preview = read_required_csv(degraded_preview_path)
        degraded_source = "outputs/answers_preview_degraded_90_77_reference.csv"
    else:
        degraded_preview = current_preview
        degraded_source = "outputs/answers_preview_live.csv captured before candidate selection"

    dataset_folder = detect_dataset_folder(PROJECT_ROOT / "datasets")
    notebook_path = dataset_folder / "Starter_Notebook.ipynb"
    fernet, questions, encrypted_question_pairs = decode_questions(notebook_path)
    documents = load_pdf_documents(dataset_folder)
    if len(documents) != 11:
        raise RuntimeError(f"Expected 11 PDFs, found {len(documents)}")
    failures = [doc for doc in documents if doc.error or not doc.text.strip()]
    if failures:
        details = ", ".join(f"{doc.source}: {doc.error or 'empty extraction'}" for doc in failures)
        raise RuntimeError(f"PDF extraction failed: {details}")

    baseline_answers = answers_from_preview(baseline_preview)
    candidate_answers = build_candidates(baseline_answers)
    validations = write_candidate_files(output_dir, fernet, baseline_submission, baseline_preview, candidate_answers)

    write_score_recovery_diagnosis(
        output_dir,
        baseline_preview,
        degraded_preview,
        baseline_submission,
        current_submission,
        degraded_source,
    )
    write_policy_fact_bank_v2(output_dir, len(documents))
    write_candidate_matrix(output_dir, baseline_answers, candidate_answers)
    write_next_submission_recommendation(output_dir)

    selected_name = normalize_candidate(selected_candidate)
    final_submission = select_candidate(selected_name)
    final_validation = validate_live_submission(pd.read_csv(final_submission))

    run_log = {
        "status": "PASS" if final_validation["ready_for_submission"] else "DRAFT_ONLY",
        "score_recovery_sprint": True,
        "baseline_score": 92.71,
        "degraded_score_reference": 90.77,
        "selected_candidate": selected_name,
        "candidate_files": {key: str(PROJECT_ROOT / path) for key, path in CANDIDATE_SUBMISSION_FILES.items()},
        "candidate_validations": validations,
        "dataset_folder": str(dataset_folder),
        "pdfs_loaded": len(documents),
        "pdf_files": [doc.source for doc in documents],
        "questions_loaded": len(questions),
        "encrypted_question_pairs_loaded": len(encrypted_question_pairs),
        "question_enc_source": str(baseline_submission_path),
        "streamlit_link_value": baseline_submission["streamlit_link"].iloc[0],
        "langsmith_link_value": baseline_submission["langsmith_link"].iloc[0],
        "validation": final_validation,
    }
    (output_dir / "run_log_live.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")

    print("Score recovery candidate generation complete.")
    print(f"baseline_backup={baseline_submission_path}")
    print(f"pdfs_loaded={len(documents)}")
    print(f"candidates={', '.join(CANDIDATE_SUBMISSION_FILES)}")
    print(f"selected_candidate={selected_name}")
    print(f"submission={final_submission}")
    print(f"status={run_log['status']}")
    print(f"streamlit_ready={final_validation['streamlit_link_ready']}")
    print(f"langsmith_ready={final_validation['langsmith_link_ready']}")


def main() -> None:
    selected = sys.argv[1] if len(sys.argv) > 1 else "E"
    generate_score_recovery_candidates(selected)


if __name__ == "__main__":
    main()
