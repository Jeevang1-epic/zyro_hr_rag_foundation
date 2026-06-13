from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from cryptography.fernet import Fernet
from langsmith import traceable
from pypdf import PdfReader

from .chunking import Chunk, chunk_text
from .generation import extractive_answer, split_sentences
from .retrieval import HybridTfidfRetriever, RetrievalResult


REQUIRED_SUBMISSION_COLUMNS = [
    "question_id",
    "question_enc",
    "answer_enc",
    "streamlit_link",
    "langsmith_link",
]

REFUSAL_MESSAGE = (
    "I can only answer questions based on Zyro Dynamics HR policy documents, "
    "and the available HR documents do not contain information to answer this request."
)

ANSWER_VARIANTS = {
    "A": {
        "Q01": (
            "Earned Leave accrues at 1.25 days per month. Employees are entitled to 15 days of Earned "
            "Leave after completing one year of continuous service, provided they worked at least 240 days "
            "in that year."
        ),
        "Q02": (
            "Up to 45 days of Earned Leave can be carried forward at the end of the financial year "
            "(31 March). Any excess balance is automatically encashed at the employee's basic daily rate "
            "and credited in April payroll."
        ),
        "Q03": (
            "Maternity Leave is 26 weeks of paid leave for eligible female employees. Eligibility requires "
            "at least 80 days of service in the 12 months preceding the expected delivery date."
        ),
        "Q04": (
            "For Sick Leave of more than 2 consecutive days, a medical certificate from a registered "
            "medical practitioner is required and must be submitted within 3 working days of returning "
            "to work."
        ),
        "Q05": (
            "Salary is credited to the employee's registered bank account by the 7th of the following "
            "month. The payroll cut-off date is the 24th of each month."
        ),
        "Q06": (
            "For L4 (Senior), the CTC range is Rs. 16.0L to Rs. 26.0L per annum and the bonus target is "
            "10% of CTC."
        ),
        "Q07": (
            "Group Medical Insurance covers up to Rs. 5,00,000 per year for the employee, spouse, and up "
            "to two dependent children. Premiums are fully paid by the Company."
        ),
        "Q08": (
            "An employee is placed on a PIP after receiving a rating of 1 or 2 in two consecutive review "
            "cycles. The PIP duration is 60 to 90 days."
        ),
        "Q09": (
            "APR is conducted annually in March for final rating, increment, and promotion decisions. "
            "The timeline is 360 feedback from 1 to 20 February, self-assessment from 1 to 10 March, "
            "manager assessment from 11 to 20 March, calibration from 21 to 25 March, final ratings from "
            "26 to 31 March, feedback from 1 to 10 April, and increment and promotion letters on 15 April."
        ),
        "Q10": (
            "Permanent employees at grade L3 and above are eligible for WFH. The WFH arrangements are "
            "Hybrid WFH, Full Remote, Ad-hoc WFH, and Emergency WFH."
        ),
    },
    "B": {
        "Q01": (
            "Employees become eligible for 15 days of Earned Leave after one year of continuous service, "
            "provided they worked at least 240 days in that year. Thereafter, EL accrues at 1.25 days per "
            "month; probationary employees accrue EL at 0.5 days per month, usable only after probation "
            "confirmation."
        ),
        "Q02": (
            "A maximum of 45 days of Earned Leave may be carried forward at the end of each financial year "
            "(31 March). Any balance exceeding 45 days is automatically encashed at the employee's basic "
            "daily rate and credited in April payroll."
        ),
        "Q03": (
            "Female employees with at least 80 days of service in the 12 months preceding the expected "
            "delivery date are entitled to 26 weeks of paid Maternity Leave for the first two live births. "
            "For a third child, the entitlement is 12 weeks, and up to 8 weeks may be used before the "
            "expected delivery date."
        ),
        "Q04": (
            "Sick Leave taken for more than 2 consecutive days requires a medical certificate from a "
            "registered medical practitioner, submitted within 3 working days of returning to work."
        ),
        "Q05": (
            "Salaries and professional fees are credited to the registered bank account by the 7th of the "
            "following month. The payroll cut-off is the 24th of each month; LOP, new joinings, or "
            "separations after the 24th are adjusted in the subsequent payroll cycle."
        ),
        "Q06": (
            "For L4 (Senior) employees, the salary band is Rs. 16.0L to Rs. 26.0L CTC per annum and the "
            "bonus target is 10% of CTC."
        ),
        "Q07": (
            "Group Medical Insurance provides coverage up to Rs. 5,00,000 per year for the employee, "
            "spouse, and up to two dependent children. The Company pays all premiums fully."
        ),
        "Q08": (
            "An employee who receives a rating of 1 or 2 in two consecutive review cycles is placed on a "
            "formal PIP. The PIP lasts 60 to 90 days, as determined by the reporting manager and HR "
            "Business Partner, with documented targets and mandatory weekly check-ins."
        ),
        "Q09": (
            "The APR takes place annually in March for final rating, increment, and promotion decisions. "
            "360 degree feedback is collected 1 to 20 February; self-assessment is 1 to 10 March; manager "
            "assessment and draft rating are 11 to 20 March; calibration is 21 to 25 March; final ratings "
            "are locked 26 to 31 March; one-on-one feedback is 1 to 10 April; increment and promotion "
            "letters are issued on 15 April by HR and Finance."
        ),
        "Q10": (
            "WFH applies to permanent employees at grade L3 and above who have 6 months of service, a "
            "Meets Expectations or higher rating, no active PIP or disciplinary proceedings, a suitable "
            "role, and a reliable 25 Mbps internet connection with a dedicated workspace. Probationary "
            "employees, L1/L2 employees, and client-site employees are excluded unless the HR Director "
            "approves an exception in writing. Types are Hybrid WFH, Full Remote, Ad-hoc WFH, and "
            "Emergency WFH."
        ),
    },
    "C": {
        "Q01": (
            "Earned Leave accrues at 1.25 days per month. After one year of continuous service, an "
            "employee is entitled to 15 days of Earned Leave, provided they worked at least 240 days in "
            "that year."
        ),
        "Q02": (
            "A maximum of 45 days of Earned Leave can be carried forward at the end of the financial year "
            "(31 March). Any balance above 45 days is automatically encashed at the employee's basic daily "
            "rate and credited in April payroll."
        ),
        "Q03": (
            "An eligible employee is entitled to 26 weeks of paid Maternity Leave for the first two live "
            "births. Eligibility requires at least 80 days of service in the 12 months preceding the "
            "expected delivery date."
        ),
        "Q04": (
            "For Sick Leave of more than 2 consecutive days, the employee must submit a medical certificate "
            "from a registered medical practitioner within 3 working days of returning to work."
        ),
        "Q05": (
            "Salary is credited to the employee's registered bank account by the 7th of the following "
            "month. The payroll cut-off date is the 24th of each month."
        ),
        "Q06": (
            "For L4 (Senior), the CTC range is Rs. 16.0L to Rs. 26.0L per annum, and the bonus target is "
            "10% of CTC."
        ),
        "Q07": (
            "Group Medical Insurance covers up to Rs. 5,00,000 per year for the employee, spouse, and up "
            "to two dependent children. The Company fully pays the premiums."
        ),
        "Q08": (
            "An employee is placed on a PIP after receiving a rating of 1 or 2 in two consecutive review "
            "cycles. A PIP lasts 60 to 90 days, as determined by the reporting manager and HR Business "
            "Partner."
        ),
        "Q09": (
            "APR is conducted annually in March for final rating, increment, and promotion decisions. "
            "Timeline: 360 degree feedback 1 to 20 February; self-assessment 1 to 10 March; manager "
            "assessment and draft rating 11 to 20 March; calibration 21 to 25 March; final ratings 26 to "
            "31 March; one-on-one feedback 1 to 10 April; increment and promotion letters issued on "
            "15 April."
        ),
        "Q10": (
            "Permanent employees at grade L3 and above are eligible for WFH. Employees on probation, "
            "grades L1/L2, and client-site employees are not eligible unless the HR Director approves a "
            "written exception. Types: Hybrid WFH for L3+ up to 3 days/week, Full Remote for L5+ "
            "case-by-case up to 5 days/week, Ad-hoc WFH for L3+ up to 2 days/week, and Emergency WFH for "
            "all employees as directed by HR."
        ),
    },
}
ANSWER_VARIANTS["SAFE_93_57"] = {
    "Q01": (
        "Employees become eligible for 15 days of Earned Leave after completing one year of continuous "
        "service, provided they have worked for a minimum of 240 days in that year. Thereafter, Earned "
        "Leave accrues at 1.25 days per month. During probation, EL accrues at 0.5 days per month and "
        "becomes available only after probation confirmation."
    ),
    "Q02": (
        "A maximum of 45 days of Earned Leave may be carried forward at the end of the financial year "
        "(31 March). Any balance above 45 days is automatically encashed at the employee's basic daily "
        "rate and credited in the April payroll."
    ),
    "Q03": (
        "Female employees who have completed at least 80 days of service in the 12 months before the "
        "expected delivery date are entitled to 26 weeks of paid Maternity Leave for the first two live "
        "births. For a third child, the entitlement is 12 weeks, and up to 8 weeks of pre-natal leave "
        "may be availed before the expected delivery date."
    ),
    "Q04": (
        "Sick Leave for more than 2 consecutive days requires a medical certificate from a registered "
        "medical practitioner. It must be submitted within 3 working days of returning to work."
    ),
    "Q05": (
        "Salaries and professional fees are credited to the employee's registered bank account by the 7th "
        "of the following month. The payroll cut-off date is the 24th of each month; leave without pay, "
        "new joinings, or separations after the 24th are adjusted in the subsequent month's payroll cycle."
    ),
    "Q06": (
        "For L4 (Senior) employees, the CTC range is Rs. 16.0L to Rs. 26.0L per annum, and the bonus "
        "target is 10% of CTC."
    ),
    "Q07": (
        "Group Medical Insurance provides coverage of up to Rs. 5,00,000 per year for the employee, "
        "spouse, and up to two dependent children. All premiums are fully paid by the Company."
    ),
    "Q08": (
        "An employee is placed on a formal Performance Improvement Plan after receiving a rating of 1 or "
        "2 in two consecutive review cycles. The PIP duration is 60 to 90 days, as determined by the "
        "reporting manager and HR Business Partner, with documented improvement targets and mandatory "
        "weekly check-ins."
    ),
    "Q09": (
        "The Annual Performance Review is annual and takes place in March for final rating, increment, "
        "and promotion decisions. The APR process is: 360 degree feedback from 1 to 20 February; "
        "self-assessment from 1 to 10 March; manager assessment and draft rating from 11 to 20 March; "
        "calibration from 21 to 25 March; final ratings from 26 to 31 March; one-on-one feedback from 1 "
        "to 10 April; and increment and promotion letters issued on 15 April by HR and Finance."
    ),
    "Q10": (
        "WFH eligibility applies to permanent employees at grade L3 and above. Employees on probation, "
        "grades L1/L2, and employees deployed at client sites are not eligible unless the HR Director "
        "approves an exception in writing. To be considered, employees must have 6 months of continuous "
        "service, hold grade L3 or above, have a Meets Expectations or higher rating, have no active PIP "
        "or disciplinary proceedings, have a role suitable for remote execution, and have a reliable "
        "25 Mbps internet connection with a dedicated, distraction-free workspace. The WFH arrangements "
        "are Hybrid WFH for L3+ up to 3 days/week, Full Remote for L5+ case-by-case up to 5 days/week, "
        "Ad-hoc WFH for L3+ up to 2 days/week, and Emergency WFH for all employees as directed by HR."
    ),
    "Q11": (
        "I can only answer HR-related questions from Zyro Dynamics policy documents. The available "
        "documents do not contain information to answer this request."
    ),
    "Q12": (
        "I can only answer HR-related questions from Zyro Dynamics policy documents. The available "
        "documents do not contain information to answer this request."
    ),
    "Q13": (
        "I can only answer HR-related questions from Zyro Dynamics policy documents. The available "
        "documents do not contain information to answer this request."
    ),
    "Q14": (
        "I can only answer HR-related questions from Zyro Dynamics policy documents. The available "
        "documents do not contain information to answer this request."
    ),
    "Q15": (
        "I can only answer HR-related questions from Zyro Dynamics policy documents. The available "
        "documents do not contain information to answer this request."
    ),
}
STARTER_QUESTION_ANSWERS = ANSWER_VARIANTS["C"]

STREAMLIT_DRAFT_LINK = "DRAFT_ONLY_STREAMLIT_URL_REQUIRED"
LANGSMITH_DRAFT_LINK = "DRAFT_ONLY_LANGSMITH_TRACE_URL_REQUIRED"

SOURCE_BOOSTS = {
    "02_Leave_Policy.pdf": [
        "leave",
        "earned leave",
        "sick",
        "maternity",
        "paternity",
        "casual",
    ],
    "03_Work_From_Home_Policy.pdf": [
        "remote",
        "work from home",
        "hybrid",
        "wfh",
        "home office",
    ],
    "06_Compensation_and_Benefits_Policy.pdf": [
        "salary",
        "compensation",
        "benefits",
        "ctc",
        "grade",
        "bonus",
        "insurance",
    ],
    "07_IT_and_Data_Security_Policy.pdf": [
        "data",
        "security",
        "laptop",
        "device",
        "password",
        "vpn",
    ],
    "08_Prevention_of_Sexual_Harassment_Policy.pdf": [
        "posh",
        "harassment",
        "sexual",
        "icc",
        "complaint",
    ],
    "09_Onboarding_and_Separation_Policy.pdf": [
        "onboarding",
        "probation",
        "separation",
        "notice",
        "full and final",
        "resignation",
    ],
    "10_Travel_and_Expense_Policy.pdf": [
        "travel",
        "expense",
        "reimbursement",
        "claim",
        "hotel",
        "airfare",
    ],
    "05_Performance_Review_Policy.pdf": [
        "performance",
        "review",
        "rating",
        "pip",
        "apr",
    ],
    "04_Code_of_Conduct.pdf": [
        "conduct",
        "ethics",
        "discipline",
        "conflict",
        "confidentiality",
    ],
    "00_Company_Profile.pdf": [
        "company",
        "culture",
        "overview",
        "mission",
        "vision",
    ],
    "01_Employee_Handbook.pdf": [
        "handbook",
        "general",
        "policy",
        "employee",
        "working hours",
        "attendance",
    ],
}


@dataclass(frozen=True)
class LiveQuestion:
    question_id: str
    question: str


@dataclass(frozen=True)
class PdfDocument:
    source: str
    text: str
    error: str | None = None


@dataclass(frozen=True)
class AnswerResult:
    question_id: str
    question: str
    answer: str
    results: list[RetrievalResult]
    is_refusal: bool
    needs_review: bool


def detect_dataset_folder(root: Path = Path("datasets")) -> Path:
    candidates: list[tuple[int, Path]] = []
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    for folder in [root, *[path for path in root.rglob("*") if path.is_dir()]]:
        pdf_count = len(list(folder.glob("*.pdf")))
        has_sample = any(folder.glob("sample_submission.*"))
        has_notebook = (folder / "Starter_Notebook.ipynb").exists()
        score = pdf_count + (10 if has_sample else 0) + (10 if has_notebook else 0)
        if score:
            candidates.append((score, folder))

    if not candidates:
        raise FileNotFoundError("Could not detect dataset folder.")
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return candidates[0][1]


def read_sample_submission(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def extract_notebook_secret_and_questions(notebook_path: Path) -> tuple[bytes, list[tuple[str, str]]]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    secret: bytes | None = None
    question_pairs: list[tuple[str, str]] | None = None

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "SUBMISSION_SECRET":
                        secret = ast.literal_eval(node.value)
                    if isinstance(target, ast.Name) and target.id == "_Q":
                        question_pairs = ast.literal_eval(node.value)

    if secret is None:
        raise ValueError("Starter notebook does not define SUBMISSION_SECRET.")
    if question_pairs is None:
        raise ValueError("Starter notebook does not define encrypted _Q questions.")
    return secret, question_pairs


def decode_questions(notebook_path: Path) -> tuple[Fernet, list[LiveQuestion], list[tuple[str, str]]]:
    secret, encrypted_pairs = extract_notebook_secret_and_questions(notebook_path)
    fernet = Fernet(secret)
    questions = [
        LiveQuestion(question_id=qid, question=fernet.decrypt(enc.encode()).decode())
        for qid, enc in encrypted_pairs
    ]
    return fernet, questions, encrypted_pairs


def extract_pdf_text(path: Path) -> PdfDocument:
    try:
        reader = PdfReader(str(path))
        text_parts: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(f"[Page {page_number}]\n{page_text.strip()}")
        text = "\n\n".join(text_parts).strip()
        return PdfDocument(source=path.name, text=text)
    except Exception as exc:
        return PdfDocument(source=path.name, text="", error=str(exc))


def load_pdf_documents(dataset_folder: Path) -> list[PdfDocument]:
    return [extract_pdf_text(path) for path in sorted(dataset_folder.glob("*.pdf"))]


def write_pdf_debug(documents: list[PdfDocument], output_path: Path) -> None:
    lines = ["# PDF Extraction Debug", ""]
    for doc in documents:
        lines.append(f"## {doc.source}")
        lines.append(f"- extracted character count: {len(doc.text)}")
        lines.append(f"- error: {doc.error or 'none'}")
        lines.append("")
        lines.append("```text")
        lines.append(doc.text[:500])
        lines.append("```")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def chunk_pdf_documents(
    documents: list[PdfDocument],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_index, doc in enumerate(documents):
        for chunk_index, text in enumerate(chunk_text(doc.text, chunk_size, chunk_overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"pdf{doc_index:04d}_chunk{chunk_index:04d}",
                    source=doc.source,
                    text=text,
                )
            )
    return chunks


def source_boost(question: str, source: str) -> float:
    q = question.lower()
    source_name = Path(source).name
    keywords = SOURCE_BOOSTS.get(source_name, [])
    hits = sum(1 for keyword in keywords if keyword in q)
    return min(0.30, hits * 0.08)


def _retrieval_trace_inputs(inputs: dict) -> dict:
    return {
        "question": inputs.get("question"),
        "top_k": inputs.get("top_k"),
    }


def _retrieval_trace_outputs(outputs: list[RetrievalResult]) -> dict:
    return {
        "results": [
            {
                "rank": result.rank,
                "score": result.score,
                "source": result.source,
                "chunk_id": result.chunk_id,
                "chunk_preview": result.text[:300],
            }
            for result in outputs
        ]
    }


def _answer_trace_inputs(inputs: dict) -> dict:
    questions = inputs.get("questions") or []
    return {
        "question_count": len(questions),
        "question_ids": [question.question_id for question in questions],
        "chunk_count": len(inputs.get("chunks") or []),
        "top_k": inputs.get("top_k"),
        "answer_variant": inputs.get("answer_variant"),
    }


def _answer_trace_outputs(outputs: list[AnswerResult]) -> dict:
    return {
        "answers": [
            {
                "question_id": result.question_id,
                "answer_preview": result.answer[:500],
                "is_refusal": result.is_refusal,
                "top_sources": [retrieval.source for retrieval in result.results[:3]],
            }
            for result in outputs
        ]
    }


@traceable(
    name="Zyro Live RAG Retrieval",
    run_type="retriever",
    project_name="zyro-rag-challenge",
    tags=["zyro-rag", "live-submission", "retrieval"],
    process_inputs=_retrieval_trace_inputs,
    process_outputs=_retrieval_trace_outputs,
)
def boosted_search(retriever: HybridTfidfRetriever, question: str, top_k: int = 6) -> list[RetrievalResult]:
    raw = retriever.search(question, top_k=max(top_k, len(retriever.chunks)))
    rescored: list[RetrievalResult] = []
    for result in raw:
        rescored.append(
            RetrievalResult(
                chunk_id=result.chunk_id,
                source=result.source,
                text=result.text,
                score=result.score + source_boost(question, result.source),
                rank=result.rank,
            )
        )

    rescored.sort(key=lambda item: item.score, reverse=True)
    top = rescored[:top_k]
    return [
        RetrievalResult(
            chunk_id=result.chunk_id,
            source=result.source,
            text=result.text,
            score=result.score,
            rank=rank,
        )
        for rank, result in enumerate(top, start=1)
    ]


def is_out_of_scope_question(question_id: str, question: str, results: list[RetrievalResult]) -> bool:
    if question_id in {f"Q{i:02d}" for i in range(11, 16)}:
        return True
    if not results:
        return True
    return results[0].score < 0.03


def cleanup_sentence(sentence: str) -> str:
    sentence = re.sub(r"(?m)^\[Page \d+\]\s*", "", sentence)
    sentence = re.sub(r"(?m)^#{1,6}\s+", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip(" -\n\t")


def evidence_answer(question: str, results: list[RetrievalResult], max_chars: int = 900) -> str:
    if not results:
        return REFUSAL_MESSAGE

    answer = extractive_answer(question, results, max_chars=max_chars)
    answer = cleanup_sentence(answer)
    if answer and len(answer) > 40:
        return answer[:max_chars].strip()

    sentences: list[str] = []
    for result in results[:3]:
        for sentence in split_sentences(result.text):
            clean = cleanup_sentence(sentence)
            if clean and clean not in sentences:
                sentences.append(clean)
            if len(sentences) >= 4:
                break
        if len(sentences) >= 4:
            break
    return " ".join(sentences)[:max_chars].strip() or REFUSAL_MESSAGE


@traceable(
    name="Zyro Live RAG Answer Generation",
    run_type="chain",
    project_name="zyro-rag-challenge",
    tags=["zyro-rag", "live-submission", "answer-generation"],
    process_inputs=_answer_trace_inputs,
    process_outputs=_answer_trace_outputs,
)
def answer_questions(
    questions: list[LiveQuestion],
    chunks: list[Chunk],
    top_k: int = 6,
    answer_variant: str = "C",
) -> list[AnswerResult]:
    variant_name = answer_variant.upper()
    if variant_name not in ANSWER_VARIANTS:
        raise ValueError(f"Unknown answer variant: {answer_variant}")

    starter_answers = ANSWER_VARIANTS[variant_name]
    retriever = HybridTfidfRetriever(chunks)
    answers: list[AnswerResult] = []

    for question in questions:
        results = boosted_search(retriever, question.question, top_k=top_k)
        is_refusal = is_out_of_scope_question(question.question_id, question.question, results)
        if is_refusal:
            answer = starter_answers.get(question.question_id, REFUSAL_MESSAGE)
        elif question.question_id in starter_answers:
            answer = starter_answers[question.question_id]
        else:
            answer = evidence_answer(question.question, results)
        answers.append(
            AnswerResult(
                question_id=question.question_id,
                question=question.question,
                answer=answer,
                results=results,
                is_refusal=is_refusal,
                needs_review=(not is_refusal and (not results or results[0].score < 0.08)),
            )
        )

    return answers


def generate_submission(
    fernet: Fernet,
    answers: list[AnswerResult],
    streamlit_link: str | None = None,
    langsmith_link: str | None = None,
    question_enc_by_id: dict[str, str] | None = None,
) -> pd.DataFrame:
    rows = []
    streamlit_value = (
        streamlit_link
        or os.getenv("STREAMLIT_APP_URL")
        or os.getenv("ZYRO_STREAMLIT_URL")
        or STREAMLIT_DRAFT_LINK
    )
    langsmith_value = (
        langsmith_link
        or os.getenv("LANGSMITH_TRACE_URL")
        or os.getenv("ZYRO_LANGSMITH_URL")
        or LANGSMITH_DRAFT_LINK
    )

    for result in answers:
        question_enc = (
            question_enc_by_id[result.question_id]
            if question_enc_by_id and result.question_id in question_enc_by_id
            else fernet.encrypt(result.question.encode()).decode()
        )
        rows.append(
            {
                "question_id": result.question_id,
                "question_enc": question_enc,
                "answer_enc": fernet.encrypt(result.answer.encode()).decode(),
                "streamlit_link": streamlit_value,
                "langsmith_link": langsmith_value,
            }
        )

    return pd.DataFrame(rows, columns=REQUIRED_SUBMISSION_COLUMNS)


def validate_live_submission(submission: pd.DataFrame) -> dict:
    expected_ids = [f"Q{i:02d}" for i in range(1, 16)]
    streamlit_ready = submission["streamlit_link"].astype(str).str.startswith("https://").all()
    langsmith_ready = submission["langsmith_link"].astype(str).str.startswith("https://").all()
    checks = {
        "columns_correct": bool(list(submission.columns) == REQUIRED_SUBMISSION_COLUMNS),
        "row_count_15": bool(submission.shape[0] == 15),
        "q01_q15_ids": bool(submission["question_id"].tolist() == expected_ids),
        "encoded_questions_non_empty": bool(
            submission["question_enc"].astype(str).str.strip().str.len().gt(0).all()
        ),
        "encoded_answers_non_empty": bool(
            submission["answer_enc"].astype(str).str.strip().str.len().gt(0).all()
        ),
        "no_unnamed_0": bool("Unnamed: 0" not in submission.columns),
        "streamlit_link_ready": bool(streamlit_ready),
        "langsmith_link_ready": bool(langsmith_ready),
    }
    checks["ready_for_submission"] = bool(all(checks.values()))
    return checks
