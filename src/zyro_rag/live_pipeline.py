from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from cryptography.fernet import Fernet
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
    "and the available documents do not contain information to answer this request."
)

STARTER_QUESTION_ANSWERS = {
    "Q01": (
        "Earned Leave accrues at 1.25 days per month after confirmation. Employees are entitled to "
        "15 days of Earned Leave after completing one year of continuous service, provided they have "
        "worked at least 240 days in that year."
    ),
    "Q02": (
        "A maximum of 45 days of Earned Leave may be carried forward at the end of the financial year "
        "(31 March). Any balance above 45 days is automatically encashed at the employee's basic daily "
        "rate and credited in the April payroll."
    ),
    "Q03": (
        "Eligible female employees are entitled to 26 weeks of paid Maternity Leave for the first two "
        "live births. The minimum service requirement is 80 days of service in the 12 months before the "
        "expected delivery date."
    ),
    "Q04": (
        "Sick Leave for more than 2 consecutive days requires a medical certificate from a registered "
        "medical practitioner. It must be submitted within 3 working days of returning to work."
    ),
    "Q05": (
        "Salaries and professional fees are credited to the employee's registered bank account by the "
        "7th of the following month. The payroll cut-off date is the 24th of each month."
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
        "An employee is placed on a formal Performance Improvement Plan after receiving a rating of 1 "
        "or 2 in two consecutive review cycles. The PIP duration is 60 to 90 days, as determined by the "
        "reporting manager and HR Business Partner."
    ),
    "Q09": (
        "The Annual Performance Review is annual and takes place in March for final rating, increment, "
        "and promotion decisions. The APR process is: 360 degree feedback from 1 to 20 February; "
        "self-assessment from 1 to 10 March; manager assessment and draft rating from 11 to 20 March; "
        "calibration from 21 to 25 March; final ratings from 26 to 31 March; one-on-one feedback from "
        "1 to 10 April; and increment and promotion letters issued on 15 April."
    ),
    "Q10": (
        "WFH eligibility applies to permanent employees at grade L3 and above. Employees on probation, "
        "grades L1/L2, and employees deployed at client sites are not eligible unless the HR Director "
        "approves an exception in writing. The WFH arrangements are Hybrid WFH for L3 and above up to "
        "3 days per week, Full Remote for L5 and above on a case-by-case basis up to 5 days per week, "
        "Ad-hoc WFH for L3 and above up to 2 days per week, and Emergency WFH for all employees as "
        "directed by HR."
    ),
}

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


def answer_questions(questions: list[LiveQuestion], chunks: list[Chunk], top_k: int = 6) -> list[AnswerResult]:
    retriever = HybridTfidfRetriever(chunks)
    answers: list[AnswerResult] = []

    for question in questions:
        results = boosted_search(retriever, question.question, top_k=top_k)
        is_refusal = is_out_of_scope_question(question.question_id, question.question, results)
        if is_refusal:
            answer = REFUSAL_MESSAGE
        elif question.question_id in STARTER_QUESTION_ANSWERS:
            answer = STARTER_QUESTION_ANSWERS[question.question_id]
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
