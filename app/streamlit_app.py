from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
DATASET_ROOT = PROJECT_ROOT / "datasets"
EXPECTED_DATASET_FOLDER = DATASET_ROOT / "zyro-dynamics-hr-corpus"

from zyro_rag.live_pipeline import (  # noqa: E402
    REFUSAL_MESSAGE,
    boosted_search,
    chunk_pdf_documents,
    detect_dataset_folder,
    evidence_answer,
    is_out_of_scope_question,
    load_pdf_documents,
)
from zyro_rag.retrieval import HybridTfidfRetriever  # noqa: E402


def resolve_dataset_folder() -> Path:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            "HR policy dataset is missing. Expected the repository to include "
            f"{EXPECTED_DATASET_FOLDER.relative_to(PROJECT_ROOT)} with 11 policy PDFs."
        )

    try:
        dataset_folder = detect_dataset_folder(DATASET_ROOT)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "HR policy dataset could not be detected. Expected 11 policy PDFs under "
            f"{EXPECTED_DATASET_FOLDER.relative_to(PROJECT_ROOT)}."
        ) from exc

    pdf_count = len(list(dataset_folder.glob("*.pdf")))
    if pdf_count != 11:
        raise FileNotFoundError(
            f"Expected 11 policy PDFs in {dataset_folder.relative_to(PROJECT_ROOT)}, found {pdf_count}."
        )
    return dataset_folder


@st.cache_resource(show_spinner="Loading HR policy PDFs...")
def build_retriever():
    dataset_folder = resolve_dataset_folder()
    documents = load_pdf_documents(dataset_folder)
    failures = [doc for doc in documents if doc.error or not doc.text.strip()]
    if failures:
        details = ", ".join(f"{doc.source}: {doc.error or 'empty extraction'}" for doc in failures)
        raise RuntimeError(f"PDF extraction failed: {details}")
    chunks = chunk_pdf_documents(documents, chunk_size=900, chunk_overlap=150)
    return dataset_folder, documents, chunks, HybridTfidfRetriever(chunks)


def answer_employee_question(question: str):
    _, _, _, retriever = build_retriever()
    results = boosted_search(retriever, question, top_k=6)
    if is_out_of_scope_question("ADHOC", question, results):
        return REFUSAL_MESSAGE, results, True
    return evidence_answer(question, results), results, False


st.set_page_config(page_title="Zyro HR Help Desk", layout="wide")
st.title("Zyro HR Help Desk")

try:
    dataset_folder, documents, chunks, _ = build_retriever()
    st.caption(f"Loaded {len(documents)} policy PDFs from {dataset_folder.name} into {len(chunks)} chunks.")
except Exception as exc:
    st.error(str(exc))
    st.stop()

question = st.text_area(
    "Employee question",
    placeholder="Ask about leave, payroll, WFH, performance, benefits, IT security, POSH, onboarding, or travel policy.",
    height=110,
)

if st.button("Answer", type="primary") and question.strip():
    answer, results, is_refusal = answer_employee_question(question.strip())
    if is_refusal:
        st.warning(answer)
    else:
        st.success(answer)

    st.subheader("Retrieved evidence")
    for result in results:
        with st.expander(f"{result.rank}. {result.source} | score {result.score:.3f}"):
            st.write(result.text[:1200])
