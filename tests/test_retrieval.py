from zyro_rag.chunking import Chunk
from zyro_rag.retrieval import HybridTfidfRetriever


def test_retrieval_returns_relevant_chunk():
    chunks = [
        Chunk("1", "leave.md", "Employees are eligible for annual leave after probation."),
        Chunk("2", "payroll.md", "Salary is processed on the last working day."),
    ]
    retriever = HybridTfidfRetriever(chunks)
    results = retriever.search("When is salary processed?", top_k=1)
    assert results[0].source == "payroll.md"


def test_retrieval_respects_top_k_limit():
    chunks = [
        Chunk("1", "leave.md", "Annual leave policy."),
        Chunk("2", "payroll.md", "Payroll policy."),
    ]
    retriever = HybridTfidfRetriever(chunks)
    assert len(retriever.search("policy", top_k=5)) == 2
