from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def detect_dataset_folder() -> Path:
    candidates: list[tuple[int, Path]] = []
    for folder in [DATASETS_DIR, *[p for p in DATASETS_DIR.rglob("*") if p.is_dir()]]:
        pdf_count = len(list(folder.glob("*.pdf")))
        has_sample = any(folder.glob("sample_submission.*"))
        has_notebook = (folder / "Starter_Notebook.ipynb").exists()
        score = pdf_count + (10 if has_sample else 0) + (10 if has_notebook else 0)
        if score:
            candidates.append((score, folder))
    if not candidates:
        raise FileNotFoundError("Could not detect a dataset folder under datasets/.")
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return candidates[0][1]


def read_sample(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def has_placeholder_link(series: pd.Series) -> bool:
    values = series.astype(str).str.lower()
    return bool(values.str.contains("your-app-name|your-trace-id|placeholder|example").any())


def has_real_https_link(series: pd.Series) -> bool:
    return bool(series.astype(str).str.startswith("https://").all() and not has_placeholder_link(series))


def code_summary(source: str) -> dict:
    imports: list[str] = []
    constants: list[str] = []
    functions: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"imports": imports, "constants": constants, "functions": functions}

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(source, node) or "")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    constants.append(target.id)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
    return {"imports": imports, "constants": constants, "functions": functions}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_folder = detect_dataset_folder()
    pdf_paths = sorted(dataset_folder.glob("*.pdf"))
    sample_paths = sorted(dataset_folder.glob("sample_submission.*"))
    notebook_path = dataset_folder / "Starter_Notebook.ipynb"

    if not sample_paths:
        raise FileNotFoundError("No sample_submission.* file found.")
    if not notebook_path.exists():
        raise FileNotFoundError("Starter_Notebook.ipynb not found.")

    sample_path = sample_paths[0]
    sample = read_sample(sample_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    headings: list[str] = []
    todo_cells: list[tuple[int, str]] = []
    imports: list[str] = []
    constants: list[str] = []
    functions: list[str] = []
    helper_cells: list[tuple[int, str]] = []
    question_cells: list[tuple[int, str]] = []
    submission_cells: list[tuple[int, str]] = []
    streamlit_cells: list[tuple[int, str]] = []
    langsmith_cells: list[tuple[int, str]] = []

    for index, cell in enumerate(notebook.get("cells", [])):
        source = "".join(cell.get("source", []))
        source_lower = source.lower()
        if "todo" in source_lower:
            todo_cells.append((index, source.strip()))

        if cell.get("cell_type") == "markdown":
            for line in source.splitlines():
                if line.lstrip().startswith("#"):
                    headings.append(line.strip())
            if "streamlit" in source_lower:
                streamlit_cells.append((index, source.strip()))
            if "langsmith" in source_lower or "langchain" in source_lower:
                langsmith_cells.append((index, source.strip()))
        elif cell.get("cell_type") == "code":
            summary = code_summary(source)
            imports.extend(summary["imports"])
            constants.extend(summary["constants"])
            functions.extend(summary["functions"])
            if any(key in source_lower for key in ["encode", "decode", "encrypt", "decrypt"]):
                helper_cells.append((index, source.strip()))
            if "question" in source_lower:
                question_cells.append((index, source.strip()))
            if "submission" in source_lower or "to_csv" in source_lower:
                submission_cells.append((index, source.strip()))
            if "streamlit" in source_lower:
                streamlit_cells.append((index, source.strip()))
            if "langsmith" in source_lower or "langchain" in source_lower:
                langsmith_cells.append((index, source.strip()))

    lines: list[str] = [
        "# Starter Notebook Findings",
        "",
        "## Dataset",
        f"- dataset folder: `{dataset_folder}`",
        f"- sample submission: `{sample_path}`",
        f"- starter notebook: `{notebook_path}`",
        f"- PDF count: {len(pdf_paths)}",
        "- PDF files:",
        *[f"  - {path.name}" for path in pdf_paths],
        "",
        "## Sample Submission",
        f"- shape: {sample.shape}",
        f"- columns: {sample.columns.tolist()}",
        f"- question_id values: {sample['question_id'].tolist() if 'question_id' in sample.columns else 'missing'}",
        f"- question_enc filled: {bool(sample['question_enc'].astype(str).str.strip().str.len().gt(0).all()) if 'question_enc' in sample.columns else 'missing'}",
        f"- answer_enc filled: {bool(sample['answer_enc'].astype(str).str.strip().str.len().gt(0).all()) if 'answer_enc' in sample.columns else 'missing'}",
        f"- streamlit_link placeholder: {has_placeholder_link(sample['streamlit_link']) if 'streamlit_link' in sample.columns else 'missing'}",
        f"- streamlit_link real https: {has_real_https_link(sample['streamlit_link']) if 'streamlit_link' in sample.columns else 'missing'}",
        f"- langsmith_link placeholder: {has_placeholder_link(sample['langsmith_link']) if 'langsmith_link' in sample.columns else 'missing'}",
        f"- langsmith_link real https: {has_real_https_link(sample['langsmith_link']) if 'langsmith_link' in sample.columns else 'missing'}",
        "",
        "## Markdown Headings",
        *[f"- {heading}" for heading in headings],
        "",
        "## TODO Cells",
        *([f"- Cell {index}: {text[:500].replace(chr(10), ' ')}" for index, text in todo_cells] or ["- none found"]),
        "",
        "## Imports",
        *[f"- `{item}`" for item in sorted(set(imports))],
        "",
        "## Config Constants",
        *([f"- `{item}`" for item in sorted(set(constants))] or ["- none found"]),
        "",
        "## Helper Functions",
        *([f"- `{item}`" for item in sorted(set(functions))] or ["- none found"]),
        "",
        "## Encoding / Decoding Cells",
    ]

    for index, text in helper_cells:
        lines.extend([f"### Code cell {index}", "```python", text, "```", ""])

    lines.append("## Question Cells")
    for index, text in question_cells:
        lines.extend([f"### Code cell {index}", "```python", text[:3000], "```", ""])

    lines.append("## Submission Generation Cells")
    for index, text in submission_cells:
        lines.extend([f"### Code cell {index}", "```python", text[:3000], "```", ""])

    lines.append("## Streamlit Instructions")
    if streamlit_cells:
        for index, text in streamlit_cells:
            lines.extend([f"- Cell {index}: {text[:1000].replace(chr(10), ' ')}"])
    else:
        lines.append("- none found")

    lines.append("")
    lines.append("## LangSmith Instructions")
    if langsmith_cells:
        for index, text in langsmith_cells:
            lines.extend([f"- Cell {index}: {text[:1000].replace(chr(10), ' ')}"])
    else:
        lines.append("- none found")

    findings_path = OUTPUT_DIR / "starter_notebook_findings.md"
    findings_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"detected dataset folder: {dataset_folder}")
    print(f"PDF count: {len(pdf_paths)}")
    for path in pdf_paths:
        print(f"PDF: {path.name}")
    print(f"sample submission: {sample_path}")
    print(f"starter notebook: {notebook_path}")
    print(f"sample shape: {sample.shape}")
    print(f"sample columns: {sample.columns.tolist()}")
    print(sample.head().to_string(index=False))
    print(f"findings written: {findings_path}")


if __name__ == "__main__":
    main()
