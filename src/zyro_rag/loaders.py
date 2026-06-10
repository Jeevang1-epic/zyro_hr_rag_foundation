from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


TEXT_EXTENSIONS = {".txt", ".md"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".csv", ".json"}


def clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def load_text_file(path: Path) -> str:
    return clean_text(path.read_text(encoding="utf-8", errors="ignore"))


def load_json_file(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    return clean_text(json.dumps(data, ensure_ascii=False, indent=2))


def csv_text_columns(df: pd.DataFrame) -> list[str]:
    direct_matches: list[str] = []
    for col in df.columns:
        lower = str(col).lower()
        if any(key in lower for key in ["policy", "content", "text", "document", "description", "body"]):
            direct_matches.append(col)

    if direct_matches:
        return direct_matches

    inferred: list[str] = []
    for col in df.columns:
        if df[col].dtype != "object":
            continue
        lengths = df[col].dropna().astype(str).str.len()
        if not lengths.empty and float(lengths.mean()) > 80:
            inferred.append(col)
    return inferred


def load_csv_as_documents(path: Path) -> list[dict]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    text_cols = csv_text_columns(df)
    documents: list[dict] = []
    for row_index, row in df.iterrows():
        parts: list[str] = []
        for col in text_cols:
            value = row.get(col)
            if pd.notna(value) and str(value).strip():
                parts.append(f"{col}: {value}")
        text = clean_text("\n".join(parts))
        if text:
            documents.append({"source": f"{path}#row={row_index}", "text": text})
    return documents


def load_documents(input_dir: Path) -> list[dict]:
    documents: list[dict] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        suffix = path.suffix.lower()
        try:
            if suffix in TEXT_EXTENSIONS:
                text = load_text_file(path)
                if text:
                    documents.append({"source": str(path), "text": text})
            elif suffix == ".json":
                text = load_json_file(path)
                if text:
                    documents.append({"source": str(path), "text": text})
            elif suffix == ".csv":
                documents.extend(load_csv_as_documents(path))
        except Exception as exc:
            documents.append({"source": str(path), "text": f"[LOAD_ERROR] {exc}"})

    return [doc for doc in documents if str(doc.get("text", "")).strip()]
