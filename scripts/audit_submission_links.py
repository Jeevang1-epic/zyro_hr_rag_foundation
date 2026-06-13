from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


REQUIRED_COLUMNS = [
    "question_id",
    "question_enc",
    "answer_enc",
    "streamlit_link",
    "langsmith_link",
]

PLACEHOLDER_MARKERS = ("DRAFT", "PLACEHOLDER", "TODO", "REQUIRED", "example.com")


def single_value(series: pd.Series) -> str:
    values = series.astype(str).str.strip().drop_duplicates().tolist()
    return values[0] if len(values) == 1 else "; ".join(values)


def is_https(url: str) -> bool:
    return urlparse(url).scheme == "https"


def looks_like_streamlit(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and (parsed.hostname or "").endswith(".streamlit.app")


def looks_like_langsmith_trace(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path.strip("/")
    if host != "smith.langchain.com":
        return False
    if "projects" in path.split("/"):
        return False
    return path.startswith("public/") and path.endswith("/r")


def contains_placeholder(value: str) -> bool:
    upper_value = value.upper()
    return any(marker in upper_value for marker in PLACEHOLDER_MARKERS)


def print_check(label: str, value: bool) -> None:
    print(f"{label}: {'PASS' if value else 'FAIL'}")


def main() -> None:
    path = Path("outputs/submission.csv")
    if not path.exists():
        raise SystemExit("LINK AUDIT: FAIL - outputs/submission.csv missing")

    submission = pd.read_csv(path)
    if "streamlit_link" not in submission.columns or "langsmith_link" not in submission.columns:
        raise SystemExit("LINK AUDIT: FAIL - link columns missing")

    streamlit_link = single_value(submission["streamlit_link"])
    langsmith_link = single_value(submission["langsmith_link"])

    checks = {
        "columns_exact": list(submission.columns) == REQUIRED_COLUMNS,
        "streamlit_link_single_value": submission["streamlit_link"].astype(str).str.strip().nunique() == 1,
        "langsmith_link_single_value": submission["langsmith_link"].astype(str).str.strip().nunique() == 1,
        "both_links_non_empty": bool(streamlit_link and langsmith_link),
        "both_links_https": is_https(streamlit_link) and is_https(langsmith_link),
        "streamlit_looks_like_streamlit_app": looks_like_streamlit(streamlit_link),
        "langsmith_looks_like_public_trace": looks_like_langsmith_trace(langsmith_link),
        "no_placeholder_link_text": not contains_placeholder(streamlit_link)
        and not contains_placeholder(langsmith_link),
    }

    print(f"streamlit_link: {streamlit_link}")
    print(f"langsmith_link: {langsmith_link}")
    for label, passed in checks.items():
        print_check(label, passed)

    if all(checks.values()):
        print("LINK AUDIT: PASS")
        return

    raise SystemExit("LINK AUDIT: FAIL")


if __name__ == "__main__":
    main()
