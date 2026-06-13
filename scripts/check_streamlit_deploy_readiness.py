from __future__ import annotations

import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_FILE = PROJECT_ROOT / "app" / "streamlit_app.py"
DATASET_FOLDER = PROJECT_ROOT / "datasets" / "zyro-dynamics-hr-corpus"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
PACKAGE_DIR = PROJECT_ROOT / "src" / "zyro_rag"

EXPECTED_REQUIREMENTS = {
    "cryptography",
    "langsmith",
    "numpy",
    "openpyxl",
    "pandas",
    "pypdf",
    "pytest",
    "scikit-learn",
    "streamlit",
}
SECRET_FILE_NAMES = {".env", "kaggle.json"}
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"\bghp_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b(?:LANGSMITH|LANGCHAIN|GROQ|OPENAI|KAGGLE)_[A-Z_]*KEY\s*=\s*['\"]?[A-Za-z0-9_\-]{16,}"),
]
MAX_TRACKED_FILE_BYTES = 25 * 1024 * 1024


def run_git_ls_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [PROJECT_ROOT / line.strip() for line in result.stdout.splitlines() if line.strip()]


def requirement_names() -> set[str]:
    names: set[str] = set()
    for raw_line in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip().lower().replace("_", "-")
        if name:
            names.add(name)
    return names


def print_check(label: str, passed: bool, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"{label}: {'PASS' if passed else 'FAIL'}{suffix}")


def scan_tracked_files_for_secrets(tracked_files: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in tracked_files:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if path.name in SECRET_FILE_NAMES:
            findings.append(f"tracked secret-like file: {rel}")
            continue
        if path.suffix.lower() in {".pdf", ".xlsx", ".png", ".jpg", ".jpeg", ".ico"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            findings.append(f"could not scan {rel}: {exc}")
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"secret-like value in tracked file: {rel}")
                break
    return findings


def main() -> None:
    tracked_files = run_git_ls_files()
    tracked_rel = {path.relative_to(PROJECT_ROOT).as_posix() for path in tracked_files}

    pdf_files = sorted(DATASET_FOLDER.glob("*.pdf")) if DATASET_FOLDER.exists() else []
    requirements = requirement_names() if REQUIREMENTS_FILE.exists() else set()
    missing_requirements = sorted(EXPECTED_REQUIREMENTS - requirements)
    oversized_files = [
        f"{path.relative_to(PROJECT_ROOT).as_posix()} ({path.stat().st_size} bytes)"
        for path in tracked_files
        if path.exists() and path.stat().st_size > MAX_TRACKED_FILE_BYTES
    ]
    secret_findings = scan_tracked_files_for_secrets(tracked_files)

    checks = {
        "required_app_file_exists": APP_FILE.exists(),
        "required_app_file_tracked": "app/streamlit_app.py" in tracked_rel,
        "required_dataset_folder_exists": DATASET_FOLDER.exists(),
        "eleven_policy_pdfs_exist": len(pdf_files) == 11,
        "dataset_files_tracked_for_streamlit_cloud": all(
            f"datasets/zyro-dynamics-hr-corpus/{path.name}" in tracked_rel for path in pdf_files
        )
        and len(pdf_files) == 11,
        "requirements_txt_exists": REQUIREMENTS_FILE.exists(),
        "requirements_cover_app_imports": not missing_requirements,
        "package_structure_valid": (PACKAGE_DIR / "__init__.py").exists()
        and (PACKAGE_DIR / "live_pipeline.py").exists()
        and (PACKAGE_DIR / "retrieval.py").exists(),
        "outputs_not_tracked": not any(rel.startswith("outputs/") for rel in tracked_rel),
        "no_secret_files_tracked": not any(Path(rel).name in SECRET_FILE_NAMES for rel in tracked_rel),
        "no_secret_values_in_tracked_text": not secret_findings,
        "no_oversized_tracked_files": not oversized_files,
    }

    print(f"app_file: {APP_FILE.relative_to(PROJECT_ROOT)}")
    print(f"dataset_folder: {DATASET_FOLDER.relative_to(PROJECT_ROOT)}")
    print(f"policy_pdf_count: {len(pdf_files)}")
    print(f"tracked_file_count: {len(tracked_files)}")
    print(f"largest_tracked_file_limit_bytes: {MAX_TRACKED_FILE_BYTES}")
    for label, passed in checks.items():
        detail = ""
        if label == "requirements_cover_app_imports" and missing_requirements:
            detail = "missing " + ", ".join(missing_requirements)
        if label == "no_secret_values_in_tracked_text" and secret_findings:
            detail = "; ".join(secret_findings)
        if label == "no_oversized_tracked_files" and oversized_files:
            detail = "; ".join(oversized_files)
        print_check(label, passed, detail)

    if all(checks.values()):
        print("STREAMLIT DEPLOY READINESS: PASS")
        return

    raise SystemExit("STREAMLIT DEPLOY READINESS: FAIL")


if __name__ == "__main__":
    main()
