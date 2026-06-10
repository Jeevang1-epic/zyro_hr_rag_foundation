from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zyro_rag.config import RagConfig
from zyro_rag.pipeline import run_pipeline


def main() -> None:
    log = run_pipeline(
        RagConfig(
            input_dir=PROJECT_ROOT / "sample_mock" / "input",
            output_dir=PROJECT_ROOT / "outputs",
            chunk_size=500,
            chunk_overlap=80,
            top_k=3,
        )
    )
    print("Smoke run passed.")
    print(f"questions={log['num_questions']}")
    print(f"documents={log['num_documents']}")
    print(f"chunks={log['num_chunks']}")
    print(f"submission={log['submission_path']}")
    print(f"validation_ok={log['validation']['ok']}")


if __name__ == "__main__":
    main()
