from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RagConfig:
    input_dir: Path
    output_dir: Path
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    max_answer_chars: int = 700
