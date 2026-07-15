from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model: str
    database_path: Path
    active_prompt_path: Path
    evals_path: Path
    state_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model=os.getenv("OPENAI_MODEL", "gpt-5.6"),
            database_path=Path(os.getenv("SECOND_BRAIN_DB", "second_brain.db")),
            active_prompt_path=Path(
                os.getenv("SECOND_BRAIN_ACTIVE_PROMPT", "prompts/active.json")
            ),
            evals_path=Path(os.getenv("SECOND_BRAIN_EVALS", "data/evals.jsonl")),
            state_dir=Path(os.getenv("SECOND_BRAIN_STATE_DIR", "state")),
        )
