from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptVersion:
    version: int
    instructions: str
    principles: list[str]
    changelog: str
    created_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptVersion":
        return cls(
            version=int(data["version"]),
            instructions=str(data["instructions"]).strip(),
            principles=[str(item) for item in data.get("principles", [])],
            changelog=str(data.get("changelog", "")).strip(),
            created_at=str(data.get("created_at", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "instructions": self.instructions,
            "principles": self.principles,
            "changelog": self.changelog,
            "created_at": self.created_at,
        }

    def render(self) -> str:
        principles = "\n".join(f"- {item}" for item in self.principles)
        return f"{self.instructions}\n\nPrincipes obligatoires:\n{principles}".strip()


def load_prompt(path: Path) -> PromptVersion:
    with path.open("r", encoding="utf-8") as handle:
        return PromptVersion.from_dict(json.load(handle))


def write_prompt(path: Path, prompt: PromptVersion) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(prompt.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def save_candidate(
    state_dir: Path,
    instructions: str,
    principles: list[str],
    rationale: str,
    baseline_version: int,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = PromptVersion(
        version=baseline_version + 1,
        instructions=instructions.strip(),
        principles=principles,
        changelog=rationale.strip(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    path = state_dir / "candidates" / f"candidate-{timestamp}.json"
    write_prompt(path, candidate)
    return path


def promote_candidate(candidate_path: Path, active_path: Path, state_dir: Path) -> PromptVersion:
    candidate = load_prompt(candidate_path)
    if active_path.exists():
        active = load_prompt(active_path)
        if candidate.version <= active.version:
            raise ValueError(
                f"Candidate version {candidate.version} must be newer than active version {active.version}."
            )
        archive = state_dir / "archive" / f"prompt-v{active.version}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(active_path, archive)

    write_prompt(active_path, candidate)
    return candidate
