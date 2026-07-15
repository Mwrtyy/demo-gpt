from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from .core import SecondBrain
from .prompt_store import PromptVersion


@dataclass(frozen=True)
class EvalCase:
    id: str
    input: str
    category: str
    weight: float = 1.0
    required_all: tuple[str, ...] = ()
    required_any: tuple[str, ...] = ()
    banned: tuple[str, ...] = ()
    min_chars: int = 0
    max_chars: int = 0
    critical: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "EvalCase":
        return cls(
            id=str(data["id"]),
            input=str(data["input"]),
            category=str(data.get("category", "general")),
            weight=float(data.get("weight", 1.0)),
            required_all=tuple(str(item).lower() for item in data.get("required_all", [])),
            required_any=tuple(str(item).lower() for item in data.get("required_any", [])),
            banned=tuple(str(item).lower() for item in data.get("banned", [])),
            min_chars=int(data.get("min_chars", 0)),
            max_chars=int(data.get("max_chars", 0)),
            critical=bool(data.get("critical", False)),
        )


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    score: float
    answer: str
    failures: tuple[str, ...]
    critical: bool


@dataclass(frozen=True)
class EvalReport:
    prompt_version: int
    mean_score: float
    weighted_score: float
    results: tuple[CaseResult, ...]

    @property
    def failed(self) -> tuple[CaseResult, ...]:
        return tuple(result for result in self.results if result.score < 0.999)

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_version": self.prompt_version,
            "mean_score": self.mean_score,
            "weighted_score": self.weighted_score,
            "results": [
                {
                    "case_id": result.case_id,
                    "category": result.category,
                    "score": result.score,
                    "answer": result.answer,
                    "failures": list(result.failures),
                    "critical": result.critical,
                }
                for result in self.results
            ],
        }


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            cases.append(EvalCase.from_dict(data))
    if not cases:
        raise ValueError("The eval dataset is empty.")
    return cases


def score_answer(case: EvalCase, answer: str) -> CaseResult:
    normalized = answer.lower()
    failures: list[str] = []
    score = 1.0

    missing_all = [term for term in case.required_all if term not in normalized]
    if missing_all:
        score -= min(0.6, 0.2 * len(missing_all))
        failures.append(f"missing required terms: {', '.join(missing_all)}")

    if case.required_any and not any(term in normalized for term in case.required_any):
        score -= 0.45
        failures.append(f"missing every alternative: {', '.join(case.required_any)}")

    present_banned = [term for term in case.banned if term in normalized]
    if present_banned:
        score -= min(0.8, 0.4 * len(present_banned))
        failures.append(f"contains banned terms: {', '.join(present_banned)}")

    if case.min_chars and len(answer) < case.min_chars:
        score -= 0.25
        failures.append(f"too short: {len(answer)} < {case.min_chars}")

    if case.max_chars and len(answer) > case.max_chars:
        overflow_ratio = (len(answer) - case.max_chars) / max(case.max_chars, 1)
        score -= min(0.5, 0.15 + overflow_ratio * 0.2)
        failures.append(f"too long: {len(answer)} > {case.max_chars}")

    return CaseResult(
        case_id=case.id,
        category=case.category,
        score=max(0.0, round(score, 4)),
        answer=answer,
        failures=tuple(failures),
        critical=case.critical,
    )


def build_report(
    prompt_version: int,
    cases_and_results: Iterable[tuple[EvalCase, CaseResult]],
) -> EvalReport:
    pairs = list(cases_and_results)
    if not pairs:
        raise ValueError("No eval results were provided.")
    scores = [result.score for _, result in pairs]
    total_weight = sum(case.weight for case, _ in pairs)
    weighted = sum(case.weight * result.score for case, result in pairs) / total_weight
    return EvalReport(
        prompt_version=prompt_version,
        mean_score=round(mean(scores), 4),
        weighted_score=round(weighted, 4),
        results=tuple(result for _, result in pairs),
    )


def evaluate_prompt(
    brain: SecondBrain,
    prompt: PromptVersion,
    cases: Iterable[EvalCase],
) -> EvalReport:
    pairs: list[tuple[EvalCase, CaseResult]] = []
    for case in cases:
        answer = brain.answer(case.input, prompt=prompt, remember=False).text
        pairs.append((case, score_answer(case, answer)))
    return build_report(prompt.version, pairs)
