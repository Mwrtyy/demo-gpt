from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .core import SecondBrain
from .evaluation import EvalReport, evaluate_prompt, load_cases
from .prompt_store import PromptVersion, load_prompt, promote_candidate, save_candidate


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    reasons: tuple[str, ...]
    baseline_score: float
    candidate_score: float


@dataclass(frozen=True)
class ImprovementRun:
    baseline: EvalReport
    candidate: EvalReport
    candidate_path: Path
    decision: PromotionDecision
    promoted: bool


def promotion_gate(
    baseline: EvalReport,
    candidate: EvalReport,
    *,
    minimum_gain: float = 0.02,
    maximum_case_regression: float = 0.10,
) -> PromotionDecision:
    reasons: list[str] = []
    gain = candidate.weighted_score - baseline.weighted_score
    if gain < minimum_gain:
        reasons.append(
            f"weighted gain {gain:.4f} is below required gain {minimum_gain:.4f}"
        )

    baseline_by_id = {result.case_id: result for result in baseline.results}
    for candidate_result in candidate.results:
        baseline_result = baseline_by_id.get(candidate_result.case_id)
        if baseline_result is None:
            reasons.append(f"candidate contains unknown case {candidate_result.case_id}")
            continue
        regression = baseline_result.score - candidate_result.score
        if regression > maximum_case_regression:
            reasons.append(f"{candidate_result.case_id} regressed by {regression:.4f}")
        if baseline_result.critical and candidate_result.score < baseline_result.score:
            reasons.append(f"critical case {candidate_result.case_id} regressed")

    return PromotionDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        baseline_score=baseline.weighted_score,
        candidate_score=candidate.weighted_score,
    )


def _extract_json(text: str) -> dict[str, object]:
    match = JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError("The improver returned no JSON object.")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"The improver returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("The improver JSON must be an object.")
    return data


def propose_candidate(
    brain: SecondBrain,
    baseline_prompt: PromptVersion,
    baseline_report: EvalReport,
) -> tuple[str, list[str], str]:
    failures = [
        {
            "case_id": result.case_id,
            "category": result.category,
            "score": result.score,
            "failures": list(result.failures),
            "answer": result.answer,
        }
        for result in baseline_report.failed
    ]
    improver_instructions = """
Tu es l'optimiseur d'un agent. Améliore ses instructions sans tricher avec les tests,
sans copier mot pour mot les réponses attendues, sans supprimer les règles de prudence,
et sans rendre les réponses inutilement longues.

Retourne uniquement un objet JSON valide:
{
  "instructions": "nouvelles instructions principales",
  "principles": ["principe 1", "principe 2"],
  "rationale": "explication brève et vérifiable"
}
""".strip()
    improver_input = json.dumps(
        {
            "active_prompt": baseline_prompt.to_dict(),
            "eval_summary": baseline_report.to_dict(),
            "failed_cases": failures,
            "objective": (
                "Improve general behavior and failed capabilities. "
                "Do not mention case IDs or benchmark wording in the new prompt."
            ),
        },
        ensure_ascii=False,
        indent=2,
    )
    raw = brain.client.generate(
        model=brain.settings.model,
        instructions=improver_instructions,
        user_input=improver_input,
    )
    data = _extract_json(raw)
    instructions = str(data.get("instructions", "")).strip()
    principles_raw = data.get("principles", [])
    rationale = str(data.get("rationale", "")).strip()
    if not instructions:
        raise ValueError("Candidate instructions are empty.")
    if not isinstance(principles_raw, list) or not principles_raw:
        raise ValueError("Candidate principles must be a non-empty list.")
    principles = [str(item).strip() for item in principles_raw if str(item).strip()]
    if not principles:
        raise ValueError("Candidate principles are empty after normalization.")
    return instructions, principles, rationale or "Generated from benchmark failures."


def run_improvement_cycle(
    settings: Settings,
    *,
    auto_promote: bool = False,
    minimum_gain: float = 0.02,
    maximum_case_regression: float = 0.10,
) -> ImprovementRun:
    brain = SecondBrain(settings)
    cases = load_cases(settings.evals_path)
    active = load_prompt(settings.active_prompt_path)
    baseline_report = evaluate_prompt(brain, active, cases)
    instructions, principles, rationale = propose_candidate(brain, active, baseline_report)
    candidate_path = save_candidate(
        settings.state_dir,
        instructions=instructions,
        principles=principles,
        rationale=rationale,
        baseline_version=active.version,
    )
    candidate_prompt = load_prompt(candidate_path)
    candidate_report = evaluate_prompt(brain, candidate_prompt, cases)
    decision = promotion_gate(
        baseline_report,
        candidate_report,
        minimum_gain=minimum_gain,
        maximum_case_regression=maximum_case_regression,
    )
    promoted = False
    if auto_promote and decision.accepted:
        promote_candidate(candidate_path, settings.active_prompt_path, settings.state_dir)
        promoted = True

    report_path = settings.state_dir / "reports" / f"{candidate_path.stem}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "candidate_path": str(candidate_path),
                "promoted": promoted,
                "decision": {
                    "accepted": decision.accepted,
                    "reasons": list(decision.reasons),
                    "baseline_score": decision.baseline_score,
                    "candidate_score": decision.candidate_score,
                },
                "baseline": baseline_report.to_dict(),
                "candidate": candidate_report.to_dict(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    return ImprovementRun(
        baseline=baseline_report,
        candidate=candidate_report,
        candidate_path=candidate_path,
        decision=decision,
        promoted=promoted,
    )
