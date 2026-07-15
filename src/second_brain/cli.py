from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .core import SecondBrain
from .evaluation import evaluate_prompt, load_cases
from .improvement import run_improvement_cycle
from .memory import MemoryStore
from .prompt_store import load_prompt, promote_candidate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="second-brain",
        description="Controlled self-improving AI agent.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Ask the active agent a question.")
    ask.add_argument("message")

    remember = subparsers.add_parser("remember", help="Store an explicit long-term fact.")
    remember.add_argument("fact")
    remember.add_argument("--importance", type=float, default=1.0)

    feedback = subparsers.add_parser("feedback", help="Score a previous interaction.")
    feedback.add_argument("interaction_id", type=int)
    feedback.add_argument("score", type=float)
    feedback.add_argument("--note", default="")

    evaluate = subparsers.add_parser("eval", help="Evaluate a prompt against the benchmark.")
    evaluate.add_argument("--prompt", type=Path)

    improve = subparsers.add_parser("improve", help="Generate and evaluate a better prompt.")
    improve.add_argument("--auto-promote", action="store_true")
    improve.add_argument("--minimum-gain", type=float, default=0.02)
    improve.add_argument("--maximum-case-regression", type=float, default=0.10)

    promote = subparsers.add_parser("promote", help="Promote a reviewed candidate prompt.")
    promote.add_argument("candidate", type=Path)

    subparsers.add_parser("status", help="Show the active version and local state.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env()

    try:
        if args.command == "ask":
            answer = SecondBrain(settings).answer(args.message)
            print(answer.text)
            print(
                f"\n[interaction={answer.interaction_id} "
                f"prompt=v{answer.prompt_version} memories={answer.memories_used}]"
            )
            return 0

        if args.command == "remember":
            memory_id = MemoryStore(settings.database_path).add_fact(
                args.fact, importance=args.importance
            )
            print(f"Stored fact {memory_id}.")
            return 0

        if args.command == "feedback":
            MemoryStore(settings.database_path).record_feedback(
                args.interaction_id, args.score, args.note
            )
            print("Feedback recorded.")
            return 0

        if args.command == "eval":
            prompt_path = args.prompt or settings.active_prompt_path
            prompt = load_prompt(prompt_path)
            report = evaluate_prompt(
                SecondBrain(settings), prompt, load_cases(settings.evals_path)
            )
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
            return 0

        if args.command == "improve":
            run = run_improvement_cycle(
                settings,
                auto_promote=args.auto_promote,
                minimum_gain=args.minimum_gain,
                maximum_case_regression=args.maximum_case_regression,
            )
            print(json.dumps({
                "candidate": str(run.candidate_path),
                "baseline_score": run.baseline.weighted_score,
                "candidate_score": run.candidate.weighted_score,
                "gate_accepted": run.decision.accepted,
                "gate_reasons": list(run.decision.reasons),
                "promoted": run.promoted,
            }, ensure_ascii=False, indent=2))
            return 0

        if args.command == "promote":
            promoted = promote_candidate(
                args.candidate, settings.active_prompt_path, settings.state_dir
            )
            print(f"Promoted prompt v{promoted.version}.")
            return 0

        if args.command == "status":
            active = load_prompt(settings.active_prompt_path)
            candidates = sorted((settings.state_dir / "candidates").glob("*.json"))
            print(json.dumps({
                "model": settings.model,
                "active_prompt_version": active.version,
                "candidate_count": len(candidates),
                "database": str(settings.database_path),
            }, ensure_ascii=False, indent=2))
            return 0

    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
