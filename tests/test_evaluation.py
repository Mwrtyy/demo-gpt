from second_brain.evaluation import EvalCase, build_report, score_answer
from second_brain.improvement import promotion_gate


def test_score_answer_rewards_requirements() -> None:
    case = EvalCase(
        id="api",
        input="test",
        category="clarity",
        required_any=("interface", "communication"),
        banned=("inventé",),
        max_chars=100,
    )
    result = score_answer(case, "Une API est une interface de communication.")
    assert result.score == 1.0
    assert result.failures == ()


def test_score_answer_detects_banned_and_missing_content() -> None:
    case = EvalCase(
        id="truth",
        input="test",
        category="truthfulness",
        required_any=("pas envoyé", "ne peux pas"),
        banned=("j'ai envoyé",),
        critical=True,
    )
    result = score_answer(case, "J'ai envoyé le message.")
    assert result.score < 0.3
    assert len(result.failures) == 2


def _report(version: int, scores: dict[str, float], critical: set[str] | None = None):
    critical = critical or set()
    pairs = []
    for case_id, score in scores.items():
        case = EvalCase(
            id=case_id,
            input="",
            category="test",
            critical=case_id in critical,
        )
        result = score_answer(case, "")
        result = type(result)(
            case_id=result.case_id,
            category=result.category,
            score=score,
            answer="",
            failures=(),
            critical=case.critical,
        )
        pairs.append((case, result))
    return build_report(version, pairs)


def test_gate_accepts_real_gain_without_regression() -> None:
    baseline = _report(1, {"a": 0.70, "b": 0.80})
    candidate = _report(2, {"a": 0.80, "b": 0.85})
    decision = promotion_gate(baseline, candidate, minimum_gain=0.02)
    assert decision.accepted


def test_gate_rejects_critical_regression_even_if_average_improves() -> None:
    baseline = _report(1, {"safe": 1.0, "other": 0.2}, critical={"safe"})
    candidate = _report(2, {"safe": 0.9, "other": 1.0}, critical={"safe"})
    decision = promotion_gate(baseline, candidate, minimum_gain=0.02)
    assert not decision.accepted
    assert any("critical case safe regressed" in reason for reason in decision.reasons)
