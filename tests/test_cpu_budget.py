from __future__ import annotations

from second_brain.cpu_budget import (
    THREAD_ENVIRONMENT_VARIABLES,
    configure_cpu_thread_budget,
    recommended_thread_budget,
)


def test_recommended_thread_budget_reserves_capacity() -> None:
    assert recommended_thread_budget(16) == 12
    assert recommended_thread_budget(8) == 6
    assert recommended_thread_budget(2) == 1
    assert recommended_thread_budget(1) == 1


def test_explicit_thread_budget_is_applied(monkeypatch) -> None:
    for variable in THREAD_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("SECOND_BRAIN_CPU_THREADS", "5")
    monkeypatch.delenv("SECOND_BRAIN_DISABLE_CPU_BUDGET", raising=False)

    assert configure_cpu_thread_budget() == 5
    for variable in THREAD_ENVIRONMENT_VARIABLES:
        assert __import__("os").environ[variable] == "5"


def test_budget_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SECOND_BRAIN_DISABLE_CPU_BUDGET", "1")
    assert configure_cpu_thread_budget() is None
