from __future__ import annotations

import math
import os


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def recommended_thread_budget(
    logical_processors: int | None = None,
    *,
    reserved_processors: int | None = None,
) -> int:
    """Return a conservative CPU thread budget that leaves capacity for the web server."""
    logical = max(1, int(logical_processors or os.cpu_count() or 1))
    if reserved_processors is None:
        reserved = max(1, min(4, math.ceil(logical * 0.25)))
    else:
        reserved = max(0, int(reserved_processors))
    return max(1, logical - min(reserved, logical - 1))


def configure_cpu_thread_budget() -> int | None:
    """Apply a process-wide BLAS/OpenMP budget unless the administrator disabled it."""
    disabled = os.getenv("SECOND_BRAIN_DISABLE_CPU_BUDGET", "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return None

    requested = os.getenv("SECOND_BRAIN_CPU_THREADS", "").strip()
    if requested:
        try:
            budget = max(1, int(requested))
        except ValueError:
            budget = recommended_thread_budget()
    else:
        budget = recommended_thread_budget()

    value = str(budget)
    for variable in THREAD_ENVIRONMENT_VARIABLES:
        os.environ.setdefault(variable, value)
    os.environ["SECOND_BRAIN_EFFECTIVE_CPU_THREADS"] = value
    return budget
