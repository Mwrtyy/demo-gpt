"""Early process configuration for Second Brain local Python environments."""

try:
    from second_brain.cpu_budget import configure_cpu_thread_budget

    configure_cpu_thread_budget()
except Exception:
    # Site customization must never prevent Python itself from starting.
    pass
