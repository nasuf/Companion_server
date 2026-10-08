"""One explicit registration path for API fallback and background consumers."""


def register_runtime_handlers() -> tuple[str, ...]:
    from app.services.agent_initialization import _run_agent_initialization_job
    from app.services.memory.generation_lock import MEMORY_GENERATION_LOCK_TTL_S
    from app.services.runtime.job_queue import register_job_handler

    # Audited historical initialization was always enqueued without delay_s.
    register_job_handler(
        "agent_initialization", _run_agent_initialization_job,
        legacy_no_delay=True, recovery_delay_s=MEMORY_GENERATION_LOCK_TTL_S,
    )
    return ("agent_initialization",)
