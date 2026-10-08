"""One fail-closed gate for chat SQL activation, independent of graph rollout."""

# R01.03.8.3-.6 and R01.06.7/.8, R01.07/.08 must qualify the full topology.
# This is a code qualification gate, never a client or environment override.
SQL_CHAT_ACTIVATION_READY = False


def validate_chat_ingress_backend(backend: str) -> None:
    if backend not in {"redis", "sql"}:
        raise RuntimeError("Unsupported chat ingress backend")
    if backend == "sql" and not SQL_CHAT_ACTIVATION_READY:
        raise RuntimeError("SQL chat activation has not passed consumer/delivery/handover gates")


def sql_chat_ingress_enabled() -> bool:
    from app.config import settings

    validate_chat_ingress_backend(settings.chat_ingress_backend)
    return settings.chat_ingress_backend == "sql"
