from __future__ import annotations

from datetime import UTC, datetime

from app.services.memory.recording.pipeline import process_memory_pipeline
from app.services.runtime.tasks import fire_background


def remember_user_event(
    *,
    user_id: str,
    workspace_id: str | None,
    text: str,
    evidence_message_ids: list[str] | None = None,
) -> None:
    if not text.strip():
        return
    fire_background(
        process_memory_pipeline(
            user_id,
            text.strip(),
            side="user",
            workspace_id=workspace_id,
            statement_time=datetime.now(UTC),
            evidence_message_ids=evidence_message_ids,
        )
    )


def remember_ai_event(
    *,
    user_id: str,
    workspace_id: str | None,
    text: str,
    evidence_message_ids: list[str] | None = None,
) -> None:
    if not text.strip():
        return
    fire_background(
        process_memory_pipeline(
            user_id,
            text.strip(),
            side="ai",
            workspace_id=workspace_id,
            statement_time=datetime.now(UTC),
            evidence_message_ids=evidence_message_ids,
        )
    )
