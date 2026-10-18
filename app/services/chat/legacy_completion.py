"""Invocation-owned completion bridge for the recursive legacy executor.

Children inherit the parent's collector, but cannot commit it. Persistence is
awaited and only the parent done event registers the complete turn. No bodies
or reusable process-wide state are retained here.
"""
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class LegacyCompletion:
    conversation_id: str
    source: dict | None = None
    failed: bool = False
    completed: bool = False

    def record(self, conversation_id, message_id, turn_id, user_message_ids):
        if conversation_id != self.conversation_id or not message_id:
            self.failed = True
            raise RuntimeError("legacy_turn_persistence_failed")
        source = dict(conversation_id=conversation_id, message_id=message_id,
                      turn_id=turn_id, user_message_ids=sorted(set(user_message_ids or [])))
        if self.source is not None:
            if any(source[key] != self.source[key] for key in ("turn_id", "user_message_ids")):
                self.failed = True
                raise RuntimeError("legacy_turn_source_mismatch")
        else:
            self.source = source

    async def finish(self):
        from app.services.runtime.turn_followups import register_completed_turn

        if self.failed:
            raise RuntimeError("legacy_turn_incomplete")
        if not self.completed and self.source is not None:
            await register_completed_turn(**self.source)
        self.completed = True


current_completion: ContextVar[LegacyCompletion | None] = ContextVar(
    "legacy_turn_completion", default=None,
)
