"""Explicit observations of memory processing, never retry receipts."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MemorySideOutcome:
    side: Literal["user", "ai"]
    status: Literal["completed", "no_new_messages", "excluded", "failed", "disabled"]
    reason: str
    # Returned memory IDs include updates; None means the count is unknown,
    # notably when a multi-memory write failed after a partial commit.
    memory_count: int | None = None
    progress: Literal["advanced", "unchanged", "not_applicable", "unknown"] = "unchanged"
    source_count: int | None = None
    source_binding: Literal["persisted_messages", "unlinked_input", "unknown"] = "unknown"


class MemoryProgressError(RuntimeError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)
