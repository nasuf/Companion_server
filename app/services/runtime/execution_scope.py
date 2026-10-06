"""Server-bound ownership snapshot for future durable conversation execution.

A scope is persisted internal state, never a client credential. Bind the actor
from verified authentication or a trusted internal entry point, not a job payload.
Redis, model output and checkpoint state cannot grant ownership/admin authority.
Read revalidation detects stale work; scoped_transaction additionally fences a
short SQL commit against concurrent lifecycle updates. No locks across LLM calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, fields
from datetime import timedelta
from typing import Any
from uuid import UUID

from app.db import db


class ExecutionScopeUnavailable(RuntimeError):
    """Resource is missing, inactive, inconsistent or inaccessible to the actor."""

    def __init__(self) -> None:
        # Do not reveal another owner's identifiers or database details.
        super().__init__("Execution resources are unavailable")


class ExecutionScopeExpired(ExecutionScopeUnavailable):
    """A resource/actor lifecycle changed since the execution was bound."""


@dataclass(frozen=True, slots=True)
class ExecutionScope:
    actor_user_id: str
    owner_user_id: str
    agent_id: str
    workspace_id: str
    conversation_id: str
    actor_generation: str
    owner_generation: str
    agent_generation: str
    workspace_generation: str
    conversation_generation: str

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ValueError("Invalid execution scope field")
            if field.name.endswith("_generation"):
                # Compare a canonical value, avoiding alternative UUID spellings.
                if str(UUID(value)) != value:
                    raise ValueError("Invalid execution generation")

    def to_record(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ExecutionScope:
        """Decode trusted storage only; decoding does not authenticate an actor."""
        names = {field.name for field in fields(cls)}
        if (not isinstance(record, Mapping) or set(record) != names | {"schema_version"}
                or type(record["schema_version"]) is not int or record["schema_version"] != 1):
            raise ValueError("Unsupported execution scope record")
        return cls(**{name: record[name] for name in names})


# One statement gives a coherent snapshot. Every edge must agree, including
# agent/workspace ownership; independently valid foreign keys are insufficient.
_SCOPE_SQL = """
SELECT actor.id AS actor_user_id, owner.id AS owner_user_id,
       agent.id AS agent_id, workspace.id AS workspace_id,
       conversation.id AS conversation_id,
       actor.execution_generation AS actor_generation,
       owner.execution_generation AS owner_generation,
       agent.execution_generation AS agent_generation,
       workspace.execution_generation AS workspace_generation,
       conversation.execution_generation AS conversation_generation
FROM conversations AS conversation
JOIN users AS owner ON owner.id = conversation.user_id
JOIN ai_agents AS agent ON agent.id = conversation.agent_id
    AND agent.user_id = owner.id
JOIN chat_workspaces AS workspace ON workspace.id = conversation.workspace_id
    AND workspace.user_id = owner.id AND workspace.agent_id = agent.id
JOIN users AS actor ON actor.id = $1
WHERE conversation.id = $2
  AND (actor.id = owner.id OR actor.role = 'admin')
  AND actor.status = 'active' AND actor.archived_at IS NULL
  AND owner.status = 'active' AND owner.archived_at IS NULL
  AND agent.status = 'active' AND agent.archived_at IS NULL
  AND workspace.status = 'active' AND workspace.archived_at IS NULL
  AND conversation.is_deleted = false AND conversation.archived_at IS NULL
"""


async def _read_scope(database: Any, actor_user_id: str, conversation_id: str,
                      *, lock: bool = False) -> ExecutionScope:
    if (not isinstance(actor_user_id, str) or not actor_user_id
            or not isinstance(conversation_id, str) or not conversation_id):
        raise ExecutionScopeUnavailable()
    sql = _SCOPE_SQL
    if lock:
        # KEY SHARE would still permit status/generation changes. SHARE blocks
        # all updates/deletes until this short transaction commits or rolls back.
        sql += " FOR SHARE OF actor, owner, agent, workspace, conversation"
    rows = await database.query_raw(sql, actor_user_id, conversation_id)
    if len(rows) != 1:
        raise ExecutionScopeUnavailable()
    return ExecutionScope(**rows[0])


async def bind_conversation_scope(*, actor_user_id: str, conversation_id: str,
                                  database: Any = None) -> ExecutionScope:
    """Resolve all resource IDs/versions and admin authority from current SQL."""
    return await _read_scope(db if database is None else database,
                             actor_user_id, conversation_id)


async def revalidate_scope(scope: ExecutionScope, *, database: Any = None) -> None:
    """Before resume/external preparation; this read alone is not a commit fence."""
    current = await bind_conversation_scope(
        actor_user_id=scope.actor_user_id, conversation_id=scope.conversation_id,
        database=database,
    )
    if current != scope:
        raise ExecutionScopeExpired()


@asynccontextmanager
async def scoped_transaction(scope: ExecutionScope, *, database: Any = None) -> AsyncIterator[Any]:
    """Validate, fence and perform SQL writes on the yielded transaction client.

    Use only for a short final commit. No model/provider/network calls, no nested
    transactions, and no writes through the global db inside the block. Exceptions
    (including lock/connection failures) propagate, rolling back the whole block.
    This cannot fence external effects or unconverted writers; those need R01/R03.
    """
    database = db if database is None else database
    async with database.tx(max_wait=timedelta(seconds=2), timeout=timedelta(seconds=5)) as tx:
        await tx.execute_raw("SET LOCAL lock_timeout = '1s'")
        await tx.execute_raw("SET LOCAL statement_timeout = '2500ms'")
        current = await _read_scope(tx, scope.actor_user_id, scope.conversation_id, lock=True)
        if current != scope:
            raise ExecutionScopeExpired()
        yield tx
