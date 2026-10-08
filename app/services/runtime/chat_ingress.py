"""Transactional SQL chat acceptance, staged until consumers are ready.

No endpoint, scheduler, Redis queue or provider is switched by importing this
module. A successful return means the SQL transaction committed, not that a reply
was produced. Callers must not publish ack or execute hooks before that return.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from typing import Any
from uuid import uuid4

from app.services.runtime.chat_ingress_contracts import (
    MAX_TURN_CHARS, MAX_TURN_MESSAGES, ChatAggregationPolicy,
    ChatExecutionSnapshot, ChatRequestInput, PreparedChatMessage, canonical_object,
)
from app.services.runtime.execution_scope import (
    ExecutionScope, ExecutionScopeExpired, ExecutionScopeUnavailable,
    scoped_transaction,
)
from app.services.runtime.chat_ingress_effects import ChatIngressEffects


CHAT_JOB_HANDLER = "chat.execute.v1"
CHAT_JOB_KEY = "chat:0"
_SCOPE_NAMES = tuple(ExecutionScope.__dataclass_fields__)


class ChatRequestConflict(ValueError):
    def __init__(self) -> None:
        super().__init__("Request identity was already used with different input")


class ChatIngressCorrupt(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Stored chat ingress is inconsistent")


@dataclass(frozen=True, slots=True)
class AcceptedChatMessage:
    message_id: str
    run_id: str
    job_id: str
    ordinal: int
    created: bool
    client_id: str | None
    run_status: str
    job_status: str
    phase: str
    available_at: datetime
    result_json: str = "{}"
    error_json: str = "{}"


@dataclass(frozen=True, slots=True)
class StoredChatTurn:
    run_id: str
    job_id: str
    message_ids: tuple[str, ...]
    prompt_text: str
    reply_context_json: str
    snapshot: ChatExecutionSnapshot
    first_ordinal: int
    last_ordinal: int
    phase: str
    available_at: datetime


def _object(value: Any) -> dict:
    value = json.loads(value) if isinstance(value, str) else value
    if type(value) is not dict:
        raise ChatIngressCorrupt()
    return value


def _time(value: Any) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ChatIngressCorrupt()
    return value.astimezone(timezone.utc)


def _check_scope(row: dict, scope: ExecutionScope) -> None:
    if ExecutionScope(**{name: row[name] for name in _SCOPE_NAMES}) != scope:
        raise ExecutionScopeExpired()


def _ingress(row: dict) -> dict:
    state = _object(row["state"])
    ingress = state.get("ingress")
    if (type(ingress) is not dict or type(ingress.get("version")) is not int or ingress.get("version") != 1
            or ingress.get("phase") not in {"collecting", "ready"}):
        raise ChatIngressCorrupt()
    return ingress


async def _lock_conversation(tx: Any, scope: ExecutionScope) -> None:
    # Serialize acceptance/sealing only, never hold this across model/network I/O.
    # 64-bit hash collisions merely serialize unrelated conversations. All SQL
    # consumers must respect this scope -> advisory -> run -> job lock order.
    await tx.query_raw(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))::text AS held",
        "chat-ingress:" + scope.conversation_id,
    )


async def _now(tx: Any) -> datetime:
    return _time((await tx.query_raw("SELECT clock_timestamp() AS now"))[0]["now"])


async def _request_row(tx: Any, scope: ExecutionScope, request: ChatRequestInput) -> dict | None:
    rows = await tx.query_raw(
        "SELECT q.message_id, q.run_id, q.job_id, q.ordinal, q.client_id, "
        "q.request_input, q.input_fingerprint, q.conversation_id AS receipt_conversation_id, "
        "m.conversation_id AS message_conversation_id, m.role AS message_role, "
        "r.state, r.result, r.error, r.status AS run_status, j.status AS job_status, j.available_at, "
        "j.run_id AS job_run_id, " + ", ".join("r." + name for name in _SCOPE_NAMES) +
        " FROM chat_ingress_receipts q JOIN agent_runs r ON r.id=q.run_id "
        "JOIN runtime_jobs j ON j.id=q.job_id JOIN messages m ON m.id=q.message_id "
        "WHERE q.conversation_id=$1 AND q.request_key=$2",
        scope.conversation_id, request.request_key,
    )
    if not rows:
        return None
    row = rows[0]
    _check_scope(row, scope)
    if (row["receipt_conversation_id"] != scope.conversation_id
            or row["message_conversation_id"] != scope.conversation_id
            or row["message_role"] != "user" or row["job_run_id"] != row["run_id"]):
        raise ChatIngressCorrupt()
    # Check the original canonical body as well as the hash. Prepared rendering,
    # current prompt versions or capability flags are not retry identity.
    if (row["input_fingerprint"] != request.fingerprint
            or canonical_object(_object(row["request_input"])) != request.input_json):
        raise ChatRequestConflict()
    return row


def _accepted(row: dict, *, created: bool) -> AcceptedChatMessage:
    return AcceptedChatMessage(
        row["message_id"], row["run_id"], row["job_id"], int(row["ordinal"]),
        created, row["client_id"], row["run_status"], row["job_status"],
        _ingress(row)["phase"], _time(row["available_at"]),
        canonical_object(_object(row["result"])), canonical_object(_object(row["error"])),
    )


async def lookup_chat_request(scope: ExecutionScope, request: ChatRequestInput,
                              *, database: Any = None) -> AcceptedChatMessage | None:
    """Early retry lookup; absence is not a reservation or quota authorization."""
    async with scoped_transaction(scope, database=database) as tx:
        row = await _request_row(tx, scope, request)
        result = _accepted(row, created=False) if row else None
    return result


async def _open_run(tx: Any, scope: ExecutionScope) -> dict | None:
    # Raw generation casts are explicit: the runtime models are client-ignored.
    parameters = tuple(asdict(scope).values())
    predicate = " AND ".join(
        "r." + name + "=$" + str(index) + ("::uuid" if name.endswith("_generation") else "")
        for index, name in enumerate(_SCOPE_NAMES, 1)
    )
    rows = await tx.query_raw(
        "SELECT r.*, j.id AS job_id, j.status AS job_status, j.available_at "
        "FROM agent_runs r JOIN runtime_jobs j ON j.run_id=r.id "
        "WHERE " + predicate + " AND r.kind='chat' AND r.status='queued' "
        "AND r.state #>> '{ingress,phase}'='collecting' "
        "AND j.job_key='chat:0' AND j.handler='chat.execute.v1' "
        "AND j.status='pending' AND j.attempts=0 "
        "ORDER BY r.created_at DESC, r.id DESC LIMIT 1 FOR UPDATE OF r,j",
        *parameters,
    )
    return rows[0] if rows else None


def _policy(ingress: dict) -> ChatAggregationPolicy:
    return ChatAggregationPolicy(**ingress["policy"])


def _available(window_due: datetime, policy: ChatAggregationPolicy) -> datetime:
    return window_due + timedelta(seconds=policy.delay_seconds)


async def _save_state(tx: Any, run: dict, state: dict, available_at: datetime) -> None:
    # Locked pending/queued rows only. Never reset an attempted, leased or terminal
    # job. R01.04 will claim under the same conversation lock and close the window.
    changed = await tx.execute_raw(
        "UPDATE agent_runs SET state=$2::jsonb, updated_at=clock_timestamp() "
        "WHERE id=$1 AND status='queued'", run["id"], canonical_object(state),
    )
    if changed != 1:
        raise ChatIngressCorrupt()
    changed = await tx.execute_raw(
        "UPDATE runtime_jobs SET available_at=$2::timestamptz, updated_at=clock_timestamp() "
        "WHERE id=$1 AND status='pending' AND attempts=0", run["job_id"], available_at.isoformat(),
    )
    if changed != 1:
        raise ChatIngressCorrupt()


async def _seal(tx: Any, run: dict, closed_at: datetime) -> None:
    state = _object(run["state"])
    ingress = _ingress(run)
    ingress["phase"] = "ready"
    ingress["window_due_at"] = closed_at.isoformat()
    state["ingress"] = ingress
    await _save_state(tx, run, state, _available(closed_at, _policy(ingress)))


async def _new_run(tx: Any, scope: ExecutionScope, request: ChatRequestInput,
                   snapshot: ChatExecutionSnapshot, policy: ChatAggregationPolicy,
                   now: datetime) -> dict:
    run_id, job_id = str(uuid4()), str(uuid4())
    due = now + timedelta(seconds=policy.quiet_seconds)
    available_at = _available(due, policy)
    if snapshot.deadline_at is not None and snapshot.deadline_at <= available_at:
        raise ValueError("Execution deadline precedes scheduled availability")
    state = {"ingress": {"version": 1, "phase": "ready" if policy.mode == "immediate" else "collecting",
                          "executor": snapshot.executor, "policy": asdict(policy),
                          "first_accepted_at": now.isoformat(), "window_due_at": due.isoformat(),
                          "message_count": 0, "prompt_chars": 0}}
    columns = (*_SCOPE_NAMES, "id", "kind", "request_key", "input_fingerprint", "input",
               "graph_version", "state_version", "config_snapshot", "prompt_snapshot",
               "budget_snapshot", "deadline_at", "state", "created_at", "updated_at")
    values = (*asdict(scope).values(), run_id, "chat", request.request_key, request.fingerprint,
              request.input_json, snapshot.graph_version, snapshot.state_version,
              snapshot.config_json, snapshot.prompts_json, snapshot.budget_json,
              snapshot.deadline_at.isoformat() if snapshot.deadline_at else None,
              canonical_object(state), now.isoformat(), now.isoformat())
    json_columns = {"input", "config_snapshot", "prompt_snapshot", "budget_snapshot", "state"}
    placeholders = ["$" + str(i) + ("::uuid" if name.endswith("_generation") else
                    "::jsonb" if name in json_columns else
                    "::timestamptz" if name.endswith("_at") else "")
                    for i, name in enumerate(columns, 1)]
    rows = await tx.query_raw(
        "INSERT INTO agent_runs (" + ",".join(columns) + ") VALUES (" +
        ",".join(placeholders) + ") RETURNING *", *values,
    )
    await tx.execute_raw(
        "INSERT INTO runtime_jobs (id,run_id,job_key,handler,payload,queue,max_attempts,"
        "available_at,created_at,updated_at) VALUES ($1,$2,'chat:0','chat.execute.v1',"
        "$3::jsonb,'foreground',$4,$5::timestamptz,$6::timestamptz,$6::timestamptz)",
        job_id, run_id, canonical_object({"schema_version": 1, "run_id": run_id,
                                         "input_source": "chat_ingress_receipts"}),
        snapshot.max_attempts, available_at.isoformat(), now.isoformat(),
    )
    return {**rows[0], "job_id": job_id, "job_status": "pending", "available_at": available_at}


async def _target_run(tx: Any, scope: ExecutionScope, request: ChatRequestInput,
                      message: PreparedChatMessage, snapshot: ChatExecutionSnapshot,
                      policy: ChatAggregationPolicy, now: datetime) -> dict:
    run = await _open_run(tx, scope)
    if run:
        ingress = _ingress(run)
        old_policy = _policy(ingress)
        due = _time(ingress["window_due_at"])
        compatible = (run["graph_version"] == snapshot.graph_version
                      and run["state_version"] == snapshot.state_version
                      and ingress["executor"] == snapshot.executor)
        bounded = (ingress["message_count"] < MAX_TURN_MESSAGES
                   and ingress["prompt_chars"] + len(message.prompt_text) + ingress["message_count"] <= MAX_TURN_CHARS)
        join = (compatible and bounded and policy.allow_join and
                (old_policy.mode == "fragment_window" or policy.mode != "immediate"))
        if due <= now or not compatible or not bounded:
            # A new input cannot extend an expired window. Stop collecting the
            # previous turn; keep its original deadline, versions and receipts.
            await _seal(tx, run, min(due, now))
        elif join:
            return run
    return await _new_run(tx, scope, request, snapshot, policy, now)


async def accept_chat_message(scope: ExecutionScope, request: ChatRequestInput,
                               message: PreparedChatMessage, snapshot: ChatExecutionSnapshot,
                               policy: ChatAggregationPolicy, *, database: Any = None,
                               effects: ChatIngressEffects | None = None) -> AcceptedChatMessage:
    """Commit message + immutable receipt + Run/Job, or return the original retry.

    Prepared metadata and snapshots must come from trusted domain services. No
    provider/achievement side effects are performed here. The explicit effects
    adapter commits quota and resource bindings under the receipt's retry guard.
    Endpoint integration remains gated with SQL consumers and delivery.
    """
    if not (isinstance(request, ChatRequestInput) and isinstance(message, PreparedChatMessage)
            and isinstance(snapshot, ChatExecutionSnapshot) and isinstance(policy, ChatAggregationPolicy)):
        raise TypeError("Prepared ingress contracts are required")
    if effects is not None and type(effects) is not ChatIngressEffects:
        raise TypeError("Trusted chat effects adapter is required")
    async with scoped_transaction(scope, database=database) as tx:
        await _lock_conversation(tx, scope)
        row = await _request_row(tx, scope, request)
        if row:
            result = _accepted(row, created=False)
        else:
            now = await _now(tx)
            run = await _target_run(tx, scope, request, message, snapshot, policy, now)
            message_id, receipt_id = str(uuid4()), str(uuid4())
            metadata = _object(message.metadata_json)
            if request.client_id is not None:
                metadata["client_id"] = request.client_id
            else:
                metadata.pop("client_id", None)
            await tx.execute_raw(
                "INSERT INTO messages (id,conversation_id,role,content,metadata,created_at) "
                "VALUES ($1,$2,'user',$3,$4::jsonb,$5::timestamptz)",
                message_id, scope.conversation_id, message.persisted_text,
                canonical_object(metadata), message.received_at.isoformat(),
            )
            if effects is not None:
                metadata["ingress_effects"] = await effects.commit(tx, scope, request, message_id, metadata)
                await tx.execute_raw("UPDATE messages SET metadata=$2::jsonb WHERE id=$1",
                                     message_id, canonical_object(metadata))
            reply_context = _object(message.reply_context_json)
            reply_context.setdefault("received_at", message.received_at.isoformat())
            prepared = canonical_object({"prompt_text": message.prompt_text,
                                         "reply_context": reply_context})
            receipt = (await tx.query_raw(
                "INSERT INTO chat_ingress_receipts (id,conversation_id,request_key,source,client_id,"
                "input_fingerprint,request_input,prepared_input,message_id,run_id,job_id,received_at,accepted_at) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8::jsonb,$9,$10,$11,$12::timestamptz,$13::timestamptz) "
                "RETURNING ordinal", receipt_id, scope.conversation_id, request.request_key,
                request.source, request.client_id, request.fingerprint, request.input_json,
                prepared, message_id, run["id"], run["job_id"],
                message.received_at.isoformat(), now.isoformat(),
            ))[0]
            state = _object(run["state"])
            ingress = _ingress(run)
            ordinal = int(receipt["ordinal"])
            ingress.setdefault("first_ordinal", ordinal)
            ingress["last_ordinal"] = ordinal
            ingress["message_count"] += 1
            ingress["prompt_chars"] += len(message.prompt_text)
            base_policy = _policy(ingress)
            if ingress["phase"] == "collecting":
                if base_policy.mode == "fragment_window" and policy.mode != "fragment_window":
                    ingress["phase"] = "ready"
                    due = now
                else:
                    due = now + timedelta(seconds=base_policy.quiet_seconds)
                    if base_policy.max_wait_seconds is not None:
                        due = min(due, _time(ingress["first_accepted_at"]) +
                                  timedelta(seconds=base_policy.max_wait_seconds))
                ingress["window_due_at"] = due.isoformat()
            available_at = _available(_time(ingress["window_due_at"]), base_policy)
            if run["deadline_at"] is not None and available_at >= _time(run["deadline_at"]):
                raise ValueError("Aggregation exceeds the execution deadline")
            state["ingress"] = ingress
            await _save_state(tx, run, state, available_at)
            result = AcceptedChatMessage(message_id, run["id"], run["job_id"], ordinal,
                                         True, request.client_id, "queued", "pending",
                                         ingress["phase"], available_at)
    # Leaving the context commits. Do not expose any receipt before it succeeds.
    return result


async def _run_row(tx: Any, scope: ExecutionScope, run_id: str) -> dict:
    rows = await tx.query_raw(
        "SELECT r.*, j.id AS job_id, j.status AS job_status, j.max_attempts, j.available_at "
        "FROM agent_runs r JOIN runtime_jobs j ON j.run_id=r.id "
        "WHERE r.id=$1 AND r.conversation_id=$2 AND r.kind='chat' "
        "AND j.job_key='chat:0' AND j.handler='chat.execute.v1' FOR UPDATE OF r,j",
        run_id, scope.conversation_id,
    )
    if not rows:
        raise ExecutionScopeUnavailable()
    _check_scope(rows[0], scope)
    return rows[0]


async def seal_due_chat_turn(scope: ExecutionScope, run_id: str,
                             *, database: Any = None) -> bool:
    """Close a due window without claiming/executing it; consumers are R01.04."""
    async with scoped_transaction(scope, database=database) as tx:
        await _lock_conversation(tx, scope)
        run = await _run_row(tx, scope, run_id)
        ingress = _ingress(run)
        if ingress["phase"] == "ready":
            ready = True
        elif (run["status"] != "queued" or run["job_status"] != "pending"
              or _time(ingress["window_due_at"]) > await _now(tx)):
            ready = False
        else:
            await _seal(tx, run, _time(ingress["window_due_at"]))
            ready = True
    return ready


async def load_chat_turn(scope: ExecutionScope, run_id: str,
                         *, database: Any = None) -> StoredChatTurn:
    """Read ordered source inputs, preserving the first turn's execution snapshot.

    This is not a claim or authority to execute side effects. R01.04 must fence
    execution/commits with the job lease. Physical source deletion fails closed.
    """
    from app.services.interaction.reply_context import merge_reply_contexts
    from app.services.interaction.turn_coalescing import coalesce_turn_messages

    async with scoped_transaction(scope, database=database) as tx:
        await _lock_conversation(tx, scope)
        run = await _run_row(tx, scope, run_id)
        ingress = _ingress(run)
        rows = await tx.query_raw(
            "SELECT q.ordinal,q.message_id,q.prepared_input,m.conversation_id,m.role "
            "FROM chat_ingress_receipts q JOIN messages m ON m.id=q.message_id "
            "WHERE q.run_id=$1 ORDER BY q.ordinal LIMIT $2", run_id, MAX_TURN_MESSAGES + 1,
        )
        if (not rows or len(rows) != ingress["message_count"]
                or len(rows) > MAX_TURN_MESSAGES
                or int(rows[0]["ordinal"]) != ingress["first_ordinal"]
                or int(rows[-1]["ordinal"]) != ingress["last_ordinal"]
                or any(row["conversation_id"] != scope.conversation_id or row["role"] != "user" for row in rows)):
            raise ChatIngressCorrupt()
        texts, context = [], None
        for row in rows:
            prepared = _object(row["prepared_input"])
            texts.append(prepared["prompt_text"])
            context = merge_reply_contexts(context, prepared["reply_context"])
        if sum(map(len, texts)) != ingress["prompt_chars"]:
            raise ChatIngressCorrupt()
        if _policy(ingress).mode == "fragment_window":
            prompt_text = "".join(texts)
        else:
            prompt_text = "\n".join(coalesce_turn_messages(texts).texts)
        snapshot = ChatExecutionSnapshot.capture(
            executor=ingress["executor"], graph_version=run["graph_version"],
            state_version=run["state_version"], config=_object(run["config_snapshot"]),
            prompts=_object(run["prompt_snapshot"]), budget=_object(run["budget_snapshot"]),
            deadline_at=_time(run["deadline_at"]) if run["deadline_at"] else None,
            max_attempts=run["max_attempts"],
        )
        result = StoredChatTurn(run_id, run["job_id"], tuple(row["message_id"] for row in rows),
                                prompt_text, canonical_object(context or {}), snapshot,
                                int(rows[0]["ordinal"]), int(rows[-1]["ordinal"]),
                                ingress["phase"], _time(run["available_at"]))
    return result
