"""Source-clock proposals and atomic, replay-safe reminder rescheduling.

Redis stores the proposal, never permission to mutate an arbitrary memory.
Both memory and reminder are pinned at the proposal and rechecked on confirm.
"""

from datetime import datetime, timedelta, timezone
import json
import logging
import re
from uuid import uuid4

from app.db import db
from app.services.memory.evidence import (
    EvidenceSource,
    _resolve_source,
    content_version,
    message_statement_time,
)
from app.services.memory.input_evidence import (
    prepare_input_origin,
    persist_input_origin,
)
from app.services.memory.interaction.contradiction_state import (
    conversation_scope,
    memory_version,
    message_sources,
)
from app.services.memory.lifecycle.lazy_update import scope_lock_key
from app.services.memory.manual_write import MemoryWriteError, lock_memories
from app.services.memory.storage import repo
from app.services.time_domain.reminders import resolve_reminder_targets
from app.services.time_domain.timestamps import parse_time

FORMAT = "reminder-reschedule-v1"
MAX_TARGETS = 5
logger = logging.getLogger(__name__)


def _version(row):
    return content_version(
        json.dumps(row, sort_keys=True, default=str, ensure_ascii=False)
    )


_TARGET_KEYS = {"id", "source", "memory_version", "trigger_id", "trigger_version"}
_PIN_KEYS = {
    "kind", "ref", "side", "relation", "expected_version", "expected_role",
    "expected_conversation_id", "expected_card_version", "expected_statement_time",
    "input_text", "temporal_basis",
}


def _short_text(value):
    return isinstance(value, str) and 0 < len(value) <= 200


def valid_ticket_rows(ticket) -> bool:
    """Structural check of a Redis proposal before any field is dereferenced.

    Redis content is untrusted: a corrupted entry must be rejected as invalid,
    never surface as KeyError/TypeError or reach evidence construction.
    """
    if not isinstance(ticket, dict):
        return False
    targets, pins = ticket.get("targets"), ticket.get("sources")
    if (
        not isinstance(targets, list)
        or not 1 <= len(targets) <= MAX_TARGETS
        or not isinstance(pins, list)
        or not 1 <= len(pins) <= 40
        or not isinstance(ticket.get("scope"), dict)
    ):
        return False
    for t in targets:
        if (
            not isinstance(t, dict)
            or set(t) != _TARGET_KEYS
            or t["source"] not in {"user", "ai"}
            or not all(_short_text(t[k]) for k in _TARGET_KEYS - {"source"})
        ):
            return False
    if len({(t["source"], t["id"]) for t in targets}) != len(targets):
        return False
    for pin in pins:
        if (
            not isinstance(pin, dict)
            or not set(pin) <= _PIN_KEYS
            or pin.get("kind") != "message"
            or not _short_text(pin.get("ref"))
            or pin.get("expected_role") != "user"
            or pin.get("expected_conversation_id") != ticket.get("conversation_id")
            or not _short_text(pin.get("expected_version"))
            or parse_time(pin.get("expected_statement_time")) is None
        ):
            return False
    return len({p["ref"] for p in pins}) == len(pins)


async def _triggers(database, *, user_id, agent_id, memory_id, side, lock=False):
    return await database.query_raw(
        """SELECT id,trigger_time,is_active,last_fired,action_data FROM time_triggers
        WHERE user_id=$1 AND ai_agent_id=$2 AND action_type='reminder' AND is_active
          AND action_data->>'memory_id'=$3 AND COALESCE(action_data->>'memory_side','user')=$4
        ORDER BY id LIMIT 2""" + (" FOR UPDATE" if lock else ""),
        user_id,
        agent_id,
        memory_id,
        side,
    )


async def prepare_ticket(
    *,
    candidates,
    conversation_id,
    user_id,
    workspace_id,
    agent_id,
    message_ids,
    user_message,
    now=None,
):
    now = now or datetime.now(timezone.utc)
    scope_args = dict(
        conversation_id=conversation_id,
        user_id=user_id,
        workspace_id=workspace_id,
        agent_id=agent_id,
    )
    scope = await conversation_scope(db, **scope_args)
    sources = await message_sources(
        db, conversation_id=conversation_id, message_ids=message_ids, text=user_message
    )
    # Strip only an explicit target marker. Do not choose an arbitrary one of
    # several dates or trust the classifier's generated ISO timestamp.
    markers = list(
        re.finditer(r"改到|改成|改为|挪到|推迟到|提前到|调整到", user_message)
    )
    if len(markers) > 1 or re.search(
        r"[“”\"「」『』]|还是|或者|或是|如果|假如|也许|可能|(?:他|她|别人|同事|朋友).{0,6}说|不要改|不用改|别改|不改",
        user_message,
    ):
        raise MemoryWriteError(422, "reschedule_time_ambiguous")
    target_text = user_message[markers[0].end() :] if markers else user_message
    clocks = [parse_time(s["expected_statement_time"]) for s in sources]
    resolution = resolve_reminder_targets(
        target_text,
        statement_at=min(clocks),
        statement_end_at=max(clocks),
        now=now,
        allow_multiple=True,
    )
    if resolution.reason != "resolved" or len(resolution.targets) != 1:
        raise MemoryWriteError(422, "reschedule_time_unresolved")
    if not 1 <= len(candidates) <= MAX_TARGETS:
        raise MemoryWriteError(422, "reschedule_targets_ambiguous")
    targets, seen = [], set()
    for candidate in candidates:
        mid, side = candidate.get("id"), candidate.get("source")
        if not mid or side not in {"user", "ai"} or (side, mid) in seen:
            raise MemoryWriteError(409, "reschedule_target_invalid")
        seen.add((side, mid))
        record = await repo.find_unique(mid)
        if (
            not record
            or record.isArchived
            or record.subCategory != "提醒"
            or (record.userId, record.workspaceId, record.source)
            != (user_id, workspace_id, side)
        ):
            raise MemoryWriteError(409, "reschedule_target_changed")
        if candidate.get("content") != record.content:
            raise MemoryWriteError(409, "reschedule_target_changed")
        triggers = await _triggers(
            db, user_id=user_id, agent_id=agent_id, memory_id=mid, side=side
        )
        if len(triggers) != 1:
            raise MemoryWriteError(409, "reschedule_trigger_ambiguous_or_inactive")
        targets.append(
            {
                "id": mid,
                "source": side,
                "memory_version": memory_version(record),
                "trigger_id": triggers[0]["id"],
                "trigger_version": _version(triggers[0]),
            }
        )
    return {
        "format": FORMAT,
        "id": uuid4().hex,
        **scope_args,
        "scope": scope,
        "sources": sources,
        "targets": targets,
        "new_time": resolution.targets[0].isoformat(),
        "created_at": now.isoformat(),
    }


async def apply_ticket(
    ticket,
    *,
    conversation_id,
    user_id,
    workspace_id,
    agent_id,
    message_ids,
    user_message,
    now=None,
):
    now = now or datetime.now(timezone.utc)
    scope_args = dict(
        conversation_id=conversation_id,
        user_id=user_id,
        workspace_id=workspace_id,
        agent_id=agent_id,
    )
    if (
        not isinstance(ticket, dict)
        or ticket.get("format") != FORMAT
        or any(ticket.get(key) != value for key, value in scope_args.items())
    ):
        raise MemoryWriteError(409, "reschedule_ticket_scope_changed")
    created, target = parse_time(ticket.get("created_at")), parse_time(
        ticket.get("new_time")
    )
    if not created or not target or not 0 <= (now - created).total_seconds() <= 1800:
        raise MemoryWriteError(409, "reschedule_ticket_expired")
    if not valid_ticket_rows(ticket):
        raise MemoryWriteError(409, "reschedule_ticket_invalid")
    targets, pins = ticket["targets"], ticket["sources"]
    origin = prepare_input_origin(
        kind="user_edit", payload={"operation": FORMAT, "ticket": ticket}
    )
    origin_id = content_version(
        json.dumps([user_id, workspace_id, agent_id, origin.kind, origin.version])
    )
    confirmations = await message_sources(
        db, conversation_id=conversation_id, message_ids=message_ids, text=user_message
    )
    if {s["ref"] for s in confirmations} & {s["ref"] for s in pins}:
        raise MemoryWriteError(409, "reschedule_confirmation_must_be_new_turn")
    sources = tuple(
        EvidenceSource(
            **{
                **s,
                "expected_statement_time": message_statement_time(
                    s["expected_statement_time"]
                ),
            }
        )
        for s in [*pins, *confirmations]
    )
    async with db.tx(timeout=timedelta(seconds=20)) as tx:
        # Shared owner order with lifecycle/manual writes, then scoped rows.
        for side in sorted({t["source"] for t in targets}):
            await tx.execute_raw(
                "SELECT pg_advisory_xact_lock($1::bigint)",
                scope_lock_key(side, user_id, workspace_id),
            )
        if await conversation_scope(tx, **scope_args, lock=True) != ticket.get("scope"):
            raise MemoryWriteError(409, "reschedule_scope_changed")
        if await tx.query_raw(
            "SELECT id FROM memory_input_origins WHERE id=$1 AND user_id=$2 AND workspace_id=$3",
            origin_id,
            user_id,
            workspace_id,
        ):
            return {
                "status": "already_applied",
                "updated": len(targets),
                "new_time": target.isoformat(),
            }
        if target <= now:
            raise MemoryWriteError(409, "reschedule_target_expired")
        for source in sources:
            await _resolve_source(
                tx,
                source,
                ticket["scope"] | {"user_id": user_id, "workspace_id": workspace_id},
            )
        plaintext = await tx.query_raw(
            """SELECT id FROM messages WHERE id=ANY($1::text[])
            AND NOT COALESCE(metadata,'{}'::jsonb) ? 'component_card' FOR SHARE""",
            list({s.ref for s in sources}),
        )
        if len(plaintext) != len({s.ref for s in sources}):
            raise MemoryWriteError(409, "reschedule_source_changed")
        records = []
        for pinned in sorted(targets, key=lambda t: (t["source"], t["id"])):
            model = tx.usermemory if pinned["source"] == "user" else tx.aimemory
            row = await model.find_unique(where={"id": pinned["id"]})
            if not row:
                raise MemoryWriteError(409, "reschedule_target_deleted")
            record = repo._to_record(row, pinned["source"])
            if (record.userId, record.workspaceId) != (
                user_id,
                workspace_id,
            ) or memory_version(record) != pinned["memory_version"]:
                raise MemoryWriteError(409, "reschedule_target_changed")
            records.append(record)
        await lock_memories(tx, records)
        input_source = await persist_input_origin(
            tx,
            user_id=user_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
            origin=origin,
        )
        for record in records:
            pinned = next(
                t
                for t in targets
                if t["id"] == record.id and t["source"] == record.source
            )
            triggers = await _triggers(
                tx,
                user_id=user_id,
                agent_id=agent_id,
                memory_id=record.id,
                side=record.source,
                lock=True,
            )
            if (
                len(triggers) != 1
                or triggers[0]["id"] != pinned["trigger_id"]
                or _version(triggers[0]) != pinned["trigger_version"]
            ):
                raise MemoryWriteError(409, "reschedule_trigger_changed")
            await repo.update(
                record.id,
                source=record.source,
                record=record,
                database=tx,
                occurTime=target,
                evidence_sources=(input_source, *sources),
                extractor_version=FORMAT,
            )
            await tx.timetrigger.update(
                where={"id": triggers[0]["id"]}, data={"triggerTime": target}
            )
            await tx.memorychangelog.create(
                data={
                    "userId": user_id,
                    "workspaceId": workspace_id,
                    "memoryId": record.id,
                    "operation": "reschedule",
                    "oldValue": (
                        record.occurTime.isoformat() if record.occurTime else None
                    ),
                    "newValue": json.dumps(
                        {
                            "new_time": target.isoformat(),
                            "input_origin": input_source.ref,
                        }
                    ),
                }
            )
    # A postcommit transport outage cannot turn a committed change into failure.
    await repo.invalidate_scope(user_id, workspace_id)
    from app.services.reminder.scheduling import notify_reminder_changed

    await notify_reminder_changed(conversation_id, kind="rescheduled")
    return {
        "status": "applied",
        "updated": len(targets),
        "new_time": target.isoformat(),
    }
