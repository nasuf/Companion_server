"""SQL-only effects fenced by the same receipt as their memory write.

Reminder preparation may read a schedule outside the transaction. The final
transaction checks the retained memory version and never resets a fired trigger.
"""

from datetime import datetime, time, timedelta
from hashlib import sha256
import json
from uuid import uuid5, NAMESPACE_URL

from prisma import Json
from app.db import db
from app.services.memory.recording import receipts
from app.services.memory.storage import entity_repo
from app.services.schedule_domain.time_service import ensure_aware


async def finish_followups(
    token,
    *,
    mem,
    entity_types,
    batch_preferences,
    user_emphasized,
    reminder_demoted_reason,
):
    table = "memories_ai" if token.side == "ai" else "memories_user"
    rows = await db.query_raw(
        f"""SELECT r.*,m.content,m.sub_category,m.occur_time,m.recurrence
        FROM memory_write_receipts r JOIN {table} m ON m.id=r.memory_id
        WHERE r.batch_id=$1 AND r.fact_index=$2 AND r.status='written'
          AND NOT r.followup_done ORDER BY r.piece_index""",
        token.batch_id,
        token.fact_index,
    )
    for prepared in rows:
        trigger_time = prepared["occur_time"]
        if (
            token.side == "user"
            and prepared["sub_category"] == "提醒"
            and trigger_time is not None
        ):
            trigger_time = ensure_aware(
                datetime.fromisoformat(trigger_time)
                if isinstance(trigger_time, str)
                else trigger_time
            )
            # PostgreSQL returns UTC. Detect a date-only midnight in the frozen
            # event's calendar, not at 00:00 UTC or today's configured timezone.
            frozen_event = receipts.saved_clock(mem).occur_time
            if frozen_event is not None:
                trigger_time = trigger_time.astimezone(frozen_event.tzinfo)
            if trigger_time.time() == time(0):
                from app.services.proactive.special_dates import (
                    find_first_idle_after_wakeup,
                )
                from app.services.schedule_domain.schedule import get_cached_schedule

                schedule = await get_cached_schedule(token.scope.agent_id)
                trigger_time = find_first_idle_after_wakeup(
                    schedule, trigger_time.date()
                )
        piece = (
            token.piece(prepared["piece_index"])
            if prepared["piece_index"] >= 0
            else token
        )
        async with db.tx(timeout=timedelta(seconds=15)) as tx:
            receipt = await receipts.claim(tx, piece, prepared["input_hash"])
            if (
                receipt is None
                or receipt["status"] != "written"
                or receipt["followup_done"]
            ):
                continue
            current = await tx.query_raw(
                f"""SELECT m.*,(SELECT v.id FROM memory_temporal_versions v WHERE v.memory_id=m.id
                    AND v.side=$4 AND v.user_id=m.user_id AND v.workspace_id=m.workspace_id
                    ORDER BY v.id DESC LIMIT 1) AS revision
                    FROM {table} m WHERE m.id=$1 AND m.user_id=$2 AND m.workspace_id=$3 FOR SHARE OF m""",
                receipt["memory_id"],
                token.scope.owner_user_id,
                token.scope.workspace_id,
                token.side,
            )
            if (
                not current
                or current[0]["is_archived"]
                or current[0]["revision"] != receipt["memory_revision"]
                or sha256(current[0]["content"].encode()).hexdigest()
                != receipt["memory_version"]
            ):
                # An administrator changed/archived the result. Old effects
                # cannot be reapplied to that newer memory version.
                await tx.execute_raw(
                    "UPDATE memory_write_receipts SET followup_done=true,followup_outcome='superseded' WHERE batch_id=$1 AND fact_index=$2 AND piece_index=$3",
                    token.batch_id,
                    token.fact_index,
                    piece.piece_index,
                )
                continue
            row = current[0]
            mid = receipt["memory_id"]
            if (
                token.side == "user"
                and row["sub_category"] == "提醒"
                and row["occur_time"] is not None
                and row["recurrence"]
            ):
                if (row["occur_time"], row["recurrence"]) != (
                    prepared["occur_time"],
                    prepared["recurrence"],
                ):
                    raise receipts.MemoryReceiptError("memory_receipt_conflict")
                # Include inactive/fired rows: a recovery is not a new request
                # to schedule a reminder that already ran or was cancelled.
                existing = await tx.query_raw(
                    "SELECT id FROM time_triggers WHERE ai_agent_id=$1 AND user_id=$2 AND action_type='reminder' AND action_data->>'memory_id'=$3 LIMIT 1",
                    token.scope.agent_id,
                    token.scope.owner_user_id,
                    mid,
                )
                if not existing:
                    await tx.timetrigger.create(
                        data={
                            "id": str(
                                uuid5(
                                    NAMESPACE_URL,
                                    f"memory-receipt:{token.batch_id}:{token.fact_index}:{piece.piece_index}",
                                )
                            ),
                            "aiAgentId": token.scope.agent_id,
                            "userId": token.scope.owner_user_id,
                            "triggerTime": trigger_time,
                            "actionType": "reminder",
                            "actionData": Json(
                                {
                                    "memory_id": mid,
                                    "summary": row["content"][:200],
                                    "recurrence": row["recurrence"],
                                    "memory_side": token.side,
                                }
                            ),
                        }
                    )
            evidence = await tx.query_raw(
                """SELECT DISTINCT source_message_id FROM memory_evidence_links
                WHERE memory_id=$1 AND memory_source=$2 AND user_id=$3 AND workspace_id=$4
                  AND content_version=$5 AND source_message_id IS NOT NULL
                  AND temporal_basis->>'source_role'='fact' ORDER BY source_message_id LIMIT 101""",
                mid,
                token.side,
                token.scope.owner_user_id,
                token.scope.workspace_id,
                receipt["memory_version"],
            )
            if len(evidence) > 100:
                raise receipts.MemoryReceiptError("memory_plan_invalid")
            if evidence:
                await tx.memorychangelog.create(
                    data={
                        "userId": token.scope.owner_user_id,
                        "workspaceId": token.scope.workspace_id,
                        "memoryId": mid,
                        "operation": "evidence_linked",
                        "newValue": json.dumps(
                            {"message_ids": [e["source_message_id"] for e in evidence]}
                        ),
                    }
                )
            for operation, note in [
                (
                    ("user_emphasized", "用户表达了该信息的重要性")
                    if user_emphasized
                    else (None, None)
                ),
                (
                    ("reminder_past_time_demoted", reminder_demoted_reason)
                    if reminder_demoted_reason
                    else (None, None)
                ),
            ]:
                if operation:
                    await tx.memorychangelog.create(
                        data={
                            "userId": token.scope.owner_user_id,
                            "workspaceId": token.scope.workspace_id,
                            "memoryId": mid,
                            "operation": operation,
                            "newValue": note,
                        }
                    )
            common = dict(
                memory_id=mid,
                memory_source=token.side,
                user_id=token.scope.owner_user_id,
                workspace_id=token.scope.workspace_id,
                database=tx,
            )
            await entity_repo.record_entities_for_memory(
                **common,
                entities=[
                    {"name": name, "type": entity_types.get(name, "other")}
                    for name in mem.get("entities", [])
                    if isinstance(name, str) and name
                ],
            )
            await entity_repo.record_topics_for_memory(
                **common,
                topics=[t for t in mem.get("topics", []) if isinstance(t, str)],
            )
            prefs = mem.get("preferences")
            if not isinstance(prefs, list):
                prefs = [
                    p
                    for p in batch_preferences
                    if p.get("value") and str(p["value"]) in row["content"]
                ]
            await entity_repo.record_preferences_for_memory(
                **common, preferences=[p for p in prefs if isinstance(p, dict)]
            )
            await tx.execute_raw(
                "UPDATE memory_write_receipts SET followup_done=true,followup_outcome='applied' WHERE batch_id=$1 AND fact_index=$2 AND piece_index=$3",
                token.batch_id,
                token.fact_index,
                piece.piece_index,
            )
