"""Account erasure: explicit ownership, fail-closed residual checks and Redis isolation.

Never searches arbitrary message text for an ID: another user may mention it.
Shared configuration, public place data, backups and provider-side data are not
account rows. Their retention boundary is documented in docs/user-erasure-audit.md.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import timedelta
import hashlib
import json
import re
from typing import Any

# Transaction-local GUC. hard_delete_user_data sets it so foreign-key
# ON DELETE SET NULL can clear resource pointers while the owner is already
# 'deleting'. Other sessions never see it, so late writes stay rejected.
# The SQL guard reads this exact name.
ERASURE_USER_SETTING = "app.account_erasure_user"

_SCOPE_COLUMNS = (
    "user_id",
    "owner_user_id",
    "actor_user_id",
    "reporter_id",
    "resolved_by_id",
    "owner_admin_id",
    "admin_id",
    "source_user_id",
    "created_by",
)
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def _quote(identifier: str) -> str:
    if not _IDENTIFIER.fullmatch(identifier):
        raise RuntimeError("Unrecognized erasure schema identifier")
    return '"' + identifier + '"'


@asynccontextmanager
async def owner_write_scope(uid: str, *, database=None, update_owner: bool = False):
    """Short fence for local media/Redis writes; never wrap provider/LLM I/O."""
    from app.db import db
    from fastapi import HTTPException

    database = database if database is not None else db
    async with database.tx(timeout=timedelta(seconds=10)) as tx:
        lock = "UPDATE" if update_owner else "SHARE"
        rows = await tx.query_raw(
            f"SELECT id FROM users WHERE id=$1 AND status<>'deleting' FOR {lock}", uid
        )
        if not rows:
            raise HTTPException(status_code=401, detail="Account unavailable")
        yield tx


@asynccontextmanager
async def conversation_write_scope(conversation_id: str, *, database=None):
    """Fence a short private Redis write against deleting its conversation owner."""
    from app.db import db
    from fastapi import HTTPException

    database = database if database is not None else db
    async with database.tx(timeout=timedelta(seconds=10)) as tx:
        rows = await tx.query_raw(
            """SELECT c.id FROM conversations c JOIN users u ON u.id=c.user_id
            WHERE c.id=$1 AND u.status<>'deleting' FOR SHARE OF c,u""",
            conversation_id,
        )
        if not rows:
            raise HTTPException(status_code=401, detail="Conversation unavailable")
        yield tx


@asynccontextmanager
async def runtime_job_write_scope(payload: dict, *, database=None):
    """Stop a late producer from reintroducing a deleted account's queue payload."""
    from app.db import db
    from fastapi import HTTPException

    database = database if database is not None else db
    uid = payload.get("user_id")
    if isinstance(uid, str) and uid:
        async with owner_write_scope(uid, database=database):
            yield
        return
    sources = (
        (
            "message_id",
            "SELECT c.user_id FROM messages m JOIN conversations c ON c.id=m.conversation_id WHERE m.id=$1",
        ),
        ("conversation_id", "SELECT user_id FROM conversations WHERE id=$1"),
        ("agent_id", "SELECT user_id FROM ai_agents WHERE id=$1"),
        (
            "turn_id",
            "SELECT scope->>'user_id' AS user_id FROM turn_followup_events WHERE turn_id=$1",
        ),
    )
    for key, query in sources:
        resource = payload.get(key)
        if isinstance(resource, str) and resource:
            rows = await database.query_raw(query, resource)
            if not rows or not rows[0]["user_id"]:
                raise HTTPException(status_code=401, detail="Job owner unavailable")
            async with owner_write_scope(rows[0]["user_id"], database=database):
                yield
            return
    # System jobs with no account-bound resource remain available.
    yield


async def begin_erasure(database: Any, uid: str) -> None:
    from datetime import timedelta
    from app.services.agent_template.registry import TEMPLATE_SYSTEM_USERNAME

    async with database.tx(timeout=timedelta(seconds=10)) as tx:
        # Serializes two admins deleting one another; an old request cannot
        # remove the last still-active administrator after its peer is disabled.
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock(hashtext('account-erasure-admin'))"
        )
        rows = await tx.query_raw(
            "SELECT username,role FROM users WHERE id=$1 FOR UPDATE", uid
        )
        if not rows or rows[0]["username"] == TEMPLATE_SYSTEM_USERNAME:
            raise RuntimeError("Protected or unavailable account")
        if rows[0]["role"] == "admin":
            peers = await tx.query_raw(
                "SELECT COUNT(*)::int AS n FROM users WHERE role='admin' AND status='active' AND archived_at IS NULL AND id<>$1",
                uid,
            )
            if peers[0]["n"] == 0:
                raise RuntimeError("Cannot delete the last active administrator")
        await tx.execute_raw(
            "UPDATE users SET status='deleting', execution_generation=gen_random_uuid()::text WHERE id=$1",
            uid,
        )


async def assert_isolated_owner(tx: Any, uid: str) -> None:
    # Malformed legacy links must never turn an account deletion into a cascade
    # through a different owner's conversations, workspaces or agent Runs.
    rows = await tx.query_raw(
        """SELECT EXISTS(
        SELECT 1 FROM conversations c JOIN ai_agents a ON a.id=c.agent_id
          WHERE (a.user_id=$1 AND c.user_id<>$1) OR (c.user_id=$1 AND a.user_id<>$1)
        UNION ALL
        SELECT 1 FROM chat_workspaces w JOIN ai_agents a ON a.id=w.agent_id
          WHERE (a.user_id=$1 AND w.user_id<>$1) OR (w.user_id=$1 AND a.user_id<>$1)
        UNION ALL
        SELECT 1 FROM agent_runs WHERE actor_user_id=$1 AND owner_user_id<>$1
    ) AS conflict""",
        uid,
    )
    if rows[0]["conflict"]:
        raise RuntimeError(
            "Cross-owner dependencies require repair before account deletion"
        )
    covers = await tx.query_raw(
        """WITH owned AS (
        SELECT image_url FROM chat_link_cover_objects WHERE left(storage_key,length($1)+1)=$1||'_'
    ) SELECT EXISTS(
        SELECT 1 FROM chat_link_cards l JOIN owned o ON o.image_url=l.image_url WHERE l.user_id<>$1
        UNION ALL SELECT 1 FROM chat_link_snapshots s JOIN chat_link_cards l ON l.id=s.link_card_id
          JOIN owned o ON o.image_url=s.payload->>'image_url' WHERE l.user_id<>$1
        UNION ALL SELECT 1 FROM messages m JOIN conversations c ON c.id=m.conversation_id
          JOIN owned o ON o.image_url=m.metadata#>>'{link_card,image_url}'
            OR o.image_url=m.metadata#>>'{component_card,payload,image_url}' WHERE c.user_id<>$1
    ) AS conflict""",
        uid,
    )
    if covers[0]["conflict"]:
        raise RuntimeError("Cross-owner cover requires repair before deletion")


async def _delete_admin_subject_alerts(tx: Any, uid: str, scope: dict) -> None:
    """Drop operator alerts whose only subject is the account being erased.

    Takeover and new-user notifications are stored on the admin, with the
    subject's conversation, message, or user id attached. The preview is a copy
    of that account. Unrelated admin alerts, and any non-admin row that points
    at the same resource, stay in place so a real cross-owner link still fails.
    """
    await tx.execute_raw(
        """
        DELETE FROM notification_events AS alert
        USING users AS admin
        WHERE alert.user_id = admin.id
          AND admin.role = 'admin'
          AND alert.user_id <> $1
          AND (
            alert.conversation_id = ANY($2::text[])
            OR alert.message_id = ANY($3::text[])
            OR alert.agent_id = ANY($4::text[])
            OR alert.workspace_id = ANY($5::text[])
            OR alert.payload->>'user_id' = $1
            OR alert.payload->>'conversation_id' = ANY($2::text[])
            OR alert.payload->>'message_id' = ANY($3::text[])
            OR alert.payload->>'agent_id' = ANY($4::text[])
            OR alert.payload->>'workspace_id' = ANY($5::text[])
          )
        """,
        uid,
        sorted(scope.get("conversation_id") or ()),
        sorted(scope.get("message_id") or ()),
        sorted(scope.get("agent_id") or ()),
        sorted(scope.get("workspace_id") or ()),
    )


async def capture_erasure_scope(tx: Any, uid: str) -> dict[str, Any]:
    scope: dict[str, Any] = {"user_id": {uid}}
    for name, table in [
        ("agent_id", "ai_agents"),
        ("workspace_id", "chat_workspaces"),
        ("conversation_id", "conversations"),
        ("memory_id", "memories_user"),
        ("ai_memory_id", "memories_ai"),
        ("entity_id", "memory_entities"),
    ]:
        rows = await tx.query_raw(
            f"SELECT id FROM {_quote(table)} WHERE user_id=$1", uid
        )
        scope[name] = {row["id"] for row in rows}
    rows = await tx.query_raw(
        """SELECT m.id FROM messages m JOIN conversations c
        ON c.id=m.conversation_id WHERE c.user_id=$1""",
        uid,
    )
    scope["message_id"] = {row["id"] for row in rows}
    scope["turn_id"] = set(scope["message_id"])
    # Root bindings survive even when the message or message_traces mirror is absent.
    traces = await tx.query_raw(
        """SELECT DISTINCT trace_id FROM trace_runs
        WHERE inputs_json->>'conversation_id'=ANY($2::text[])
          OR inputs_json->>'user_id'=$1
          OR extra_json->'metadata'->>'user_id'=$1
          OR extra_json->'metadata'->>'owner_user_id'=$1
        UNION SELECT trace_id FROM message_traces WHERE conversation_id=ANY($2::text[])
        UNION SELECT trace_id FROM llm_usage WHERE
          (user_id=$1 OR conversation_id=ANY($2::text[]) OR agent_id=ANY($3::text[])) AND trace_id IS NOT NULL
        UNION SELECT metadata->>'trace_id' FROM messages
          WHERE conversation_id=ANY($2::text[]) AND metadata->>'trace_id' IS NOT NULL""",
        uid,
        sorted(scope["conversation_id"]),
        sorted(scope["agent_id"]),
    )
    scope["trace_id"] = {row["trace_id"] for row in traces}
    shared_traces = await tx.query_raw(
        """SELECT EXISTS(
        SELECT 1 FROM llm_usage WHERE user_id<>$1 AND trace_id=ANY($2::text[])
        UNION ALL SELECT 1 FROM llm_usage l JOIN conversations c ON c.id=l.conversation_id
          WHERE c.user_id<>$1 AND l.trace_id=ANY($2::text[])
        UNION ALL SELECT 1 FROM llm_usage l JOIN ai_agents a ON a.id=l.agent_id
          WHERE a.user_id<>$1 AND l.trace_id=ANY($2::text[])
        UNION ALL SELECT 1 FROM message_traces m JOIN conversations c ON c.id=m.conversation_id
          WHERE c.user_id<>$1 AND m.trace_id=ANY($2::text[])
        UNION ALL SELECT 1 FROM trace_runs r JOIN conversations c ON c.id=r.inputs_json->>'conversation_id'
          WHERE c.user_id<>$1 AND r.trace_id=ANY($2::text[])
        UNION ALL SELECT 1 FROM trace_runs WHERE trace_id=ANY($2::text[])
          AND (extra_json->'metadata'->>'owner_user_id'<>$1
            OR extra_json->'metadata'->>'user_id'<>$1 OR inputs_json->>'user_id'<>$1)
    ) AS conflict""",
        uid,
        sorted(scope["trace_id"]),
    )
    if shared_traces[0]["conflict"]:
        raise RuntimeError("Cross-owner Trace requires repair before deletion")
    jobs = await tx.query_raw(
        """SELECT job_id FROM message_followup_events
        WHERE scope->>'user_id'=$1 AND job_id IS NOT NULL
        UNION SELECT job_id FROM turn_followup_events
        WHERE scope->>'user_id'=$1 AND job_id IS NOT NULL
        UNION SELECT job_id FROM location_memory_events
        WHERE scope->>'user_id'=$1 AND job_id IS NOT NULL""",
        uid,
    )
    scope["job_id"] = {row["job_id"] for row in jobs}
    users = await tx.query_raw("SELECT username FROM users WHERE id=$1", uid)
    scope["username"] = {row["username"] for row in users}
    identities = await tx.query_raw(
        "SELECT provider,provider_account_id,openid,unionid FROM auth_identities WHERE user_id=$1",
        uid,
    )
    scope["auth_identity"] = {
        value
        for row in identities
        for field in ("provider_account_id", "openid", "unionid")
        if (value := row.get(field))
    }
    identity_conflict = await tx.query_raw(
        """SELECT EXISTS(SELECT 1 FROM auth_identities
        WHERE user_id<>$1 AND (provider_account_id=ANY($2::text[]) OR openid=ANY($2::text[])
                              OR unionid=ANY($2::text[]))) AS conflict""",
        uid,
        sorted(scope["auth_identity"]),
    )
    if identity_conflict[0]["conflict"]:
        raise RuntimeError("Cross-owner identity requires repair before deletion")
    scope["phone"] = {
        row["provider_account_id"] for row in identities if row["provider"] == "phone"
    }
    scope["openid"] = {row["openid"] for row in identities if row.get("openid")}
    scope["unionid"] = {row["unionid"] for row in identities if row.get("unionid")}
    # Admin inbox copies name the subject but are addressed to an operator.
    # They are removed here, before the cross-owner check, because their
    # conversation/message columns would otherwise look like a peer's records.
    await _delete_admin_subject_alerts(tx, uid, scope)
    # A bad legacy owner edge must be repaired rather than cascade into a peer.
    columns = await tx.query_raw("""SELECT table_name,array_agg(column_name::text) AS columns
        FROM information_schema.columns WHERE table_schema='public'
        GROUP BY table_name HAVING bool_or(column_name='user_id')""")
    bindings = {
        key: sorted(scope[key])
        for key in ("agent_id", "workspace_id", "conversation_id", "entity_id")
    }
    for row in columns:
        for column, ids in bindings.items():
            if column not in row["columns"] or not ids:
                continue
            conflict = await tx.query_raw(
                f"SELECT EXISTS(SELECT 1 FROM {_quote(row['table_name'])} "
                f"WHERE user_id<>$1 AND {_quote(column)}=ANY($2::text[])) AS conflict",
                uid,
                ids,
            )
            if conflict[0]["conflict"]:
                raise RuntimeError(
                    "Cross-owner resource link requires repair before deletion: "
                    f"{row['table_name']}.{column}"
                )
    provenance = await tx.query_raw(
        "SELECT EXISTS(SELECT 1 FROM memory_evidence_links WHERE source_user_id=$1 AND user_id<>$1) AS conflict",
        uid,
    )
    if provenance[0]["conflict"]:
        raise RuntimeError("Cross-owner provenance requires repair before deletion")
    collision = await tx.query_raw(
        """SELECT EXISTS(SELECT 1 FROM memories_user u
        JOIN memories_ai a ON a.id=u.id WHERE
        (u.user_id=$1 AND a.user_id<>$1) OR (a.user_id=$1 AND u.user_id<>$1)) AS conflict""",
        uid,
    )
    if collision[0]["conflict"]:
        raise RuntimeError("Cross-owner memory identifier collision requires repair")
    return scope


async def delete_unlinked_rows(tx: Any, uid: str, scope: dict) -> None:
    # Legacy diagnostic rows can have NULL user_id but a definite private scope.
    # A fixed table allowlist avoids guessing ownership of shared configuration.
    columns = await tx.query_raw(
        """SELECT table_name,array_agg(column_name::text) AS columns
        FROM information_schema.columns WHERE table_schema='public' AND table_name=ANY($1::text[])
        GROUP BY table_name""",
        [
            "llm_usage",
            "crisis_events",
            "memory_visible_use_events",
            "memory_repair_items",
            "memory_consolidation_runs",
        ],
    )
    for row in columns:
        clauses, params = [], []
        for column in ("agent_id", "workspace_id", "conversation_id"):
            if column in row["columns"] and scope[column]:
                params.append(sorted(scope[column]))
                clauses.append(f"{_quote(column)}=ANY(${len(params)}::text[])")
        if clauses:
            await tx.execute_raw(
                f"DELETE FROM {_quote(row['table_name'])} WHERE user_id IS NULL AND ("
                + " OR ".join(clauses)
                + ")",
                *params,
            )
    # Preserve another owner's purchase chain. This is a repair condition, never
    # a reason to remove their notifications or history.
    shared = await tx.query_raw(
        """WITH chains AS (
        SELECT user_id,provider,environment,original_transaction_id FROM iap_transactions
        UNION SELECT user_id,provider,environment,original_transaction_id FROM iap_subscription_state
    ) SELECT EXISTS(SELECT 1 FROM chains a
        JOIN chains b ON b.provider=a.provider AND b.environment=a.environment
          AND b.original_transaction_id=a.original_transaction_id
        WHERE a.user_id=$1 AND b.user_id<>$1) AS conflict""",
        uid,
    )
    if shared[0]["conflict"]:
        raise RuntimeError("Shared purchase chain requires repair before deletion")
    await tx.execute_raw(
        """DELETE FROM iap_notifications n WHERE EXISTS (
        SELECT 1 FROM iap_transactions t WHERE t.user_id=$1 AND t.provider=n.provider
          AND t.environment=n.environment
          AND (t.transaction_id=n.transaction_id OR t.original_transaction_id=n.original_transaction_id)
    ) OR EXISTS (SELECT 1 FROM iap_subscription_state t WHERE t.user_id=$1 AND t.provider=n.provider
          AND t.environment=n.environment AND t.original_transaction_id=n.original_transaction_id)""",
        uid,
    )
    await tx.execute_raw(
        "DELETE FROM trace_runs WHERE trace_id=ANY($1::text[])",
        sorted(scope["trace_id"]),
    )
    # These have no user FK in the old schema. Keep the explicit list reviewed;
    # the catalog-based postcondition detects new tables rather than skipping them.
    for table in ("speech_usage", "tts_usage"):
        await tx.execute_raw(f"DELETE FROM {_quote(table)} WHERE user_id=$1", uid)
    await tx.execute_raw(
        "UPDATE memory_repair_items SET resolved_by_id=NULL WHERE resolved_by_id=$1",
        uid,
    )
    # Shared tuning/config history belongs to the system; redact its actor only.
    await tx.execute_raw(
        "UPDATE proactive_tuning_changes SET admin_id='[deleted]' WHERE admin_id=$1",
        uid,
    )
    await tx.execute_raw(
        "UPDATE vip_activation_codes SET created_by='[deleted]' WHERE created_by=$1",
        uid,
    )
    await tx.execute_raw(
        "UPDATE real_world_gifts SET product_snapshot=product_snapshot-'created_by_admin_id' WHERE product_snapshot->>'created_by_admin_id'=$1",
        uid,
    )


async def delete_cover_ledger(tx: Any, uid: str) -> int:
    """Files have been removed; release only the target's registered bytes."""
    rows = await tx.query_raw(
        """DELETE FROM chat_link_cover_objects
        WHERE left(storage_key,length($1)+1)=$1||'_' RETURNING byte_size""",
        uid,
    )
    if rows:
        size = sum(int(row["byte_size"]) for row in rows)
        changed = await tx.execute_raw(
            """UPDATE chat_link_storage_state
            SET reserved_bytes=reserved_bytes-$1::bigint WHERE id=1 AND reserved_bytes >= $1::bigint""",
            size,
        )
        if changed != 1:
            raise RuntimeError("Cover storage ledger requires repair before deletion")
    return len(rows)


async def verify_erasure(tx: Any, uid: str, scope: dict) -> None:
    rows = await tx.query_raw(
        """SELECT table_name,column_name FROM information_schema.columns
        WHERE table_schema='public' AND column_name=ANY($1::text[])
        ORDER BY table_name,column_name""",
        list(_SCOPE_COLUMNS),
    )
    for row in rows:
        table, column = _quote(row["table_name"]), _quote(row["column_name"])
        found = await tx.query_raw(
            f"SELECT EXISTS(SELECT 1 FROM {table} WHERE {column}=$1) AS remains", uid
        )
        if found[0]["remains"]:
            raise RuntimeError(
                f"Erasure residual in {row['table_name']}.{row['column_name']}"
            )
    traces = await tx.query_raw(
        "SELECT EXISTS(SELECT 1 FROM trace_runs WHERE trace_id=ANY($1::text[])) AS remains",
        sorted(scope["trace_id"]),
    )
    if traces[0]["remains"]:
        raise RuntimeError("Erasure residual in trace_runs")
    resources = {
        "agent_id": scope["agent_id"],
        "ai_agent_id": scope["agent_id"],
        "workspace_id": scope["workspace_id"],
        "conversation_id": scope["conversation_id"],
        "memory_id": scope["memory_id"] | scope["ai_memory_id"],
        "source_memory_id": scope["memory_id"] | scope["ai_memory_id"],
        "entity_id": scope["entity_id"],
        "message_id": scope["message_id"],
        "source_message_id": scope["message_id"],
        "trace_id": scope["trace_id"],
    }
    columns = await tx.query_raw(
        """SELECT table_name,column_name FROM information_schema.columns
        WHERE table_schema='public' AND column_name=ANY($1::text[])""",
        list(resources),
    )
    for row in columns:
        identifiers = resources[row["column_name"]]
        if not identifiers:
            continue
        table, column = _quote(row["table_name"]), _quote(row["column_name"])
        remaining = await tx.query_raw(
            f"SELECT EXISTS(SELECT 1 FROM {table} WHERE {column}::text=ANY($1::text[])) AS remains",
            sorted(identifiers),
        )
        if remaining[0]["remains"]:
            raise RuntimeError(
                f"Erasure resource residual in {row['table_name']}.{row['column_name']}"
            )


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _contains_scope(value: Any, scope: dict) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = {
                "sub": "user_id",
                "userId": "user_id",
                "owner_user_id": "user_id",
                "actor_user_id": "user_id",
                "agentId": "agent_id",
                "conversationId": "conversation_id",
                "workspaceId": "workspace_id",
                "id": "resource_id",
            }.get(key, key)
            allowed = scope.get(normalized, set())
            if normalized == "resource_id":
                allowed = set().union(
                    *(
                        scope.get(k, set())
                        for k in (
                            "memory_id",
                            "ai_memory_id",
                            "entity_id",
                            "message_id",
                        )
                    )
                )
            if isinstance(item, str) and item in allowed:
                return True
            if _contains_scope(item, scope):
                return True
    elif isinstance(value, list):
        return any(_contains_scope(item, scope) for item in value)
    return False


def _glob(value: str) -> str:
    return "".join("\\" + char if char in "*?[]\\" else char for char in value)


async def _keys(redis: Any, pattern: str):
    cursor, batches = 0, 0
    while True:
        cursor, keys = await redis.scan(cursor, match=pattern, count=100)
        yield keys
        batches += 1
        if int(cursor) == 0:
            return
        if batches >= 10000:
            raise RuntimeError("Redis erasure scan exceeded its bounded budget")


_JOB_ERASE = """
if redis.call('HGET',KEYS[1],'status')=='running' then return -1 end
redis.call('DEL',KEYS[1])
for i=2,4 do redis.call('LREM',KEYS[i],0,ARGV[1]) end
for i=5,6 do redis.call('ZREM',KEYS[i],ARGV[1]) end
return 1
"""


async def clear_private_redis(redis: Any, uid: str, scope: dict) -> int:
    deleted = 0
    ids = set().union(
        *(
            scope.get(k, set())
            for k in (
                "user_id",
                "agent_id",
                "workspace_id",
                "conversation_id",
                "trace_id",
                "auth_identity",
            )
        )
    )
    # One bounded pass instead of one full Redis scan per conversation.
    async for keys in _keys(redis, "*"):
        owned = [
            key for key in keys if ids.intersection(re.split(r"[:{}]", _text(key)))
        ]
        if owned:
            deleted += await redis.delete(*owned)
    for phone in scope.get("phone", set()):
        deleted += await redis.delete(
            *(f"sms:{kind}:{phone}" for kind in ("code", "cooldown", "daily", "tries"))
        )
    # Login limits are keyed by a hash of username, rather than account UUID.
    for username in scope.get("username", set()):
        digest = hashlib.sha256(username.strip().lower().encode()).hexdigest()[:16]
        async for keys in _keys(redis, f"auth:login_fail:{digest}:*"):
            if keys:
                deleted += await redis.delete(*keys)
    # Shared indexes must lose only this account's members, never the whole key.
    for aid in scope["agent_id"]:
        deleted += await redis.zrem("pending:delayed", f"{aid}:{uid}")
        deleted += await redis.zrem("turn:delayed", f"{aid}:{uid}")
    if scope["conversation_id"]:
        deleted += await redis.zrem("delayed:due", *sorted(scope["conversation_id"]))
    for pattern in ("runtime:job:*", "ws:ticket:*", "cache:ret:*", "cache:graph:*"):
        async for keys in _keys(redis, pattern):
            for raw_key in keys:
                key = _text(raw_key)
                if key.startswith("runtime:job:"):
                    row = {
                        _text(k): _text(v)
                        for k, v in (await redis.hgetall(key)).items()
                    }
                    job_id = key.removeprefix("runtime:job:")
                    payload = json.loads(row.get("payload", "{}"))
                    if job_id not in scope["job_id"] and not _contains_scope(
                        payload, scope
                    ):
                        continue
                    result = await redis.eval(
                        _JOB_ERASE,
                        6,
                        key,
                        "runtime:jobs:ready",
                        "runtime:jobs:dlq",
                        "runtime:jobs:succeeded",
                        "runtime:jobs:running",
                        "runtime:jobs:delayed",
                        job_id,
                    )
                    if result == -1:
                        raise RuntimeError(
                            "Account still has a running background job; retry after it stops"
                        )
                    scope["job_id"].add(job_id)
                    deleted += int(result)
                else:
                    value = await redis.get(key)
                    if value and _contains_scope(json.loads(value), scope):
                        deleted += await redis.delete(key)
    async for keys in _keys(redis, "runtime:job_idem:*"):
        for key in keys:
            value = await redis.get(key)
            if value and _text(value) in scope["job_id"]:
                deleted += await redis.delete(key)
    # Shared diagnostics/presence indexes contain IDs in members/values.
    cursor = 0
    while True:
        cursor, rows = await redis.hscan(
            "memory:processing:observations:data", cursor, count=100
        )
        for field, raw in rows.items():
            if _contains_scope(json.loads(raw), scope):
                deleted += await redis.hdel(
                    "memory:processing:observations:data", field
                )
                deleted += await redis.zrem(
                    "memory:processing:observations:index", field
                )
        if int(cursor) == 0:
            break
    deleted += await redis.zrem("presence:online:hb", uid)
    cursor = 0
    while True:
        cursor, members = await redis.zscan(
            "presence:online:ws", cursor, match=_glob(uid) + "|*", count=100
        )
        if members:
            deleted += await redis.zrem(
                "presence:online:ws", *(member for member, _ in members)
            )
        if int(cursor) == 0:
            break
    # Ticket limiter stores no personal contents but is still account-specific.
    deleted += await redis.delete(
        "ws:ticket-rate:" + hashlib.sha256(uid.encode()).hexdigest()
    )
    return deleted
