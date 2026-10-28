"""Admin-only dialogue policy with optimistic concurrency and transactional audit."""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from app.api.jwt_auth import require_admin_jwt
from app.api.admin.proactive_tuning import _utc_iso
from app.db import db
from app.services.chat import context_policy as policy
from app.services.runtime_config import load_caches

router = APIRouter(prefix='/admin-api/chat-context-policy', tags=['admin'],
                   dependencies=[Depends(require_admin_jwt)])


class PolicyPatch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    values: dict[str, Any]
    expected_revision: str


async def _payload(raw):
    effective = policy.resolve(raw)
    rows = await db.query_raw('''SELECT c.id, c.created_at, c.changes, c.admin_id, u.username AS admin_name
        FROM chat_context_policy_changes c LEFT JOIN users u ON u.id=c.admin_id
        ORDER BY c.created_at DESC, c.id DESC LIMIT 30''')
    return {**effective.public(), 'stored': policy.stored(raw), 'params': list(policy.PARAMS),
            'exclusions': list(policy.EXCLUSIONS), 'refresh_seconds': 2,
            'token_estimate': {'kind': 'upper_bound_not_measured',
                               'max_context_chars': effective.max_chars,
                               'line_chars': policy.LINE_CHARS,
                               'window_caps': policy.window_caps(),
                               'search_main_history_tokens': policy.SEARCH_MAIN_HISTORY_TOKENS,
                               'note': '实际 token 与对话长度及模型有关。请结合 Trace 和逐轮评测判断。'},
            'sampled_at': datetime.now(UTC).isoformat(),
            'changes_limit': 30,
            'changes': [{**row, 'created_at': _utc_iso(row['created_at']),
                         'changes': json.loads(row['changes']) if isinstance(row['changes'], str) else row['changes']}
                        for row in rows]}


@router.get('')
async def get_policy():
    rows = await db.query_raw('SELECT chat_context_policy FROM system_config WHERE id=1')
    if not rows:
        raise HTTPException(503, '配置尚未初始化')
    return await _payload(rows[0].get('chat_context_policy'))


@router.put('')
async def put_policy(payload: PolicyPatch, admin: dict = Depends(require_admin_jwt)):
    # Reject invalid requests before any writes (including seed writes).
    try:
        policy.apply_patch({}, payload.values)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    async with db.tx() as tx:
        rows = await tx.query_raw('SELECT chat_context_policy FROM system_config WHERE id=1 FOR UPDATE')
        if not rows:
            raise HTTPException(503, '配置尚未初始化')
        raw = policy.stored(rows[0].get('chat_context_policy'))
        if policy.revision(raw) != payload.expected_revision:
            raise HTTPException(409, '配置已被修改，请刷新后重试')
        updated = policy.apply_patch(raw, payload.values)
        changed = updated != raw
        if changed:
            await tx.execute_raw('UPDATE system_config SET chat_context_policy=$1::jsonb, '
                                 'updated_at=CURRENT_TIMESTAMP WHERE id=1', json.dumps(updated) if updated else None)
            before, after = policy.resolve(raw), policy.resolve(updated)
            # Resolved values are what turns used; stored values keep a reset
            # invalid value visible (resolved 4000 → 4000 would hide it).
            changes = [{'key': key, 'before': getattr(before, key), 'after': getattr(after, key),
                        'stored_before': raw.get(key), 'stored_after': updated.get(key),
                        'restored_default': payload.values[key] is None}
                       for key in ('rounds', 'main_history_tokens')
                       if key in payload.values and raw.get(key) != updated.get(key)]
            await tx.execute_raw('INSERT INTO chat_context_policy_changes (id,admin_id,changes) '
                                 'VALUES ($1,$2,$3::jsonb)', str(uuid.uuid4()),
                                 str(admin.get('sub') or '') or None, json.dumps(changes))
    if changed:
        await load_caches()
    return await _payload(updated)


CACHE_QUALIFIED_MODEL_PREFIXES = ('ark/',)


@router.get('/usage')
async def context_usage(days: int = Query(7, ge=1, le=30)):
    """Aggregate recorded foreground turns only, bounded by date and row count.

    Trace retention/disabled tracing produces an explicit unknown group. It is
    not assigned today's policy. Model buckets carry input/cached/output tokens.
    Costs are recorded token costs plus normalized search fees, without quota.
    """
    now = datetime.now(UTC)
    start = now - timedelta(days=days)
    rows = await db.query_raw('''WITH bounded AS (
          SELECT * FROM llm_usage WHERE scope='chat' AND created_at >= $1::timestamp
            AND created_at < $2::timestamp ORDER BY created_at DESC, id DESC LIMIT 10001
        )
        SELECT u.id, u.tokens_by_model, u.cost_cny, u.web_search_calls,
               t.extra_json->'metadata'->'chat_context_policy'->>'effective_rounds' AS rounds,
               t.extra_json->'metadata'->'chat_context_policy'->>'main_history_token_budget' AS main_tokens
        FROM bounded u LEFT JOIN trace_runs t ON t.run_id=u.trace_id''',
        start.replace(tzinfo=None).isoformat(), now.replace(tzinfo=None).isoformat())
    if len(rows) > 10000:
        raise HTTPException(422, '该时间窗超过 10000 个回合，请缩短时间窗；未返回截断统计')

    from app.services.effect.context_summary import summarize
    return summarize(rows, start, now, now)
