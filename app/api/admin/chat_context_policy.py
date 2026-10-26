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
                               'note': '实际 token 与对话长度及模型有关；主回复历史预算不变。请结合 Trace 和逐轮评测判断。'},
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
            changes = [{'key': 'rounds', 'before': policy.resolve(raw).rounds,
                        'after': policy.resolve(updated).rounds,
                        'restored_default': payload.values.get('rounds') is None}]
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
               t.extra_json->'metadata'->'chat_context_policy'->>'effective_rounds' AS rounds
        FROM bounded u LEFT JOIN trace_runs t ON t.run_id=u.trace_id''',
        start.replace(tzinfo=None).isoformat(), now.replace(tzinfo=None).isoformat())
    if len(rows) > 10000:
        raise HTTPException(422, '该时间窗超过 10000 个回合，请缩短时间窗；未返回截断统计')
    groups = {}
    for row in rows:
        n = str(row.get('rounds') or '')
        n = int(n) if n.isdigit() and 1 <= int(n) <= 20 else None
        buckets = policy.stored(row.get('tokens_by_model'))
        for model, tokens in buckets.items():
            if not isinstance(tokens, dict):
                continue
            key = (n, model)
            group = groups.setdefault(key, {'rounds': n, 'model': model, 'turns': 0,
                'input_tokens': 0, 'cached_input_tokens': 0, 'output_tokens': 0,
                'cache_known_input_tokens': 0, 'cache_unknown_turns': 0})
            group['turns'] += 1
            group['input_tokens'] += int(tokens.get('input') or 0)
            group['output_tokens'] += int(tokens.get('output') or 0)
            cached = tokens.get('cached_input')
            # The usage ledger writes 0 for every provider. Only Ark's 0 is a
            # measured miss (raw responses, 2026-10-26); elsewhere it is unknown.
            qualified = model.startswith(CACHE_QUALIFIED_MODEL_PREFIXES)
            if qualified and type(cached) is int and 0 <= cached <= int(tokens.get('input') or 0):
                group['cached_input_tokens'] += cached
                group['cache_known_input_tokens'] += int(tokens.get('input') or 0)
            else:
                group['cache_unknown_turns'] += 1
    # Persisted turn cost is a sum across models. Group it by N separately rather
    # than attributing the whole turn's cost to each model (double counting).
    costs = {}
    for row in rows:
        n = str(row.get('rounds') or '')
        n = int(n) if n.isdigit() and 1 <= int(n) <= 20 else None
        item = costs.setdefault(n, {'rounds': n, 'turns': 0, 'recorded_token_cost_cny': 0,
                                     'web_search_calls': 0})
        item['turns'] += 1
        item['recorded_token_cost_cny'] += float(row.get('cost_cny') or 0)
        item['web_search_calls'] += int(row.get('web_search_calls') or 0)
    for g in groups.values():
        g['cache_hit_rate'] = (g['cached_input_tokens']/g['cache_known_input_tokens']
                               if g['cache_known_input_tokens'] else None)
    return {'sampled_at': now.isoformat(), 'window_start': start.isoformat(), 'window_end': now.isoformat(),
            'scope': 'chat foreground usage rows; excludes background and proactive',
            'rows': len(rows), 'max_rows': 10000, 'groups': list(groups.values()),
            'costs': [{**item, 'normalized_search_cost_cny': item['web_search_calls'] * .004}
                      for item in costs.values()],
            'status': 'measured' if rows else 'empty',
            'notes': ['轮数未知：历史未采集、Trace关闭或已过保留期；不回填当前值。',
                      '缓存只对已核实"0=未命中"的供应商（方舟）计入；其他供应商记录的0视为未知，不当作未命中。',
                      '费用按回合已记录token费用；搜索按0.004元/次标准化，非免费额度账单。',
                      '该轮数仅控制模板上下文，主回复继续4000token历史预算。']}
