"""Admin-only reply repetition policy with optimistic concurrency and transactional audit."""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from app.api.jwt_auth import require_admin_jwt
from app.api.admin.proactive_tuning import _utc_iso
from app.db import db
from app.services.chat.reply_repetition import breaker, policy
from app.services.runtime_config import load_caches

router = APIRouter(prefix='/admin-api/reply-repetition-policy', tags=['admin'],
                   dependencies=[Depends(require_admin_jwt)])
CHANGES_LIMIT = 30


class PolicyPatch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    values: dict[str, Any]
    expected_revision: str


async def _payload(raw):
    effective = policy.resolve(raw)
    rows = await db.query_raw('''SELECT c.id, c.created_at, c.changes, c.admin_id, u.username AS admin_name
        FROM reply_repetition_policy_changes c LEFT JOIN users u ON u.id=c.admin_id
        ORDER BY c.created_at DESC, c.id DESC LIMIT $1''', CHANGES_LIMIT)
    return {**effective.public(), 'stored': policy.stored(raw), 'params': list(policy.PARAMS),
            'refresh_seconds': 2,
            # Live counts from Redis; "unknown" when they can't be read, never a fake zero.
            'breaker': await breaker.state(effective),
            'prompt_keys': ['chat.repetition_guard_section', 'chat.previous_reply_ack_section',
                            'chat.previous_reply_overlap_section', 'chat.previous_reply_other_section',
                            'chat.previous_reply_retry'],
            'sampled_at': datetime.now(UTC).isoformat(),
            'changes_limit': CHANGES_LIMIT,
            'changes': [{**row, 'created_at': _utc_iso(row['created_at']),
                         'changes': json.loads(row['changes']) if isinstance(row['changes'], str) else row['changes']}
                        for row in rows]}


@router.get('')
async def get_policy():
    rows = await db.query_raw('SELECT reply_repetition_policy FROM system_config WHERE id=1')
    if not rows:
        raise HTTPException(503, '配置尚未初始化')
    return await _payload(rows[0].get('reply_repetition_policy'))


@router.put('')
async def put_policy(payload: PolicyPatch, admin: dict = Depends(require_admin_jwt)):
    # Reject invalid requests before any writes.
    try:
        policy.apply_patch({}, payload.values)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    async with db.tx() as tx:
        rows = await tx.query_raw('SELECT reply_repetition_policy FROM system_config WHERE id=1 FOR UPDATE')
        if not rows:
            raise HTTPException(503, '配置尚未初始化')
        raw = policy.stored(rows[0].get('reply_repetition_policy'))
        if policy.revision(raw) != payload.expected_revision:
            raise HTTPException(409, '配置已被修改，请刷新后重试')
        updated = policy.apply_patch(raw, payload.values)
        changed = updated != raw
        if changed:
            await tx.execute_raw('UPDATE system_config SET reply_repetition_policy=$1::jsonb, '
                                 'updated_at=CURRENT_TIMESTAMP WHERE id=1', json.dumps(updated) if updated else None)
            before, after = policy.resolve(raw), policy.resolve(updated)
            changes = [{'key': key, 'before': getattr(before, key), 'after': getattr(after, key),
                        'stored_before': raw.get(key), 'stored_after': updated.get(key),
                        'restored_default': payload.values[key] is None}
                       for key in payload.values if raw.get(key) != updated.get(key)]
            await tx.execute_raw('INSERT INTO reply_repetition_policy_changes (id,admin_id,changes) '
                                 'VALUES ($1,$2,$3::jsonb)', str(uuid.uuid4()),
                                 str(admin.get('sub') or '') or None, json.dumps(changes))
    if changed:
        await load_caches()
    return await _payload(updated)
