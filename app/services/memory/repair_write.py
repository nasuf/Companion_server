"""Atomic administrator repair commands, separate from model reconciliation.

An explicit replacement must never be silently merged into an unrelated row.
Only the selected records change, together with their audit, origins and queue
completion. Embeddings are prepared outside the transaction.
"""
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

from app.db import db
from app.services.memory.evidence import EvidenceSource, bind_memory_evidence, content_version
from app.services.memory.input_evidence import prepare_input_origin, persist_input_origin
from app.services.memory.lifecycle.lazy_update import scope_lock_key
from app.services.memory.manual_write import (
    MemoryWriteError, memory_scope, lock_actor, lock_repair, lock_memories, resolve_repair,
)
from app.services.memory.provenance import KNOWLEDGE_SEED
from app.services.memory.retrieval.knowledge_hits import invalidate_knowledge_rows
from app.services.memory.storage import repo
from app.services.memory.storage.embedding import generate_embedding, store_embedding
from app.services.memory.storage.persistence import after_memory_changelog, normalize_memory_type
from app.services.memory.taxonomy import resolve_taxonomy, is_singleton
from app.services.schedule_domain.time_parser import resolve_occur_time


OPERATIONS = {
    'archive_memory': 'repair_archive', 'downgrade_memory': 'repair_downgrade',
    'mark_verified': 'admin_verified', 'merge_memories': 'repair_merge',
    'insert_replacement_memory': 'repair_insert_replacement',
}


def snapshot(record):
    data = asdict(record)
    return {key: value.isoformat() if isinstance(value, datetime) else value for key, value in data.items()}


def check_scope(item, record):
    for key, actual in (('user_id',record.userId),('workspace_id',record.workspaceId),('memory_source',record.source)):
        if item.get(key) is not None and item[key] != actual:
            raise MemoryWriteError(403, f'memory_{key}_does_not_match_repair_item')


async def apply_atomic_repair(*, repair_item, payload, admin_id, record):
    action = payload.action
    replacement = action == 'insert_replacement_memory'
    content = (payload.content or '').strip()
    if not admin_id:
        raise MemoryWriteError(403, 'memory_edit_actor_invalid')
    subject = record or SimpleNamespace(id=None, source=payload.source or repair_item.get('memory_source') or 'user',
        userId=payload.user_id or repair_item.get('user_id'), workspaceId=payload.workspace_id or repair_item.get('workspace_id'))
    if subject.source not in {'user','ai'} or not subject.userId or not subject.workspaceId:
        raise MemoryWriteError(400, 'repair_requires_explicit_user_workspace_source')
    check_scope(repair_item, subject)
    for supplied, expected in ((payload.source,subject.source),(payload.user_id,subject.userId),(payload.workspace_id,subject.workspaceId)):
        if supplied is not None and supplied != expected:
            raise MemoryWriteError(403, 'replacement_scope_must_match_target')
    records = [record] if record else []
    merged = []
    if action == 'merge_memories':
        ids = list(dict.fromkeys(mid for mid in payload.memory_ids if mid and mid != record.id))
        if not ids or len(payload.memory_ids) > 20:
            raise MemoryWriteError(400, 'merge_requires_1_to_20_memories')
        for mid in sorted(ids):
            other = await repo.find_unique(mid)
            if not other or other.isArchived:
                raise MemoryWriteError(404, 'merge_memory_not_found')
            if (other.source,other.userId,other.workspaceId) != (record.source,record.userId,record.workspaceId):
                raise MemoryWriteError(403, 'merge_memories_must_share_source_user_workspace')
            merged.append(other)
        records += merged
    if (replacement or merged) and any(row.subCategory == '提醒' for row in records):
        raise MemoryWriteError(400, 'repair_reminder_requires_schedule_workflow')
    scope = await memory_scope(db, subject)
    if repair_item.get('agent_id') is not None and repair_item['agent_id'] != scope['agent_id']:
        raise MemoryWriteError(403, 'memory_agent_does_not_match_repair_item')
    level = payload.level if payload.level is not None else (record.level if record else 2)
    importance = payload.importance if payload.importance is not None else (record.importance if record else .7)
    taxonomy = None
    if replacement:
        taxonomy = resolve_taxonomy(main_category=payload.main_category or (record.mainCategory if record else None),
            sub_category=payload.sub_category or (record.subCategory if record else None),
            legacy_type=normalize_memory_type(payload.memory_type or (record.type if record else None)),
            source=subject.source,level=level,provenance=record.provenance if record else None)
        if not taxonomy.allowed:
            raise MemoryWriteError(400, 'replacement_taxonomy_not_allowed')
        if taxonomy.sub_category == '提醒':
            raise MemoryWriteError(400, 'repair_reminder_requires_schedule_workflow')
    if action == 'downgrade_memory':
        level = payload.level if payload.level is not None else 3
        if level not in {2,3} or level < record.level:
            raise MemoryWriteError(400, 'downgrade_level_must_not_increase')
    embedding = await generate_embedding(content) if replacement or action == 'merge_memories' else None
    origin = prepare_input_origin(kind='admin_repair',payload={
        'action':action,'actor_id':admin_id,'repair_item_id':repair_item['id'],
        'repair_item_version':repair_item['updated_at'],'command':asdict(payload),
        'before':[snapshot(row) for row in records],
    })
    events = []
    result = {'action':action,'memory_source':subject.source,'user_id':subject.userId,'workspace_id':subject.workspaceId}
    async with db.tx(timeout=timedelta(seconds=30)) as tx:
        await tx.execute_raw('SELECT pg_advisory_xact_lock($1::bigint)',scope_lock_key(subject.source,subject.userId,subject.workspaceId))
        if await memory_scope(tx,subject,lock=True) != scope:
            raise MemoryWriteError(409,'memory_scope_changed_or_inactive')
        await lock_actor(tx,admin_id,'admin')
        await lock_repair(tx,repair_item,subject,scope)
        await lock_memories(tx,records,allow_archived=replacement)
        source = await persist_input_origin(tx,user_id=subject.userId,workspace_id=subject.workspaceId,agent_id=scope['agent_id'],origin=origin)
        target = record
        table = 'memories_ai' if subject.source == 'ai' else 'memories_user'
        if replacement:
            # Serialize with lifecycle and normal singleton inserts using their
            # existing advisory lock. Ignore only the explicitly replaced row.
            if level == 1 and is_singleton(taxonomy.main_category,taxonomy.sub_category):
                taken = await tx.query_raw(f'''SELECT id FROM {table} WHERE user_id=$1 AND workspace_id=$2
                    AND level=1 AND NOT is_archived AND main_category=$3 AND sub_category=$4
                    AND id IS DISTINCT FROM $5::text LIMIT 1''',subject.userId,subject.workspaceId,
                    taxonomy.main_category,taxonomy.sub_category,record.id if record else None)
                if taken:
                    raise MemoryWriteError(409,'replacement_singleton_already_exists')
            duplicate = await tx.query_raw(f'''SELECT id FROM {table} WHERE user_id=$1 AND workspace_id=$2
                AND NOT is_archived AND content=$3 AND id IS DISTINCT FROM $4::text LIMIT 1''',
                subject.userId,subject.workspaceId,content,record.id if record else None)
            if duplicate:
                raise MemoryWriteError(409,'replacement_content_already_exists')
            now = datetime.now(timezone.utc)
            target = await repo.create(source=subject.source,database=tx,userId=subject.userId,workspaceId=subject.workspaceId,
                content=content,level=level,importance=importance,type=normalize_memory_type(taxonomy.legacy_type),
                mainCategory=taxonomy.main_category,subCategory=taxonomy.sub_category,
                provenance=record.provenance if record else None,valueUpdatedAt=now,statementTime=now,
                occurTime=resolve_occur_time(content,statement_time=now,main_category=taxonomy.main_category,
                    sub_category=taxonomy.sub_category,provenance=record.provenance if record else None))
        elif action == 'merge_memories':
            await repo.update(record.id,source=record.source,record=record,database=tx,content=content,importance=importance)
        elif action == 'archive_memory':
            await repo.update(record.id,source=record.source,record=record,database=tx,isArchived=True)
        elif action == 'downgrade_memory':
            await repo.update(record.id,source=record.source,record=record,database=tx,level=level)
            result['level'] = level
        if embedding is not None:
            await store_embedding(target.id,embedding,database=tx)
        parents = records if replacement else merged
        sources = [source]+[EvidenceSource('memory',row.id,side=row.source,relation='derived_from',
            expected_version=content_version(row.content)) for row in parents]
        await bind_memory_evidence(database=tx,memory_id=target.id,side=subject.source,user_id=subject.userId,
            workspace_id=subject.workspaceId,sources=sources,extractor_version='admin-repair-v1')
        result['memory_id'] = target.id
        if merged:
            result['absorbed_memory_ids'] = [row.id for row in merged]
        if replacement and record:
            result['replaced_memory_id'] = record.id
        for row in parents:
            await repo.update(row.id,source=row.source,record=row,database=tx,isArchived=True)
            op = 'repair_merge_archived' if merged else 'repair_archive'
            await audit(tx,row,op,repair_item,admin_id,payload,{'merged_into':target.id,'superseded_by_memory_id':target.id})
            events.append((row.id,op))
        await audit(tx,target,OPERATIONS[action],repair_item,admin_id,payload,{
            'content':content if embedding is not None else target.content,
            'verified':action=='mark_verified','result':result})
        events.append((target.id,OPERATIONS[action]))
        result['repair_item'] = await resolve_repair(tx,repair_item,admin_id,
            action+(': '+payload.reason.strip() if payload.reason and payload.reason.strip() else ''),dict(result))
    await repo.invalidate_scope(subject.userId,subject.workspaceId)
    if any(row.source == 'ai' and row.provenance == KNOWLEDGE_SEED for row in [*records,target]):
        await invalidate_knowledge_rows(subject.workspaceId)
    for mid,op in events:
        await after_memory_changelog(user_id=subject.userId,memory_id=mid,operation=op,workspace_id=subject.workspaceId)
    return result


async def audit(tx, record, operation, item, admin_id, payload, after):
    await tx.memorychangelog.create(data={'userId':record.userId,'workspaceId':record.workspaceId,
        'memoryId':record.id,'operation':operation,
        'oldValue':None if operation=='repair_insert_replacement' else record.content,
        'newValue':json.dumps({'repair_item_id':item['id'],'admin_id':admin_id,'action':payload.action,
                             'reason':payload.reason,'after':after},ensure_ascii=False)})
