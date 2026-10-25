"""Persist receipts on one image; replay on another against the same synthetic DB.

External model/embedding outputs are deterministic. Persistence, evidence, receipt
claims and completed-turn guards are real. No fixture is removed between phases.
"""
import asyncio
import json
import os
from pathlib import Path
import sys
from unittest.mock import AsyncMock
from urllib.parse import urlsplit
from uuid import uuid4


async def main():
    assert os.environ.get('APP_ENV') == 'test'
    url = urlsplit(os.environ['DATABASE_URL'])
    assert url.hostname == 'g03-postgres' and url.path == '/rollback'
    assert sys.argv[1] in ('prepare', 'replay')
    from app.db import db
    from prisma import Json
    from app.services.memory.recording import pipeline
    from app.services.memory.storage import persistence, repo
    from app.services.memory.storage.reconciliation import ReconciliationDecision
    from app.services.runtime import turn_followups
    from app.services.achievements import mode, engine

    await db.connect()
    state_path = Path('/proof/receipts.json')
    try:
        if sys.argv[1] == 'prepare':
            assert not state_path.exists()
            user = await db.user.create(data={'username':'g03-synthetic-' + uuid4().hex})
            agent = await db.aiagent.create(data={'userId':user.id,'name':'G03 Synthetic'})
            workspace = await db.chatworkspace.create(data={'userId':user.id,'agentId':agent.id,'status':'active'})
            conv = await db.conversation.create(data={'userId':user.id,'agentId':agent.id,'workspaceId':workspace.id})
            msg = await db.message.create(data={'conversationId':conv.id,'role':'user','content':'用户参观了合成展览'})
            turn_id = uuid4().hex
            reply = await db.message.create(data={'conversationId':conv.id,'role':'assistant','content':'合成回复',
                                                 'metadata':Json({'achievement_turn_id':turn_id})})
            state = dict(user=user.id, workspace=workspace.id, conversation=conv.id, message=msg.id,
                         reply=reply.id, turn_id=turn_id, content=msg.content)
        else:
            state = json.loads(state_path.read_text())
        extract = AsyncMock(return_value={'memories':[{'content':'用户参观了合成展览','importance':0.6,
            'main_category':'生活','sub_category':'其他','source_indices':[1]}],
            'entities':[],'preferences':[],'topics':[]})
        pipeline.should_memorize = AsyncMock(return_value=True)
        pipeline.should_extract_memory = lambda _: True
        pipeline.extract_memories = extract
        persistence.generate_embedding = AsyncMock(return_value=[1.0]+[0.0]*1023)
        persistence.resolve_memory_write = AsyncMock(return_value=ReconciliationDecision(action='insert_new'))
        persistence.after_memory_changelog = AsyncMock()
        repo._invalidate_caches = AsyncMock()
        memory_ids = await pipeline.process_memory_pipeline(user_id=state['user'],workspace_id=state['workspace'],
             side='user',new_conversation='user: '+state['content'],evidence_message_ids=[state['message']],
             evidence_conversation_id=state['conversation'])
        assert len(memory_ids) == 1
        assert extract.await_count == (1 if sys.argv[1]=='prepare' else 0)
        assert await turn_followups.register_completed_turn(conversation_id=state['conversation'],
            message_id=state['reply'],turn_id=state['turn_id'],user_message_ids=[state['message']])
        # The replay must return before even consulting the feature/model layer.
        mode.achievement_evaluation_enabled = AsyncMock(return_value=False)
        engine.handle_achievement_event = AsyncMock(side_effect=AssertionError('unexpected business replay'))
        await turn_followups.run_turn_followup({'turn_id':state['turn_id'],'generation':0})
        assert mode.achievement_evaluation_enabled.await_count == (1 if sys.argv[1]=='prepare' else 0)
        assert engine.handle_achievement_event.await_count == 0
        counts = dict(
            memories=await db.usermemory.count(where={'userId':state['user']}),
            changelog=await db.memorychangelog.count(where={'userId':state['user']}),
            receipts=await db.query_raw('SELECT status,followup_done,followup_outcome FROM memory_write_receipts'),
            turns=await db.query_raw('SELECT status,outcome FROM turn_followup_events WHERE turn_id=$1',state['turn_id']))
        assert counts['memories']==1 and counts['receipts'][0]['followup_done']
        assert counts['turns'][0]['status']=='completed'
        if sys.argv[1]=='prepare':
            state['counts'] = counts
            state_path.write_text(json.dumps(state))
        else:
            assert counts == state['counts'], (counts,state['counts'])
        print(json.dumps(dict(status='passed',phase=sys.argv[1],counts=counts,
                              extraction_calls=extract.await_count,model_calls=0)))
    finally:
        await db.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
