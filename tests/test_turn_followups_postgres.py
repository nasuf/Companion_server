"""Complete-turn qualification and real achievement replay on disposable SQL/Redis."""
import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from prisma import Json

from app.services.runtime import turn_followups as t, job_queue as q
from app.services.achievements import engine
from app.services.achievements.rules import assistant_message_rules
from tests.test_message_followups_postgres import followups, redis, origins


@pytest.fixture
async def turn(followups, monkeypatch):
    db, user, *rest = followups
    monkeypatch.setattr(t, 'db', db)
    monkeypatch.setattr(engine, '_evaluate_event', assistant_message_rules.evaluate_assistant_turn)
    user = await db.message.update(where={'id':user.id}, data={'content':'甲'*50})
    tid = uuid4().hex
    replies = [await db.message.create(data={'conversationId':user.conversationId,
        'role':'assistant','content':'乙'*25,'metadata':Json({'achievement_turn_id':tid})}) for _ in range(2)]
    q.register_job_handler(t.JOB_TYPE, t.run_turn_followup)
    return db, user, replies, {'conversation_id':user.conversationId,'message_id':replies[0].id,
        'turn_id':tid,'user_message_ids':[user.id]}


async def receipt(db, tid):
    rows = await db.query_raw('SELECT * FROM turn_followup_events WHERE turn_id=$1', tid)
    return rows[0] if rows else None


async def test_explicit_completion_replay_uses_full_persisted_turn(turn):
    db,user,replies,args = turn
    assert await receipt(db,args['turn_id']) is None
    assert await t.register_completed_turn(**args)
    assert await t.register_completed_turn(**args)
    assert await t.dispatch_turn_followups(turn_id=args['turn_id']) == 1
    await q.process_runtime_jobs(max_jobs=1)
    await t.run_turn_followup({'turn_id':args['turn_id'],'generation':0})
    assert (await receipt(db,args['turn_id']))['status']=='completed'
    assert await db.achievementunlock.count(where={'conversationId':user.conversationId,'achievementId':81})==1


@pytest.mark.parametrize('change', ['user_content','reply_content','reply_deleted','extra_reply','scope','human','root_deleted'])
async def test_changed_source_never_completes_old_turn(turn, change):
    db,user,replies,args = turn
    assert await t.register_completed_turn(**args)
    if change=='user_content':await db.message.update(where={'id':user.id},data={'content':'changed'})
    elif change=='reply_content':await db.message.update(where={'id':replies[1].id},data={'content':'changed'})
    elif change=='reply_deleted':await db.message.delete(where={'id':replies[1].id})
    elif change=='extra_reply':await db.message.create(data={'conversationId':user.conversationId,'role':'assistant','content':'extra','metadata':Json({'achievement_turn_id':args['turn_id']})})
    elif change=='scope':await db.conversation.update(where={'id':user.conversationId},data={'isDeleted':True})
    elif change=='human':await db.message.update(where={'id':user.id},data={'metadata':Json({'human_takeover':True})})
    else:await db.message.delete(where={'id':replies[0].id})
    await t.run_turn_followup({'turn_id':args['turn_id'],'generation':0})
    item = await receipt(db,args['turn_id'])
    assert item is None if change=='root_deleted' else item['status']=='cancelled'
    assert await db.achievementunlock.count(where={'conversationId':user.conversationId,'achievementId':81})==0


@pytest.mark.parametrize('bad', [None,[],['missing'],['x']*101,[7]])
async def test_incomplete_user_identity_is_not_guessed(turn,bad):
    db,_,_,args = turn
    assert not await t.register_completed_turn(**{**args,'user_message_ids':bad})
    assert await receipt(db,args['turn_id']) is None


async def test_registration_identity_conflict_does_not_overwrite_original(turn):
    db,_,replies,args = turn
    assert await t.register_completed_turn(**args)
    before = await receipt(db,args['turn_id'])
    await db.message.update(where={'id':replies[1].id},data={'content':'changed'})
    with pytest.raises(t.TurnSourceUnavailable,match='completion_identity_conflict'):
        await t.register_completed_turn(**args)
    assert await receipt(db,args['turn_id']) == before


async def test_parallel_registration_and_consumption_do_not_duplicate_unlock(turn):
    db,user,_,args = turn
    assert all(await asyncio.gather(*(t.register_completed_turn(**args) for _ in range(3))))
    await asyncio.gather(*(t.run_turn_followup({'turn_id':args['turn_id'],'generation':0}) for _ in range(3)))
    assert (await receipt(db,args['turn_id']))['status']=='completed'
    assert await db.achievementunlock.count(where={'conversationId':user.conversationId,'achievementId':81})==1


async def test_future_generation_and_private_database_errors_do_not_ack(turn,monkeypatch):
    db,_,_,args = turn
    await t.register_completed_turn(**args)
    with pytest.raises(t.TurnFollowupFailed,match='RuntimeError'):
        await t.run_turn_followup({'turn_id':args['turn_id'],'generation':1})
    assert (await receipt(db,args['turn_id']))['status']=='pending_dispatch'
    monkeypatch.setattr(t,'db',type('Broken',(),{'query_raw':AsyncMock(side_effect=RuntimeError('PRIVATE_SQL')),
        'execute_raw':AsyncMock(side_effect=RuntimeError('PRIVATE_RECEIPT'))})())
    with pytest.raises(t.TurnFollowupFailed,match='^RuntimeError$'):
        await t.run_turn_followup({'turn_id':args['turn_id'],'generation':0})


@pytest.mark.parametrize('invalid', [{}, {'turn_id': True,'generation':0},
    {'turn_id':'x'*201,'generation':0}, {'turn_id':'x','generation':True},
    {'turn_id':'x','generation':-1}])
async def test_invalid_turn_payload_rejected(invalid):
    with pytest.raises(ValueError,match='invalid_turn_followup_payload'):
        await t.run_turn_followup(invalid)


@pytest.mark.parametrize('change', ['workspace_closed','root_proactive','child_proactive','no_root','too_many_replies','wrong_conversation'])
async def test_registration_rejects_ineligible_sources(turn, change):
    db,user,replies,args = turn
    if change == 'workspace_closed':
        workspace = await db.conversation.find_unique(where={'id':user.conversationId})
        await db.chatworkspace.update(where={'id':workspace.workspaceId},data={'status':'archived'})
    elif change in {'root_proactive','child_proactive'}:
        reply = replies[0 if change == 'root_proactive' else 1]
        await db.message.update(where={'id':reply.id},data={'metadata':Json({'achievement_turn_id':args['turn_id'],'proactive':True})})
    elif change == 'no_root':
        args = {**args,'message_id':'missing'}
    elif change == 'wrong_conversation':
        args = {**args,'conversation_id':'other'}
    else:
        await db.message.create_many(data=[{'id':uuid4().hex,'conversationId':user.conversationId,
            'role':'assistant','content':'bounded','metadata':Json({'achievement_turn_id':args['turn_id']})} for _ in range(100)])
    assert not await t.register_completed_turn(**args)
    assert await receipt(db,args['turn_id']) is None


async def test_missing_receipt_and_disabled_mode_are_explicit(turn,monkeypatch):
    from app.services.achievements import mode, source
    db,_,_,args = turn
    payload = {'turn_id':args['turn_id'],'generation':0}
    await t.run_turn_followup(payload)
    await t.register_completed_turn(**args)
    monkeypatch.setattr(mode,'achievement_evaluation_enabled',AsyncMock(return_value=False))
    await t.run_turn_followup(payload)
    assert (await receipt(db,args['turn_id']))['outcome']=='achievement_disabled'
    # A terminal event remains terminal on duplicate delivery.
    await t.run_turn_followup(payload)


async def test_late_source_rejection_records_qualification_outcome(turn,monkeypatch):
    from app.services.achievements import source
    db,_,_,args = turn
    await t.register_completed_turn(**args)
    monkeypatch.setattr(source,'resolve_message_event',AsyncMock(side_effect=source.AchievementSourceUnavailable('gone')))
    await t.run_turn_followup({'turn_id':args['turn_id'],'generation':0})
    assert (await receipt(db,args['turn_id']))['outcome']=='achievement_source_unavailable'
