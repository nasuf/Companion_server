"""Persisted identity, per-row exclusions and source changes on real PG/Redis."""
from datetime import timedelta
from unittest.mock import AsyncMock

from prisma import Json
import pytest

from app.services.chat import post_process as p
from app.services.memory import evidence as e
from app.services.memory.recording import chat_sources as s, pipeline as pl, watermark as wm
from app.services.memory.recording.outcomes import MemoryProgressError
from tests.test_memory_evidence_postgres import origins
from tests.test_memory_processing_diagnostics import observations, redis


@pytest.fixture
async def chat(origins, observations, monkeypatch):
    db,uid,wid,_,msg,*_=origins
    for module in (p,s):monkeypatch.setattr(module,'db',db)
    monkeypatch.setattr(wm,'get_redis',AsyncMock(return_value=observations))
    monkeypatch.setattr(pl,'resolve_workspace_id',AsyncMock(return_value=wid))
    monkeypatch.setattr(pl,'should_extract_memory',lambda _:True)
    monkeypatch.setattr(pl,'should_memorize',AsyncMock(return_value=True))
    monkeypatch.setattr(pl,'extract_memories',AsyncMock(return_value={'memories':[]}))
    monkeypatch.setattr(p.manager,'send_event',AsyncMock())
    try:yield db,uid,wid,msg
    finally:await observations.delete(*(wm._key(msg.conversationId,side) for side in ('user','ai')))


async def test_parent_source_reads_all_persisted_bubbles_and_ignores_invented_body(chat):
    db,uid,wid,msg=chat
    replies=[]
    for text in ['实际气泡一','实际气泡二']:
        replies.append(await db.message.create(data={'conversationId':msg.conversationId,'role':'assistant',
            'content':text,'metadata':Json({'achievement_turn_id':'source-turn'})}))
    await db.message.create(data={'conversationId':msg.conversationId,'role':'assistant','content':'另一个回合',
        'metadata':Json({'achievement_turn_id':'unrelated-turn'})})
    result=await p._bg_memory_pipeline(uid,[{'id':msg.id,'role':'user','content':'伪造未保存内容'}],
        msg.conversationId,wid,source_message_ids=[msg.id],reply_turn_id='source-turn')
    inputs={call.kwargs['side']:call.args[0] for call in pl.extract_memories.await_args_list}
    assert inputs['user']=='user: '+msg.content
    assert inputs['ai']=='assistant: 实际气泡一\nassistant: 实际气泡二'
    assert all(r.status=='completed' and r.source_binding=='persisted_messages' for r in result)
    assert [r.source_count for r in result]==[1,2]
    assert await wm.get_watermark(msg.conversationId,'ai')==replies[-1].createdAt


@pytest.mark.parametrize('mutation',['missing','foreign','archived','deleted','missing_turn'])
async def test_source_resolution_fails_closed_before_any_model(chat,origins,mutation):
    db,uid,wid,msg=chat
    ids=[msg.id];turn=None
    if mutation=='missing':ids=['missing-message']
    if mutation=='foreign':
        foreign=await db.message.create(data={'conversationId':origins[-1][-1],'role':'user','content':'other owner'})
        ids=[foreign.id]
    if mutation=='archived':await db.chatworkspace.update(where={'id':wid},data={'status':'archived'})
    if mutation=='deleted':await db.conversation.update(where={'id':msg.conversationId},data={'isDeleted':True})
    if mutation=='missing_turn':turn='missing-turn'
    result=await p._bg_memory_pipeline(uid,[],msg.conversationId,wid,source_message_ids=ids,reply_turn_id=turn)
    assert all(r.status=='failed' and r.progress=='unchanged' for r in result)
    pl.extract_memories.assert_not_awaited()
    assert await wm.get_watermark(msg.conversationId,'user') is None


@pytest.mark.parametrize('ids,turn,reason',[(None,None,'source_identity_missing'),
    ([''],None,'source_identity_invalid'),(['x']*101,None,'source_identity_invalid'),
    (['x'],'x'*201,'source_identity_invalid')])
async def test_invalid_identity_never_guesses_latest_or_text(chat,ids,turn,reason):
    db,uid,wid,msg=chat
    with pytest.raises(MemoryProgressError,match=reason):
        await s.load_chat_sources(conversation_id=msg.conversationId,user_id=uid,workspace_id=wid,
            messages=[{'role':'user','content':msg.content}],source_message_ids=ids,reply_turn_id=turn)


async def test_source_read_failure_is_explicit(chat,monkeypatch):
    db,uid,wid,msg=chat
    monkeypatch.setattr(s,'db',type('Unavailable',(),{'query_raw':AsyncMock(side_effect=RuntimeError('PRIVATE'))})())
    result=await p._bg_memory_pipeline(uid,[],msg.conversationId,wid,source_message_ids=[msg.id])
    assert all(r.reason=='source_read_unavailable' and r.source_count is None for r in result)
    pl.extract_memories.assert_not_awaited()


async def test_source_bound_is_enforced_without_partial_input(chat):
    db,uid,wid,msg=chat
    await db.message.create_many(data=[{'conversationId':msg.conversationId,'role':'assistant','content':str(i),
        'metadata':Json({'achievement_turn_id':'too-many'})} for i in range(101)])
    with pytest.raises(MemoryProgressError,match='source_batch_limit'):
        await s.load_chat_sources(conversation_id=msg.conversationId,user_id=uid,workspace_id=wid,
            messages=[],source_message_ids=[msg.id],reply_turn_id='too-many')
    pl.extract_memories.assert_not_awaited()


@pytest.mark.parametrize('marker',[{'memory_extraction_exclusion':'current_state'},
    {'response_diagnostics':{'short_circuit_kind':'schedule_query'}}])
async def test_skipped_current_reply_does_not_discard_older_pending_reply(chat,marker):
    db,uid,wid,msg=chat
    older=await db.message.create(data={'conversationId':msg.conversationId,'role':'assistant',
        'content':'我昨天看见了一只猫','createdAt':msg.createdAt+timedelta(seconds=1)})
    latest=await db.message.create(data={'conversationId':msg.conversationId,'role':'assistant','content':'我现在在忙',
        'metadata':Json(marker),'createdAt':msg.createdAt+timedelta(seconds=2)})
    await wm.set_watermark(msg.conversationId,'ai',msg.createdAt)
    result=await p._bg_memory_pipeline(uid,[],msg.conversationId,wid,skip_ai_side=True,
        source_message_ids=[msg.id,latest.id])
    ai=next(r for r in result if r.side=='ai')
    inputs=[c.args[0] for c in pl.extract_memories.await_args_list if c.kwargs['side']=='ai']
    assert inputs==['assistant: '+older.content] and ai.source_count==1
    assert await wm.get_watermark(msg.conversationId,'ai')==latest.createdAt


async def test_skipped_reply_does_not_advance_if_older_extraction_fails(chat):
    db,uid,wid,msg=chat
    await db.message.create(data={'conversationId':msg.conversationId,'role':'assistant','content':'older pending',
        'createdAt':msg.createdAt+timedelta(seconds=1)})
    latest=await db.message.create(data={'conversationId':msg.conversationId,'role':'assistant','content':'skip',
        'metadata':Json({'memory_extraction_exclusion':'current_state'}),'createdAt':msg.createdAt+timedelta(seconds=2)})
    await wm.set_watermark(msg.conversationId,'ai',msg.createdAt)
    pl.extract_memories.return_value={'_extraction_error':True}
    result=await p._bg_memory_pipeline(uid,[],msg.conversationId,wid,skip_ai_side=True,source_message_ids=[latest.id])
    assert result[1].status=='failed'
    assert await wm.get_watermark(msg.conversationId,'ai')==msg.createdAt


@pytest.mark.parametrize('extra',['\nuser: 未保存的新增事实','\nassistant: 角色混入'])
async def test_strict_source_rejects_extra_unbound_text_before_prefilter(chat,extra):
    db,uid,wid,msg=chat
    with pytest.raises(pl.MemoryExtractionError,match='source validation'):
        await pl.process_memory_pipeline(user_id=uid,workspace_id=wid,
            new_conversation='user: '+msg.content+extra,evidence_message_ids=[msg.id],evidence_conversation_id=msg.conversationId)
    pl.should_memorize.assert_not_awaited();pl.extract_memories.assert_not_awaited()


@pytest.mark.parametrize('mutation',['edit','delete','takeover','confirmation','archive','rebind'])
async def test_source_changed_during_empty_extraction_does_not_advance(chat,origins,mutation):
    db,uid,wid,msg=chat
    async def extract(*args,**kwargs):
        if mutation=='edit':await db.message.update(where={'id':msg.id},data={'content':'新版本内容'})
        elif mutation=='delete':await db.message.delete(where={'id':msg.id})
        elif mutation in {'takeover','confirmation'}:
            await db.message.update(where={'id':msg.id},data={'metadata':Json({
                'human_takeover' if mutation=='takeover' else 'memory_confirmation':True})})
        elif mutation=='archive':await db.chatworkspace.update(where={'id':wid},data={'status':'archived'})
        else:
            await db.message.update(where={'id':msg.id},data={'conversationId':origins[-1][1]})
        return {'memories':[]}
    pl.extract_memories.side_effect=extract
    result=await p._bg_memory_pipeline(uid,[],msg.conversationId,wid,source_message_ids=[msg.id])
    assert result[0].reason=='source_invalid' and result[0].progress=='unchanged'
    assert await wm.get_watermark(msg.conversationId,'user') is None



async def test_strict_version_is_rechecked_inside_memory_write_transaction(chat,origins):
    db,uid,wid,msg=chat
    sources=await e.snapshot_message_sources(user_id=uid,workspace_id=wid,side='user',
        message_ids=[msg.id],extraction_input='user: '+msg.content,conversation_id=msg.conversationId)
    await db.message.update(where={'id':msg.id},data={'metadata':Json({'human_takeover':True})})
    with pytest.raises(ValueError):await origins[5](*sources)
    assert (await origins[6]())['state']=='historical_unknown'


async def test_prefilter_noop_rechecks_sources_before_acknowledging_progress(chat,monkeypatch):
    db,uid,wid,msg=chat
    from app.config import settings
    monkeypatch.setattr(settings,'enable_memory_prefilter',True)
    async def prefilter(*args,**kwargs):
        await db.message.update(where={'id':msg.id},data={'content':'changed while filtering'})
        return False
    pl.should_memorize.side_effect=prefilter
    result=await p._bg_memory_pipeline(uid,[],msg.conversationId,wid,source_message_ids=[msg.id])
    assert result[0].reason=='source_invalid'
    pl.extract_memories.assert_not_awaited()
    assert await wm.get_watermark(msg.conversationId,'user') is None


async def test_boundary_reply_and_memory_share_persisted_turn_identity(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from app.services.chat import boundary_phase as bp
    from app.services.chat.message_utils import _achievement_turn_id
    ctx=bp.BoundaryPhaseCtx(conversation_id='c',agent_id='a',user_id='u',agent=None,
        user_message='对不起',sub_intent_mode=False,parent_patience=0,
        tracer=SimpleNamespace(safe_trace_id=None),short_circuit_fn=AsyncMock(return_value=[]),
        fire_background_fn=MagicMock(),bg_memory_pipeline_fn=MagicMock(return_value=None),
        source_message_ids=['message-1','message-2'],workspace_id='w')
    _=[event async for event in bp._emit_short_circuit(ctx,'temporary apology',{})]
    bp._fire_memory_pipeline(ctx,'temporary apology')
    assert ctx.short_circuit_fn.await_args.kwargs['turn_user_message_ids']==ctx.source_message_ids
    assert ctx.bg_memory_pipeline_fn.call_args.args==('u',[])
    assert ctx.bg_memory_pipeline_fn.call_args.kwargs==dict(conversation_id='c',workspace_id='w',
        source_message_ids=ctx.source_message_ids,reply_turn_id=_achievement_turn_id(ctx.source_message_ids))


async def test_unchanged_prefilter_noop_can_acknowledge_persisted_source(chat,monkeypatch):
    db,uid,wid,msg=chat
    from app.config import settings
    monkeypatch.setattr(settings,'enable_memory_prefilter',True)
    pl.should_memorize.return_value=False
    result=await p._bg_memory_pipeline(uid,[],msg.conversationId,wid,source_message_ids=[msg.id])
    assert result[0].status=='completed' and result[0].source_count==1
    pl.extract_memories.assert_not_awaited()
    assert await wm.get_watermark(msg.conversationId,'user')==msg.createdAt


async def test_chat_pipeline_cannot_omit_persisted_ids(chat):
    _,uid,wid,msg=chat
    with pytest.raises(pl.MemoryExtractionError,match='identities missing'):
        await pl.process_memory_pipeline(user_id=uid,workspace_id=wid,new_conversation=msg.content,
            evidence_message_ids=[],evidence_conversation_id=msg.conversationId)
    pl.should_memorize.assert_not_awaited()


@pytest.mark.parametrize('change',['none','confirmation','archived','conversation'])
async def test_strict_source_binding_checks_context_inside_write(chat,origins,change):
    db,uid,wid,msg=chat
    sources=await e.snapshot_message_sources(user_id=uid,workspace_id=wid,side='user',
        message_ids=[msg.id],extraction_input='user: '+msg.content,conversation_id=msg.conversationId)
    if change=='confirmation':
        await db.message.update(where={'id':msg.id},data={'metadata':Json({'memory_confirmation':True})})
    elif change=='archived':await db.chatworkspace.update(where={'id':wid},data={'status':'archived'})
    elif change=='conversation':await db.message.update(where={'id':msg.id},data={'conversationId':origins[-1][1]})
    if change=='none':
        assert await origins[5](*sources)==1
        detail=await origins[6]()
        assert detail['state']=='linked' and detail['items'][0]['source_version']==e.content_version(msg.content)
    else:
        with pytest.raises(ValueError,match='changed_or_unavailable'):await origins[5](*sources)
        assert (await origins[6]())['state']=='historical_unknown'


async def test_legacy_nonchat_evidence_change_is_not_hidden_by_substring_match(chat):
    db,uid,wid,msg=chat
    changed='合成用户'
    assert changed in msg.content
    async def extract(*args,**kwargs):
        await db.message.update(where={'id':msg.id},data={'content':changed})
        return {'memories':[]}
    pl.extract_memories.side_effect=extract
    with pytest.raises(pl.MemoryExtractionError,match='source changed'):
        await pl.process_memory_pipeline(user_id=uid,workspace_id=wid,new_conversation='user: '+msg.content,
            evidence_message_ids=[msg.id])


async def test_nonchat_reserved_input_has_no_progress_or_extraction(chat):
    _,uid,wid,_=chat
    result=await p._pipeline_with_watermark(uid,[{'role':'assistant','content':'reserved',
        'metadata':{'real_world_type':'gift'}}],None,side='ai',workspace_id=wid)
    assert result.status=='excluded' and result.source_count==0 and result.source_binding=='unlinked_input'
    pl.extract_memories.assert_not_awaited()


def music_metadata(title='测试歌曲'):
    return {'component_card': {'type': 'music_track', 'payload': {'intent': 'invite',
        'track': {'id': 'track-1', 'title': title, 'artist': '测试歌手'}}}}


@pytest.mark.parametrize('backlog', [False, True])
async def test_card_only_music_keeps_persisted_track_as_source(chat, origins, backlog):
    db,uid,wid,msg=chat
    msg=await db.message.update(where={'id':msg.id},data={'content':'','metadata':Json(music_metadata())})
    if backlog:await wm.set_watermark(msg.conversationId,'user',msg.createdAt-timedelta(seconds=1))
    result=await p._bg_memory_pipeline(uid,[],msg.conversationId,wid,source_message_ids=[msg.id])
    assert result[0].source_count==1 and result[0].status=='completed'
    text=pl.extract_memories.await_args.args[0]
    assert text=='user: 用户在这条消息中分享了歌曲《测试歌曲》，歌手：测试歌手'
    sources=await e.snapshot_message_sources(user_id=uid,workspace_id=wid,side='user',
        message_ids=[msg.id],extraction_input=text,conversation_id=msg.conversationId)
    assert sources[0].expected_card_version is not None
    assert await origins[5](*sources)==1
    assert (await origins[6]())['items'][0]['availability']=='available'
    assert (await db.message.find_unique(where={'id':msg.id})).content==''
    await db.message.update(where={'id':msg.id},data={'metadata':Json(music_metadata('修改后的歌曲'))})
    assert (await origins[6]())['items'][0]['availability']=='changed'
    with pytest.raises(ValueError,match='card_changed'):await origins[5](*sources)


async def test_music_card_change_during_empty_extraction_keeps_progress(chat):
    db,uid,wid,msg=chat
    await db.message.update(where={'id':msg.id},data={'content':'','metadata':Json(music_metadata())})
    async def extract(*args,**kwargs):
        card=music_metadata();card['component_card']['payload']['track']['id']='different-track-same-title'
        await db.message.update(where={'id':msg.id},data={'metadata':Json(card)})
        return {'memories':[]}
    pl.extract_memories.side_effect=extract
    result=await p._bg_memory_pipeline(uid,[],msg.conversationId,wid,source_message_ids=[msg.id])
    assert result[0].reason=='source_invalid'
    assert await wm.get_watermark(msg.conversationId,'user') is None


async def test_music_card_snapshot_is_guarded_by_database_and_immutable(chat,origins):
    from uuid import uuid4
    from prisma.errors import RawQueryError
    db,uid,wid,msg=chat
    await db.message.update(where={'id':msg.id},data={'content':'','metadata':Json(music_metadata())})
    text=s.extraction_text({'content':'','role':'user','metadata':music_metadata()})
    sources=await e.snapshot_message_sources(user_id=uid,workspace_id=wid,side='user',
        message_ids=[msg.id],extraction_input='user: '+text,conversation_id=msg.conversationId)
    await origins[5](*sources)
    with pytest.raises(RawQueryError,match='card_version_changed'):
        await db.execute_raw("""INSERT INTO memory_evidence_links SELECT (jsonb_populate_record(
            NULL::memory_evidence_links,to_jsonb(e)||jsonb_build_object('id',$1::text,'source_card_version',$2::text))).*
            FROM memory_evidence_links e WHERE memory_id=$3 AND memory_source='user'""",uuid4().hex,'0'*64,origins[3])
    with pytest.raises(RawQueryError,match='immutable'):
        await db.execute_raw('UPDATE memory_evidence_links SET source_card_version=NULL WHERE memory_id=$1',origins[3])


@pytest.mark.parametrize('card', [None, {}, {'payload':{}}, {'payload':{'track':{}}},
    {'payload':{'track':{'title':' '}}}])
async def test_incomplete_music_metadata_is_not_reported_as_noop(chat,card):
    db,uid,wid,msg=chat
    metadata={'component_card':dict(card or {},type='music_track')}
    await db.message.update(where={'id':msg.id},data={'content':'','metadata':Json(metadata)})
    result=await p._bg_memory_pipeline(uid,[],msg.conversationId,wid,source_message_ids=[msg.id])
    assert result[0].reason=='source_invalid'
    pl.extract_memories.assert_not_awaited()


def test_music_text_keeps_user_words_without_inventing_agent_acceptance():
    card=music_metadata();del card['component_card']['payload']['track']['artist']
    text=s.extraction_text({'role':'user','content':'你听过吗','metadata':card})
    assert text=='你听过吗\n用户在这条消息中分享了歌曲《测试歌曲》'
    assert s.extraction_text({'role':'assistant','content':'普通回复','metadata':card})=='普通回复'


async def test_empty_plain_message_cannot_be_claimed_as_structured_source(chat):
    db,uid,wid,msg=chat
    await db.message.update(where={'id':msg.id},data={'content':''})
    with pytest.raises(ValueError,match='scope_or_content_mismatch'):
        await e.snapshot_message_sources(user_id=uid,workspace_id=wid,side='user',
            message_ids=[msg.id],extraction_input='user: ',conversation_id=msg.conversationId)
