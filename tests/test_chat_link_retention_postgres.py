"""Real PG retention, legacy reference and file recovery contracts on synthetic data."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from prisma import Json
from prisma.errors import DataError, RawQueryError
import pytest

from app.services.chat_links import retention as r, cards, repo
from app.services.chat_links.extraction import LinkMetadata
from app.services.chat_media import storage
from app.services.auth import create_jwt
from tests.test_chat_link_snapshots_postgres import links, origins


@pytest.fixture
async def governed(links, monkeypatch, tmp_path):
    db, uid, users, convs, prepare, save = links
    monkeypatch.setattr(r, 'db', db)
    monkeypatch.setattr(storage, '_MEDIA_DIR', tmp_path)
    await db.execute_raw('DELETE FROM chat_link_cover_objects')
    await db.execute_raw("UPDATE chat_link_storage_state SET reserved_bytes=0,cursors='{}',last_run=NULL,last_preview=NULL,run_id=NULL,running_since=NULL")
    try:
        yield db, uid, convs, prepare, save, tmp_path
    finally:
        await db.execute_raw('DELETE FROM chat_link_cover_objects')
        await db.execute_raw("UPDATE chat_link_storage_state SET reserved_bytes=0,cursors='{}',last_run=NULL,last_preview=NULL,run_id=NULL,running_since=NULL")


async def age(db, link):
    # Recreate an unbound immutable version with a historical recorded_at. Never
    # disable its trigger just to make a fixture pass.
    row = (await db.query_raw('DELETE FROM chat_link_snapshots WHERE id=$1 RETURNING *', link.snapshot_id))[0]
    await db.execute_raw("INSERT INTO chat_link_snapshots(id,link_card_id,payload,recorded_at) VALUES ($1,$2,$3::jsonb,NOW()-INTERVAL '8 days')",
                         row['id'], row['link_card_id'], json.dumps(row['payload']))
    await db.execute_raw("UPDATE chat_link_cards SET updated_at=NOW()-INTERVAL '8 days' WHERE id=$1", link.id)


async def test_unused_preview_dry_run_then_cleanup_is_bounded_and_idempotent(governed):
    db, uid, convs, prepare, save, root = governed
    link = await prepare('private-body-never-in-report')
    await age(db, link)
    report = await r.status()
    assert report['last_run'] is None and report['last_preview'] is None
    assert report['sample']['snapshots']['eligible'] == 1
    assert 'private-body' not in str(report) and link.id not in str(report)
    result = await r.preview()
    assert result['status'] == 'completed'
    assert await repo.find_link_card(link_id=link.id,user_id=uid,conversation_id=convs[0])
    result = await r.execute()
    assert result['deleted'] == {'snapshots':1,'previews':1,'covers':0}
    assert result['sql_outcome'] == 'committed'
    assert await repo.find_link_card(link_id=link.id,user_id=uid,conversation_id=convs[0]) is None
    assert (await r.execute())['deleted'] == {'snapshots':0,'previews':0,'covers':0}
    assert (await r.status())['last_preview']['status'] == 'completed'


@pytest.mark.parametrize('reference', ['projection','component','legacy','sql','pointer','assistant'])
async def test_all_historical_reference_shapes_protect_live_resources(governed, reference):
    db, uid, convs, prepare, save, root = governed
    link = await prepare(role='assistant' if reference=='assistant' else 'user')
    await age(db, link)
    projection = cards.metadata_for_link_card(link)
    component = cards.component_card_for_link(link)
    metadata = {'link_card':projection,'component_card':component}
    if reference == 'projection': metadata={'link_card':projection}
    if reference == 'component': metadata={'component_card':component}
    if reference == 'legacy':
        metadata={'link_card':cards.metadata_for_link_card(replace(link,snapshot_id=None,snapshot_recorded_at=None))}
    if reference in ('sql','pointer'): metadata={}
    msg = await db.message.create(data={'conversationId':convs[0],'role':link.role,'content':'synthetic','metadata':Json(metadata)})
    if reference=='sql': await db.execute_raw('INSERT INTO chat_link_message_bindings(message_id,snapshot_id) VALUES ($1,$2)',msg.id,link.snapshot_id)
    if reference=='pointer':
        await repo.bind_link_card_to_message(link_id=link.id,message_id=msg.id,user_id=uid,conversation_id=convs[0])
        await db.execute_raw("UPDATE chat_link_cards SET updated_at=NOW()-INTERVAL '8 days' WHERE id=$1",link.id)
    assert (await r.execute())['deleted']=={'snapshots':0,'previews':0,'covers':0}
    exact=await repo.find_link_card(link_id=link.id,snapshot_id=link.snapshot_id,user_id=uid,conversation_id=convs[0])
    assert exact.title==link.title


async def test_recent_versions_survive_and_deleting_conversation_releases_bindings(governed):
    db, uid, convs, prepare, save, root = governed
    link=await prepare();msg=await save(link)
    await db.execute_raw('INSERT INTO chat_link_message_bindings(message_id,snapshot_id) VALUES ($1,$2)',msg.id,link.snapshot_id)
    assert (await r.execute())['deleted']['snapshots']==0
    async with db.tx() as tx:
        await tx.message.delete_many(where={'conversationId':convs[0]})
        await tx.conversation.delete(where={'id':convs[0]})
    assert await repo.find_link_card(link_id=link.id,user_id=uid,conversation_id=convs[0]) is None


async def test_cursor_advances_past_protected_rows_and_wraps(governed, monkeypatch):
    db, uid, convs, prepare, save, root=governed
    monkeypatch.setattr(r,'SCAN_LIMIT',2)
    for i in range(5):
        link=await repo.create_or_update_link_card(user_id=uid,conversation_id=convs[0],metadata=LinkMetadata(
            source_url=f'https://example.com/{i}',final_url=f'https://example.com/{i}',platform='链接',title=str(i)))
        await age(db,link)
        if i%2==0: await save(link)
    totals={'snapshots':0,'previews':0,'covers':0}
    for _ in range(8):
        out=await r.execute()
        assert all(x['sampled']<=2 for x in out['sample'].values())
        assert all(v<=2 for v in out['deleted'].values())
        for k,v in out['deleted'].items():totals[k]+=v
    assert totals=={'snapshots':2,'previews':2,'covers':0}
    assert len(await db.query_raw('SELECT id FROM chat_link_cards WHERE user_id=$1',uid))==3


async def test_managed_cover_recovery_and_unknown_files_are_preserved(governed):
    db,uid,convs,prepare,save,root=governed
    unknown=root/'old-unregistered.jpg';unknown.write_bytes(b'old')
    key=await r.write_cover(user_id=uid,conversation_id=convs[0],blob=b'new',mime='image/jpeg')
    assert (root/key).read_bytes()==b'new'
    assert (await r.status())['managed_cover_reserved_bytes']==3
    assert (await r.execute())['deleted']['covers']==0
    await db.execute_raw("UPDATE chat_link_cover_objects SET created_at=NOW()-INTERVAL '8 days'")
    assert (await r.execute())['deleted']['covers']==1
    assert not (root/key).exists() and unknown.read_bytes()==b'old'
    assert (await r.status())['managed_cover_reserved_bytes']==0
    # A process may die after reserving bytes but before creating a file.
    await r.reserve_cover('absent.jpg',8)
    await db.execute_raw("UPDATE chat_link_cover_objects SET created_at=NOW()-INTERVAL '8 days'")
    assert (await r.execute())['deleted']['covers']==1


@pytest.mark.parametrize('reference',['cache','snapshot','message','component'])
async def test_cover_references_protect_and_retiring_files_reject_new_refs(governed,reference):
    db,uid,convs,prepare,save,root=governed
    key=await r.write_cover(user_id=uid,conversation_id=convs[0],blob=b'image',mime='image/png')
    url=storage.media_url(key)
    await db.execute_raw("UPDATE chat_link_cover_objects SET created_at=NOW()-INTERVAL '8 days'")
    link=await prepare()
    if reference=='cache': await db.execute_raw('UPDATE chat_link_cards SET image_url=$1 WHERE id=$2',url,link.id)
    elif reference=='snapshot':
        row=(await db.query_raw('SELECT payload FROM chat_link_snapshots WHERE id=$1',link.snapshot_id))[0]['payload'];row['image_url']=url
        await db.execute_raw('INSERT INTO chat_link_snapshots(id,link_card_id,payload) VALUES ($1,$2,$3::jsonb)',str(uuid4()),link.id,json.dumps(row))
    else:
        metadata={'link_card':{'image_url':url}} if reference=='message' else {'component_card':{'type':'external_link','payload':{'image_url':url}}}
        await db.message.create(data={'conversationId':convs[0],'role':'user','content':'fixture','metadata':Json(metadata)})
    assert (await r.execute())['deleted']['covers']==0 and (root/key).exists()


async def test_retirement_commit_before_unlink_and_failed_file_retries(governed,monkeypatch):
    db,uid,convs,prepare,save,root=governed
    link=await prepare()
    key=await r.write_cover(user_id=uid,conversation_id=convs[0],blob=b'image',mime='image/png')
    await db.execute_raw("UPDATE chat_link_cover_objects SET created_at=NOW()-INTERVAL '8 days'")
    real=r._unlink
    def fail(_):raise OSError('secret-file-path')
    monkeypatch.setattr(r,'_unlink',fail)
    result=await r.execute()
    assert result['status']=='failed' and result['file_failures']==1
    assert 'secret-file' not in str(await r.status())
    assert (await db.query_raw('SELECT state FROM chat_link_cover_objects WHERE storage_key=$1',key))[0]['state']=='deleting'
    with pytest.raises((RawQueryError,DataError),match='retiring'):
        await db.execute_raw('UPDATE chat_link_cards SET image_url=$1 WHERE id=$2',storage.media_url(key),link.id)
    with pytest.raises((RawQueryError,DataError),match='retiring'):
        await db.message.create(data={'conversationId':convs[0],'role':'user','content':'fixture','metadata':Json({'link_card':{'image_url':storage.media_url(key)}})})
    monkeypatch.setattr(r,'_unlink',real)
    assert (await r.execute())['deleted']['covers']==1
    assert (await r.status())['managed_cover_reserved_bytes']==0


async def test_budget_reservation_is_atomic_and_disk_floor_does_not_create_files(governed,monkeypatch):
    db,uid,convs,prepare,save,root=governed
    monkeypatch.setattr(r,'COVER_BUDGET_BYTES',10)
    assert await r.reserve_cover('first.jpg',6)
    assert not await r.reserve_cover('second.jpg',6)
    assert not await r.reserve_cover('empty.jpg',0)
    assert not await r.reserve_cover('large.jpg',storage._MAX_IMAGE_BYTES+1)
    monkeypatch.setattr(r.shutil,'disk_usage',lambda _:type('Usage',(),{'free':0})())
    assert await r.write_cover(user_id=uid,conversation_id=convs[0],blob=b'file',mime='image/png') is None
    assert list(root.iterdir())==[]
    assert (await r.status())['managed_cover_reserved_bytes']==6


async def test_exclusion_stale_recovery_and_message_cleanup_serialization(governed):
    db,uid,convs,prepare,save,root=governed
    link=await prepare();await age(db,link)
    async with r.locked(db):
        with pytest.raises(r.MaintenanceBusy):await r.execute()
    await db.execute_raw("UPDATE chat_link_storage_state SET run_id='interrupted',running_since=NOW()")
    with pytest.raises(r.MaintenanceBusy):await r.execute()
    await db.execute_raw("UPDATE chat_link_storage_state SET running_since=NOW()-INTERVAL '11 minutes'")
    await save(link)
    assert (await r.execute())['deleted']['snapshots']==0
    # The migration guard fences legacy JSON message writes as well as SQL binds.
    async with r.locked(db) as tx:
        writer=asyncio.create_task(save(link))
        await asyncio.sleep(.05)
        assert not writer.done()
    await writer
    assert (await r.execute())['deleted']['snapshots']==0


async def test_missing_foreign_versions_fail_closed_and_non_link_updates_still_work(governed):
    db,uid,convs,prepare,save,root=governed
    link=await prepare();msg=await save(link)
    bad=cards.metadata_for_link_card(replace(link,snapshot_id='missing'))
    with pytest.raises((RawQueryError,DataError),match='unavailable'):
        await db.message.create(data={'conversationId':convs[0],'role':'user','content':'bad','metadata':Json({'link_card':bad})})
    with pytest.raises((RawQueryError,DataError),match='unavailable'):
        await db.message.create(data={'conversationId':convs[1],'role':'user','content':'bad','metadata':Json({'link_card':cards.metadata_for_link_card(link)})})
    metadata=dict(msg.metadata);metadata['trace_id']='synthetic'
    await db.message.update(where={'id':msg.id},data={'metadata':Json(metadata)})
    assert (await db.message.find_unique(where={'id':msg.id})).metadata['trace_id']=='synthetic'


async def test_admin_api_auth_policy_and_real_operations(governed,monkeypatch):
    db,uid,convs,prepare,save,root=governed
    from app.api.admin import stats
    app=FastAPI();app.include_router(stats.router)
    headers={'Authorization':'Bearer '+create_jwt(uid,'admin')}
    async with AsyncClient(transport=ASGITransport(app=app),base_url='http://isolated') as client:
        path='/admin-api/stats/link-storage'
        for method,suffix in [('GET',''),('POST','/preview'),('POST','/execute?confirmed=true&policy_version='+r.POLICY_VERSION)]:
            assert (await client.request(method,path+suffix)).status_code==401
            assert (await client.request(method,path+suffix,headers={'Authorization':'Bearer '+create_jwt(uid,'user')})).status_code==403
        assert (await client.post(path+'/execute?confirmed=true&policy_version=old',headers=headers)).status_code==409
        assert (await client.post(path+'/execute?policy_version='+r.POLICY_VERSION,headers=headers)).status_code==409
        assert (await client.get(path,headers=headers)).json()['last_run'] is None
        assert (await client.post(path+'/preview',headers=headers)).status_code==200
        assert (await client.post(path+'/execute?confirmed=true&policy_version='+r.POLICY_VERSION,headers=headers)).json()['status']=='completed'
        async with r.locked(db):
            assert (await client.post(path+'/execute?confirmed=true&policy_version='+r.POLICY_VERSION,headers=headers)).status_code==409
        for name,suffix in [('status',''),('preview','/preview'),('execute','/execute?confirmed=true&policy_version='+r.POLICY_VERSION)]:
            with monkeypatch.context() as m:
                m.setattr(r,name,AsyncMock(side_effect=RuntimeError('private SQL secret')))
                out=await client.request('GET' if name=='status' else 'POST',path+suffix,headers=headers)
                assert out.status_code==503 and 'secret' not in out.text


async def test_sql_failure_records_uncertain_attempt_without_claiming_success(governed,monkeypatch):
    db,uid,convs,prepare,save,root=governed
    link=await prepare();await age(db,link)
    sample=r._sample
    monkeypatch.setattr(r,'_sample',AsyncMock(side_effect=RuntimeError('secret internal SQL')))
    with pytest.raises(RuntimeError):await r.execute()
    monkeypatch.setattr(r,'_sample',sample)
    report=await r.status()
    assert report['last_run']['status']=='failed' and report['last_run']['reason']=='RuntimeError'
    assert report['running_since'] is None and 'secret' not in str(report)
    assert (await r.execute())['deleted']['snapshots']==1


async def test_interrupted_file_retirement_is_recoverable(governed,monkeypatch):
    db,uid,convs,prepare,save,root=governed
    key=await r.write_cover(user_id=uid,conversation_id=convs[0],blob=b'abc',mime='image/jpeg')
    await db.execute_raw("UPDATE chat_link_cover_objects SET created_at=NOW()-INTERVAL '8 days'")
    real=r._unlink
    def cancel(_):raise asyncio.CancelledError()
    monkeypatch.setattr(r,'_unlink',cancel)
    with pytest.raises(asyncio.CancelledError):await r.execute()
    assert (await r.status())['last_run']['status']=='interrupted'
    monkeypatch.setattr(r,'_unlink',real)
    assert (await r.execute())['deleted']['covers']==1
    assert not (root/key).exists()


async def test_lost_reservation_commit_never_writes_unregistered_file(governed):
    db,uid,convs,prepare,save,root=governed
    from contextlib import asynccontextmanager
    class Lost:
        @asynccontextmanager
        async def tx(self,**kwargs):
            async with db.tx(**kwargs) as tx:yield tx
            raise ConnectionError('synthetic lost commit acknowledgement')
    with pytest.raises(ConnectionError):
        await r.write_cover(user_id=uid,conversation_id=convs[0],blob=b'abc',mime='image/jpeg',database=Lost())
    assert list(root.iterdir())==[]
    assert (await r.status())['managed_cover_reserved_bytes']==3
    await db.execute_raw("UPDATE chat_link_cover_objects SET created_at=NOW()-INTERVAL '8 days'")
    assert (await r.execute())['deleted']['covers']==1


async def test_failed_retirement_does_not_starve_later_objects(governed,monkeypatch):
    db,uid,convs,prepare,save,root=governed
    monkeypatch.setattr(r,'SCAN_LIMIT',1)
    for key in ['a.jpg','b.jpg']:
        await r.reserve_cover(key,3);(root/key).write_bytes(b'abc')
    await db.execute_raw("UPDATE chat_link_cover_objects SET state='deleting',created_at=NOW()-INTERVAL '8 days'")
    real=r._unlink
    def unlink(key):
        if key=='a.jpg':raise OSError('synthetic')
        real(key)
    monkeypatch.setattr(r,'_unlink',unlink)
    assert (await r.execute())['file_failures']==1
    assert (await r.execute())['deleted']['covers']==1
    assert (root/'a.jpg').exists() and not (root/'b.jpg').exists()


async def test_cleanup_winning_race_rejects_stale_message_projection(governed):
    db,uid,convs,prepare,save,root=governed
    link=await prepare();await age(db,link)
    assert (await r.execute())['deleted']['previews']==1
    with pytest.raises(DataError,match='unavailable'):await save(link)
    assert await db.message.count(where={'conversationId':convs[0],'content':'shared'})==0
