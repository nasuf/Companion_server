"""Compatibility of recommendation reshares and old unversioned previews."""
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from app.services.chat_links import cards, repo
from app.services.chat_links.refresh import needs_refresh
from app.services.runtime import chat_link_preparation as preparation
from app.services.runtime.chat_ingress_effects import ChatIngressResourceInvalid
from tests.test_chat_link_ingress_postgres import links, flow, binding_rows


async def test_user_can_reshare_own_assistant_recommendation_without_changing_source(links):
    flow,fetch,preview,receive=links
    link=await preview(role='assistant')
    result=await receive(card=cards.component_card_for_link(link),text='我想聊这个')
    msg=await flow.db.message.find_unique(where={'id':result.message_id})
    assert msg.role=='user' and msg.metadata['link_card']['role']=='assistant'
    assert (await binding_rows(flow))[0]['snapshot_id']==link.snapshot_id
    assert (await repo.list_user_link_groups(flow.ids['owner']))[0].links[0].role=='user'
    fetch.assert_not_awaited()


@pytest.mark.parametrize('role',['user','assistant'])
async def test_unversioned_invalid_preview_refreshes_using_only_server_source(links,role):
    flow,fetch,preview,receive=links
    fetch.return_value=replace(fetch.return_value,platform='微博',title='Sina Visitor System')
    link=await preview('Sina Visitor System',role=role)
    card=cards.component_card_for_link(link);del card['payload']['snapshot_id']
    card['title']='FORGED';card['body']='FORGED';card['payload']['source_url']='https://forged.invalid'
    fetch.return_value=replace(fetch.return_value,title='real',content_text='trusted updated body',final_url='https://example.com/redirected')
    result=await receive(card=card,text='看看这个')
    msg=await flow.db.message.find_unique(where={'id':result.message_id})
    assert msg.metadata['link_card']['content_text']=='trusted updated body'
    assert msg.metadata['link_card']['id']!=link.id
    assert 'FORGED' not in str(fetch.call_args) and 'forged.invalid' not in str(fetch.call_args)
    assert fetch.call_args.kwargs['url']==link.source_url


async def test_pinned_unusable_preview_stays_exact_and_foreign_assistant_is_rejected(links):
    flow,fetch,preview,receive=links
    fetch.return_value=replace(fetch.return_value,platform='微博')
    link=await preview('Sina Visitor System',role='assistant')
    result=await receive(card=cards.component_card_for_link(link))
    msg=await flow.db.message.find_unique(where={'id':result.message_id})
    assert msg.metadata['link_card']['snapshot_id']==link.snapshot_id
    other=await preview(role='assistant',conversation_id=flow.ids['other_conversation'])
    with pytest.raises(ChatIngressResourceInvalid):await receive(key='other',card=cards.component_card_for_link(other))
    fetch.assert_not_awaited()


@pytest.mark.parametrize('platform,body,expected',[
 ('链接','Sina Visitor System',False),('微博','https://example.com',True),
 ('微博','足够完整的正文'*100+' https://example.com',False),('微博','正常正文',False),
 ('微博','Sina Visitor System',True),
])
def test_shared_refresh_policy(platform,body,expected):
    from types import SimpleNamespace
    link=SimpleNamespace(platform=platform,title=body,description=body,summary='',content_text='')
    assert needs_refresh(link) is expected


async def test_refresh_validation_failure_uses_resource_error_contract(links):
    flow,fetch,preview,receive=links
    fetch.return_value=replace(fetch.return_value,platform='微博')
    link=await preview('Sina Visitor System')
    card=cards.component_card_for_link(link);del card['payload']['snapshot_id']
    fetch.side_effect=ValueError('unsupported redirect')
    with pytest.raises(ChatIngressResourceInvalid):await receive(card=card)
    assert await binding_rows(flow)==[]


@pytest.mark.parametrize('tamper',['version','origin','payload','foreign_lineage'])
async def test_binding_rejects_mismatched_or_forged_refresh_identity(links,tamper):
    flow,fetch,preview,receive=links
    from tests.test_chat_ingress_effects_postgres import submit
    link=await preview()
    card=cards.component_card_for_link(link)
    if tamper=='version':card['payload']['snapshot_id']='different'
    elif tamper=='origin':
        del card['payload']['snapshot_id'];card['payload']['link_id']='not-this-cache'
    elif tamper=='payload':card['payload']=[]
    else:
        foreign=await preview(conversation_id=flow.ids['other_conversation'])
        link=await preview(extra_metadata={'refreshed_from_link_id':foreign.id})
        card=cards.component_card_for_link(foreign);del card['payload']['snapshot_id']
    metadata={'link_card':cards.metadata_for_link_card(link),'component_card':cards.component_card_for_link(link)}
    with pytest.raises(ChatIngressResourceInvalid):await submit(flow,card=card,metadata=metadata)
    assert await binding_rows(flow)==[]
