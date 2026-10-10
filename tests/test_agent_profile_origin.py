"""Initialization forwards its actual frozen inputs, not invented receipts."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from app.services import agent_initialization as init


@pytest.mark.parametrize('imported',[False,True])
async def test_initialization_passes_actual_profile_career_and_input_origin(monkeypatch,imported):
    agent=SimpleNamespace(id='synthetic-agent',userId='synthetic-owner',name='Synthetic',gender='女')
    profile={'identity':{'name':'Synthetic','location':'Synthetic home','age':27}}
    career={'title':'Synthetic career'};personality={'warmth':75}
    monkeypatch.setattr(init,'db',SimpleNamespace(aiagent=SimpleNamespace(update=AsyncMock())))
    for name in ('set_progress','activate_agent','dispatch_first_greeting_for_agent','generate_daily_schedule'):
        monkeypatch.setattr(init,name,AsyncMock())
    monkeypatch.setattr(init,'seven_dim_to_mbti',AsyncMock(return_value={'EI':50,'summary':'Synthetic'}))
    monkeypatch.setattr(init,'build_mbti',AsyncMock(return_value={'EI':50}))
    from app.services.speech_output import voices
    monkeypatch.setattr(voices,'ensure_agent_voice',AsyncMock())
    monkeypatch.setattr(init,'pick_random_active_career',AsyncMock(return_value=career))
    monkeypatch.setattr(init,'generate_full_profile',AsyncMock(return_value=profile))
    monkeypatch.setattr(init,'_apply_postprocess_overrides',lambda value,**kwargs:value)
    monkeypatch.setattr(init,'_safe_life_overview',AsyncMock(return_value='Synthetic overview'))
    captured=[]
    async def memories(**kwargs):
        captured.append(kwargs)
        snapshot=json.loads(kwargs['origin'].payload)
        assert snapshot['profile']==kwargs['profile'] and snapshot['career']==career
        assert snapshot['invocation_inputs']['personality']==personality
        assert snapshot['invocation_inputs']['name']=='Synthetic'
        assert snapshot['invocation_inputs']['mbti']=={'EI':50}
        assert snapshot['invocation_inputs']['profile_override']==(profile if imported else None)
        kwargs['profile']['identity']['location']='Caller mutation after snapshot'
        assert json.loads(kwargs['origin'].payload)['profile']['identity']['location']=='Synthetic home'
        return 12
    monkeypatch.setattr(init,'generate_l1_coverage',memories)
    await init._run_agent_initialization_inner(agent,'synthetic-owner','synthetic-space',personality,
        profile_override=profile if imported else None,career_template_override=career if imported else None)
    assert len(captured)==1 and captured[0]['workspace_id']=='synthetic-space'
    assert captured[0]['origin'].kind==('imported_profile' if imported else 'generated_profile')
    init.activate_agent.assert_awaited_once_with(agent.id)
    assert init.set_progress.await_args.args[1]=='complete'


async def test_invalid_origin_does_not_activate_or_replace_persona(monkeypatch):
    agent=SimpleNamespace(id='synthetic-agent',name='Synthetic',gender='女')
    monkeypatch.setattr(init,'db',SimpleNamespace(aiagent=SimpleNamespace(update=AsyncMock())))
    monkeypatch.setattr(init,'seven_dim_to_mbti',AsyncMock(side_effect=RuntimeError('controlled mbti failure')))
    monkeypatch.setattr(init,'set_progress',AsyncMock())
    monkeypatch.setattr(init,'generate_l1_coverage',AsyncMock())
    monkeypatch.setattr(init,'activate_agent',AsyncMock())
    monkeypatch.setattr(init,'_apply_postprocess_overrides',lambda value,**kwargs:value)
    await init._run_agent_initialization_inner(agent,'owner','space',{},
        profile_override={'invalid':float('nan')},career_template_override={})
    init.generate_l1_coverage.assert_not_awaited();init.activate_agent.assert_not_awaited()
    assert init.set_progress.await_args.args[1]=='failed'
