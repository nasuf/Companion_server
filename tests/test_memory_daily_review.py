"""Scoped daily-review claims, date windows, trust boundaries and capacity."""
import asyncio
from datetime import datetime,timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock,MagicMock
from zoneinfo import ZoneInfo

import pytest

from app.services.schedule_domain import schedule
from app.services.workspace import workspaces as workspace
from app.services.llm import models
from app.services.memory.storage import persistence
from app.services.memory.lifecycle import capacity
from app.services.games import daily_digest
from app.services.memory.taxonomy import resolve_taxonomy
from tests.test_memory_lifecycle_postgres import memory_flow,memory
from tests.test_runtime_execution_foundation import flow


NOW=datetime(2026,10,9,0,30,tzinfo=ZoneInfo("Asia/Shanghai"))


async def setup_claim(f,monkeypatch):
    monkeypatch.setattr(schedule,"db",f.db)
    monkeypatch.setattr(schedule,"_local_now",lambda:NOW)
    w=await f.db.chatworkspace.find_unique(where={"id":f.ids["workspace"]})
    monkeypatch.setattr(workspace,"get_active_workspace",AsyncMock(return_value=w))


async def test_concurrent_daily_review_claims_run_once(memory_flow,monkeypatch):
    f=memory_flow;await setup_claim(f,monkeypatch)
    helper=AsyncMock(return_value=["synthetic-memory"])
    monkeypatch.setattr(schedule,"_review_daily_schedule",helper)
    results=await asyncio.gather(*(schedule.review_daily_schedule(f.ids["agent"],f.ids["owner"]) for _ in range(8)))
    assert sum(bool(r) for r in results)==1 and helper.await_count==1
    args=helper.await_args.args
    assert args[3]==f.ids["workspace"] and args[4]==NOW.replace(hour=0,minute=0)-timedelta(days=1) and args[5]==NOW.replace(hour=0,minute=0)
    assert (await f.db.query_raw("SELECT status FROM memory_daily_reviews WHERE workspace_id=$1",f.ids["workspace"]))[0]["status"]=="completed"


async def test_failed_partial_attempt_is_recorded_and_not_repeated(memory_flow,monkeypatch):
    f=memory_flow;await setup_claim(f,monkeypatch)
    helper=AsyncMock(side_effect=RuntimeError("storage failed"));monkeypatch.setattr(schedule,"_review_daily_schedule",helper)
    with pytest.raises(RuntimeError):await schedule.review_daily_schedule(f.ids["agent"],f.ids["owner"])
    assert await schedule.review_daily_schedule(f.ids["agent"],f.ids["owner"])==[]
    assert helper.await_count==1
    assert (await f.db.query_raw("SELECT status FROM memory_daily_reviews WHERE workspace_id=$1",f.ids["workspace"]))[0]["status"]=="failed"


async def test_capacity_includes_archived_originals(memory_flow,monkeypatch):
    f=memory_flow;await setup_claim(f,monkeypatch)
    monkeypatch.setattr(capacity,"DAILY_SUMMARY_ROW_BUDGET",1)
    await memory(f,"ai",provenance="daily_summary",isArchived=True)
    helper=AsyncMock();monkeypatch.setattr(schedule,"_review_daily_schedule",helper)
    assert await schedule.review_daily_schedule(f.ids["agent"],f.ids["owner"])==[]
    helper.assert_not_awaited()
    assert (await f.db.query_raw("SELECT status FROM memory_daily_reviews WHERE workspace_id=$1",f.ids["workspace"]))[0]["status"]=="capacity_skipped"


@pytest.mark.parametrize("field",["userId","agentId"])
async def test_mismatched_workspace_cannot_claim(memory_flow,monkeypatch,field):
    f=memory_flow;await setup_claim(f,monkeypatch)
    wrong=SimpleNamespace(id=f.ids["workspace"],userId=f.ids["owner"],agentId=f.ids["agent"])
    setattr(wrong,field,"other");monkeypatch.setattr(workspace,"get_active_workspace",AsyncMock(return_value=wrong))
    assert await schedule.review_daily_schedule(f.ids["agent"],f.ids["owner"])==[]
    assert not await f.db.query_raw("SELECT id FROM memory_daily_reviews WHERE workspace_id=$1",f.ids["workspace"])


async def test_summary_reads_yesterday_exact_scope_and_rejects_persona(monkeypatch):
    d=MagicMock()
    d.aidailyschedule.find_unique=AsyncMock(return_value=SimpleNamespace(scheduleData=[{"start":"09:00","end":"10:00","event":"散步"}]))
    d.scheduleadjustlog.find_many=AsyncMock(return_value=[])
    d.proactivechatlog.find_many=AsyncMock(return_value=[])
    d.chatworkspace.find_first=AsyncMock(return_value=object())
    monkeypatch.setattr(schedule,"db",d)
    monkeypatch.setattr(schedule,"get_prompt_text",AsyncMock(return_value="{schedule_text}{games_text}"))
    # The extraction template has different placeholders.
    async def prompt(key):return "{summary_text}" if key.endswith("memories") else "{schedule_text}{games_text}"
    monkeypatch.setattr(schedule,"get_prompt_text",prompt)
    monkeypatch.setattr(schedule,"get_utility_model",lambda:object())
    monkeypatch.setattr(models,"invoke_text",AsyncMock(return_value="散步后的心情不错"))
    monkeypatch.setattr(schedule,"invoke_json",AsyncMock(return_value={"memories":[
        {"type":"身份","content":"我是医生","score":99},{"type":"偏好","content":"我喜欢咖啡","score":90},
        {"type":"思维","content":"我总是这样想","score":85},{"type":"生活","content":"散步","score":99},
        {"content":"invalid","importance":"bad"},{"content":"nan","score":float("nan")},
        {"type":"情绪","content":"心情不错","score":90}]}))
    games=AsyncMock(return_value=daily_digest.GameDayDigest())
    monkeypatch.setattr(daily_digest,"collect_today_games",games)
    store=AsyncMock(side_effect=["life","emotion"]);monkeypatch.setattr(persistence,"store_memory",store)
    end=NOW.replace(hour=0,minute=0);start=end-timedelta(days=1)
    assert await schedule._review_daily_schedule("agent","owner","伙伴","ws",start,end,5)==["life","emotion"]
    assert d.aidailyschedule.find_unique.await_args.kwargs["where"]["agentId_date"]=={"agentId":"agent","date":start.replace(tzinfo=None)}
    where=d.proactivechatlog.find_many.await_args.kwargs["where"]
    assert where=={"agentId":"agent","userId":"owner","workspaceId":"ws","createdAt":{"gte":start,"lt":end}}
    games.assert_awaited_once_with(workspace_id="ws",local_day_start=start,local_day_end=end)
    for c in store.await_args_list:
        assert c.kwargs["workspace_id"]=="ws" and c.kwargs["level"]==3 and c.kwargs["importance"]==0.49
        assert c.kwargs["provenance"]=="daily_summary" and c.kwargs["occur_time"]==start


@pytest.mark.parametrize("level",[2,3])
@pytest.mark.parametrize("category,sub",[("身份","职业"),("偏好","兴趣"),("思维","价值观")])
@pytest.mark.parametrize("provenance",["profile_seed","knowledge_seed"])
def test_trusted_seed_taxonomy_survives_cooling(level,category,sub,provenance):
    result=resolve_taxonomy(main_category=category,sub_category=sub,source="ai",level=level,provenance=provenance)
    assert result.allowed and result.main_category==category
    denied=resolve_taxonomy(main_category=category,sub_category=sub,source="ai",level=level,provenance="ai_authored")
    assert not denied.allowed
