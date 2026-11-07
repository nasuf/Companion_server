"""VOICE01.10: 推荐过的线下活动从推荐记录回忆, 不编造、不把没去的说成去过。"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.services.chat import graph_phases
from app.services.chat.prompt_builder import _build_activity_history_section
from app.services.offline import module_settings, recommendation_history as history
from app.services.prompting import defaults
from tests.test_service_health_postgres import world  # noqa: F401

DAY = datetime(2026, 11, 4, 2, tzinfo=UTC)  # 北京时间 11-04 10:00


@pytest.mark.parametrize("message", [
    "你还记得给我推荐过什么好玩的地方吗", "你推荐的那个咖啡馆叫啥", "我们一起去过哪些地方",
    "上次那个活动怎么样了", "我待出行里有什么",
])
def test_recall_questions_are_detected(message):
    assert history.asks_about_recommendations(message)


@pytest.mark.parametrize("message", ["今天好累", "推荐一部电影吧", "", None])
def test_ordinary_turns_do_not_load_records(message):
    assert not history.asks_about_recommendations(message)


def test_lines_carry_date_place_and_a_status_that_never_upgrades_to_visited():
    rows = [
        dict(title="江边自来水厂旧址", location_name="自来水厂旧址", city="镇江", status="accepted",
             reached=False, created_at=DAY),
        dict(title="红小姐咖啡馆", location_name="西津渡店", city="镇江", status="pending",
             reached=False, created_at=(DAY - timedelta(days=3)).isoformat()),
        dict(title="湖边散步", location_name="镇江湖边", city="镇江", status="completed", reached=True,
             created_at=DAY.replace(tzinfo=None)),
        dict(title="旧展览", location_name="", city="", status="cancelled", reached=False, created_at="bad"),
        dict(title="", status="pending"),
        dict(title="已到现场", location_name="公园", city=None, status="accepted", reached=True, created_at=None),
        dict(title="过期", location_name="", city="南京", status="expired", reached=False, created_at=DAY),
        dict(title="未知", status="weird", created_at=DAY),
    ]
    lines = history.render_lines(rows)
    assert lines[0] == "2026-11-04 推荐「江边自来水厂旧址」（镇江·自来水厂旧址）：对方放进了待出行，还没去"
    assert lines[1] == "2026-11-01 推荐「红小姐咖啡馆」（镇江·西津渡店）：推荐了，对方还没回应"
    assert lines[2] == "2026-11-04 推荐「湖边散步」（镇江湖边）：对方去过，已完成"
    assert lines[3] == "推荐「旧展览」：已取消，对方没去"
    assert lines[4] == "推荐「已到现场」（公园）：对方到过现场"
    assert lines[5] == "2026-11-04 推荐「过期」（南京）：已过期，对方没去"
    assert lines[6].endswith("：weird") and len(lines) == 7


async def test_section_renders_records_and_is_removed_when_disabled():
    async def template(key):
        assert key == "chat.activity_recommendation_history_section"
        return defaults.CHAT_ACTIVITY_RECOMMENDATION_HISTORY_PROMPT

    with patch("app.services.chat.prompt_builder._get_optional_prompt", side_effect=template):
        section = await _build_activity_history_section(["2026-11-04 推荐「江边」：已完成"])
        assert await _build_activity_history_section([]) is None
    assert section.prompt_key == "chat.activity_recommendation_history_section"
    assert "- 2026-11-04 推荐「江边」：已完成" in section.body and "不要编" in section.body
    with patch("app.services.chat.prompt_builder._get_optional_prompt", AsyncMock(return_value=None)):
        assert await _build_activity_history_section(["x"]) is None


def _ctx(message):
    return SimpleNamespace(user_message=message, user_id="u", workspace_id="w", response_diagnostics={},
                           services=SimpleNamespace(logger=SimpleNamespace(warning=lambda *_: None)))


async def test_chat_loads_records_only_when_asked(monkeypatch):
    monkeypatch.setattr(module_settings, "is_activity_enabled", AsyncMock(return_value=True))
    load = AsyncMock(return_value=[dict(title="江边", status="pending", created_at=DAY)])
    monkeypatch.setattr(history, "load_recent", load)
    ordinary = _ctx("今天好累")
    assert await graph_phases._load_activity_history(ordinary) is None
    assert ordinary.response_diagnostics == {} and not load.await_count
    asked = _ctx("你推荐过我什么地方")
    assert await graph_phases._load_activity_history(asked) == ["2026-11-04 推荐「江边」：推荐了，对方还没回应"]
    assert asked.response_diagnostics["activity_history"] == {"asked": True, "status": "injected", "records": 1}
    load.return_value = []
    empty = _ctx("你推荐过我什么地方")
    # 没有记录也要明说, 不让模型自己编一个
    assert await graph_phases._load_activity_history(empty) == ["（还没有给对方推荐过任何线下活动）"]


@pytest.mark.parametrize("mode", ["disabled", "read_failed"])
async def test_chat_skips_records_when_unavailable(monkeypatch, mode):
    monkeypatch.setattr(module_settings, "is_activity_enabled", AsyncMock(return_value=mode != "disabled"))
    monkeypatch.setattr(history, "load_recent", AsyncMock(side_effect=RuntimeError("db down")))
    ctx = _ctx("你推荐过我什么地方")
    assert await graph_phases._load_activity_history(ctx) is None
    assert ctx.response_diagnostics["activity_history"]["status"] == (
        "module_disabled" if mode == "disabled" else "read_failed")


async def test_records_are_scoped_to_user_and_workspace_newest_first(world, monkeypatch):
    database, conv, space, user, _ = world
    monkeypatch.setattr(history, "db", database)
    agent = (await database.conversation.find_unique(where={"id": conv})).agentId
    ids = []
    for i, (title, workspace) in enumerate((("旧的", space), ("新的", space), ("别的工作区", None))):
        ident = str(uuid4())
        ids.append(ident)
        await database.execute_raw(
            """INSERT INTO offline_activity_recommendations(id,user_id,agent_id,workspace_id,title,status,created_at,updated_at)
               VALUES($1,$2,$3,$4,$5,'pending',$6::timestamptz,$6::timestamptz)""",
            ident, user, agent, workspace, title, (DAY + timedelta(hours=i)).isoformat())
    try:
        rows = await history.load_recent(user, space)
        assert [r["title"] for r in rows] == ["新的", "旧的"]
        assert [r["title"] for r in await history.load_recent(user, None, limit=1)] == ["别的工作区"]
        assert await history.load_recent("someone-else", space) == []
    finally:
        await database.execute_raw("DELETE FROM offline_activity_recommendations WHERE id = ANY($1::text[])", ids)


async def test_lightweight_reply_paths_see_the_records_too():
    """时间回忆/轻量回复只看 context 占位符: 推荐记录也经它给到, 不会在那条路径上编造。"""
    from app.services.chat.reply_generate import generate_reply
    from tests.test_orchestrator_phases import _make_reply_generate_kwargs

    kwargs = _make_reply_generate_kwargs(
        user_message="你推荐过我什么地方", activity_history=["2026-11-04 推荐「江边」：已完成"])
    with patch("app.services.chat.reply_generate.get_chat_model"), \
            patch("app.services.chat.prompt_builder._build_personality_section", AsyncMock(return_value=None)):
        await generate_reply(**kwargs)
    context = kwargs["tier_fns"]["weak"].await_args.kwargs["context"]
    assert "你推荐过的线下活动（推荐记录）：\n- 2026-11-04 推荐「江边」：已完成" in context


async def test_section_is_in_the_turn_part_of_the_main_prompt_only_when_asked():
    from app.services.chat import prompt_builder
    from tests.test_agent_routine_reply import _prompt_text

    async def build(history_lines):
        diagnostics: dict = {}
        with patch.object(prompt_builder, "get_prompt_text", AsyncMock(side_effect=_prompt_text)), \
                patch.object(prompt_builder, "get_prompt_text_or_default", AsyncMock(side_effect=_prompt_text)):
            prompt = await prompt_builder.build_system_prompt(
                agent=SimpleNamespace(name="阿恺", values={"gender": "male"}), memories=None,
                memory_relevance="weak", reply_count=1, reply_total=150,
                activity_history=history_lines, diagnostics=diagnostics)
        return prompt, diagnostics

    prompt, _ = await build(["2026-11-04 推荐「江边」：已完成"])
    _, turn = prompt_builder.split_system_prompt(prompt)
    assert "## 你推荐过的活动" in turn and "- 2026-11-04 推荐「江边」：已完成" in turn
    prompt, diagnostics = await build(None)
    assert "## 你推荐过的活动" not in prompt
    assert "你推荐过的活动" in diagnostics["empty_prompt_sections_removed"]


async def test_call_setup_reads_records_unless_the_module_is_off(monkeypatch):
    from app.services.realtime_voice import context

    monkeypatch.setattr(history, "load_recent", AsyncMock(return_value=[
        dict(title="江边", status="accepted", reached=False, created_at=DAY)]))
    monkeypatch.setattr(module_settings, "is_activity_enabled", AsyncMock(return_value=True))
    assert await context._activity_history("u", "w") == ["2026-11-04 推荐「江边」：对方放进了待出行，还没去"]
    monkeypatch.setattr(module_settings, "is_activity_enabled", AsyncMock(return_value=False))
    assert await context._activity_history("u", "w") == []
