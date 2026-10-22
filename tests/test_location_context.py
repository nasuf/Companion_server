"""双方所在地: 位置事实、主回复「你们各自在哪」段、联网检索位置、双边地点校验。

生产 2026-10-21 (用户在西安, AI 档案在杭州转塘): 模型拿转塘去搜餐厅推荐给用户,
被问"我们在一个城市吗"答"对啊", 校验器只核对 AI 自己在哪, 照样放行。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services import persona_grounding as g
from app.services.chat import location_context as lc
from app.services.chat.location_context import LocationContext
from app.services.prompting import defaults

HANGZHOU = "浙江省 杭州市 西湖区 转塘街道"
NOW = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)


def agent(city=HANGZHOU):
    return SimpleNamespace(id="agent", userId="owner", name="盛星茉", city=city,
                           age=29, occupation="果酒酿造师")


def xian(**overrides):
    values = dict(ai_place=HANGZHOU, ai_city="杭州市", user_city="西安市", user_area="雁塔区",
                  user_country="中国", age_days=0)
    values.update(overrides)
    return LocationContext(**values)


# ── 位置事实 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("place,city", [
    (HANGZHOU, "杭州市"),
    ("云南省普洱市思茅区南屏镇凤凰路社区", "普洱市"),
    ("北京市朝阳区", "北京市"),
    ("青海省黄南藏族自治州同仁市", "黄南藏族自治州"),
    ("", None),
    (None, None),
])
def test_agent_city_takes_the_prefecture_level_city(place, city):
    assert lc.agent_city(SimpleNamespace(city=place)) == city


def test_relation_needs_fresh_device_location_on_both_sides():
    assert xian().relation == "different"
    assert xian(user_city="杭州", user_area="浙江省").relation == "same"
    assert xian(age_days=lc.STALE_AFTER_DAYS + 1).relation == "unknown"
    assert xian(age_days=None).relation == "unknown"
    assert xian(ai_city=None).relation == "unknown"
    assert LocationContext(ai_city="杭州市", stated_residence="用户住在西安").relation == "unknown"


@pytest.mark.parametrize("age,text", [
    (0, "今天更新"), (1, "昨天更新"), (5, "5 天前更新"),
    (lc.STALE_AFTER_DAYS + 1, "一个多月前更新，可能已经不准"), (None, "更新时间未知"),
])
def test_describe_user_reports_device_freshness(age, text):
    assert xian(age_days=age).describe_user() == f"西安市雁塔区（对方手机定位，{text}）"


def test_describe_user_orders_province_first_and_adds_stated_residence():
    both = xian(user_area="陕西省", stated_residence="用户住在西安")
    assert both.describe_user() == "陕西省西安市（对方手机定位，今天更新）；对方说过：用户住在西安"
    assert both.user_source == "device+stated"
    assert LocationContext().describe_user() == "还不知道"
    assert LocationContext().user_source == "unknown"
    assert xian(user_area="西安市").user_place() == "西安市"


def test_web_search_location_is_the_users_never_the_agents():
    assert xian().web_search_location() == {"type": "approximate", "country": "中国", "city": "西安市"}
    assert xian(user_area="陕西省").web_search_location()["region"] == "陕西省"
    assert xian(user_country=None).web_search_location()["country"] == "中国"
    assert xian(age_days=60).web_search_location() is None
    assert LocationContext(ai_city="杭州市").web_search_location() is None


def test_verifier_facts_and_diagnostics_hold_no_coordinates():
    assert LocationContext().verifier_facts() == {"known": False}
    assert xian(stated_residence="用户住在西安").verifier_facts() == {
        "known": True, "city": "西安市雁塔区", "source": "手机定位", "stale": False,
        "stated": "用户住在西安",
    }
    assert LocationContext(stated_residence="用户住在北京").verifier_facts() == {
        "known": True, "stated": "用户住在北京",
    }
    diag = xian().diagnostics()
    assert diag == {"ai_city_known": True, "user_source": "device", "user_age_days": 0,
                    "user_stale": False, "relation": "different", "web_search_location": True}
    assert "西安" not in json.dumps(diag, ensure_ascii=False)


async def test_load_uses_granted_device_location_and_stated_residence():
    query = AsyncMock(side_effect=[
        [{"location_city": "西安市", "location_region": "雁塔区", "location_country": "中国",
          "location_permission_status": "whileInUse",
          "location_updated_at": (NOW - timedelta(days=1)).isoformat()}],
        [{"content": "用户住在西安"}],
    ])
    with patch.object(lc.db, "query_raw", query):
        ctx = await lc.load_location_context(agent(), "u1", "w1", now=NOW)
    assert (ctx.user_city, ctx.age_days, ctx.stated_residence, ctx.relation) == (
        "西安市", 1, "用户住在西安", "different")
    assert ctx.ai_city == "杭州市" and ctx.ai_place == HANGZHOU
    assert query.await_args_list[1].args[1:] == ("u1", "w1")


async def test_revoked_permission_drops_the_old_device_location():
    query = AsyncMock(side_effect=[
        [{"location_city": "西安市", "location_permission_status": "denied",
          "location_updated_at": NOW}],
        [],
    ])
    with patch.object(lc.db, "query_raw", query):
        ctx = await lc.load_location_context(agent(), "u1", None, now=NOW)
    assert ctx.user_city is None and ctx.stated_residence is None and ctx.ai_city == "杭州市"


async def test_naive_and_missing_timestamps_and_missing_user_row():
    naive = (NOW - timedelta(days=3)).replace(tzinfo=None)
    query = AsyncMock(side_effect=[
        [{"location_city": "西安市", "location_permission_status": "always",
          "location_updated_at": naive}], [],
    ])
    with patch.object(lc.db, "query_raw", query):
        assert (await lc.load_location_context(agent(), "u1", "w1", now=NOW)).age_days == 3
    query = AsyncMock(side_effect=[
        [{"location_city": "西安市", "location_permission_status": "always",
          "location_updated_at": "not-a-date"}], [],
    ])
    with patch.object(lc.db, "query_raw", query):
        assert (await lc.load_location_context(agent(), "u1", "w1", now=NOW)).age_days is None
    with patch.object(lc.db, "query_raw", AsyncMock(side_effect=[[], []])):
        assert (await lc.load_location_context(agent(), "u1", "w1")).user_city is None


async def test_read_failure_degrades_to_unknown_user_location():
    with patch.object(lc.db, "query_raw", AsyncMock(side_effect=RuntimeError("db down"))):
        ctx = await lc.load_location_context(agent(None), "u1", "w1")
    assert ctx == LocationContext()


@pytest.mark.parametrize("question", [
    "我们在一个地方吗", "我们在一个城市吗", "我是哪里的？", "我在哪", "你知道我在哪吗",
    "离我远吗", "我这边附近有什么好吃的",
])
def test_questions_about_where_the_user_is(question):
    assert lc.asks_about_user_location(question)


@pytest.mark.parametrize("question", [
    "我不是刚跟你说过我在哪上班吗", "有推荐吗", "你在干嘛", "今天天气好好啊 你在干嘛啊",
])
def test_ordinary_questions_are_not_location_questions(question):
    assert not lc.asks_about_user_location(question)


@pytest.mark.parametrize("text", [
    "对啊！我就在转塘呢 这家店就在我附近呀！", "你也在杭州吗？太巧了吧！", "我们都在杭州", "离你不远",
    "转塘就有一家砂锅粥店！要不要去这家试试呀？", "我刚刷到转塘有家湘菜馆！要不要去试试？",
])
def test_claims_about_the_user_enter_verification(text):
    assert lc.claims_user_location(text)


@pytest.mark.parametrize("text", ["你在干嘛呀", "转塘就有一家砂锅粥店", "你那边天气真这么好？"])
def test_ordinary_replies_are_not_user_location_claims(text):
    assert not lc.claims_user_location(text)


# ── 主回复「你们各自在哪」段 ─────────────────────────────────────────────────


async def _section(location, template=defaults.CHAT_LOCATION_CONTEXT_SECTION_PROMPT):
    from app.services.chat import prompt_builder as pb

    with patch.object(pb, "_get_optional_prompt", AsyncMock(return_value=template)):
        return await pb._build_location_section(location)


async def test_section_states_both_cities_and_that_they_differ():
    section = await _section(xian())
    assert section.prompt_key == "chat.location_context_section"
    assert f"你在：{HANGZHOU}" in section.body
    assert "对方在：西安市雁塔区（对方手机定位，今天更新）" in section.body
    assert "你们不在同一个城市" in section.body
    assert "以对方所在地为准" in section.body


async def test_section_without_user_location_forbids_assuming_same_city():
    section = await _section(LocationContext(ai_place=None))
    assert "你在：（档案里没写城市）" in section.body
    assert "对方在：还不知道" in section.body
    assert "不确定是否在同一个城市" in section.body
    assert "不要假设你们在同一个城市" in section.body


async def test_section_absent_without_context_or_when_disabled():
    assert await _section(None) is None
    assert await _section(xian(), template=None) is None


async def test_system_prompt_places_the_section_with_the_slow_changing_context():
    from app.services.chat import prompt_builder as pb
    from tests.test_prompt_layout_cache import _patch_prompts

    p1, p2 = _patch_prompts()
    with p1, p2:
        prompt = await pb.build_system_prompt(agent=agent(), location_context=xian(),
                                              portrait="喜欢历史")
        diag: dict = {}
        await pb.build_system_prompt(agent=agent(), diagnostics=diag)
    assert "## 你们各自在哪" in prompt
    assert prompt.index("## 用户画像") < prompt.index("## 你们各自在哪")
    assert "你们各自在哪" in json.dumps(diag, ensure_ascii=False)


# ── 联网检索带对方城市 ────────────────────────────────────────────────────


async def test_web_search_request_carries_user_location(monkeypatch):
    from app.services.llm import ark_web_search

    bodies: list[dict] = []

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"output": [{"type": "message", "content": [{"type": "output_text",
                                                                 "text": "西安这家不错"}]}]}

    class Client:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def post(self, _url, *, headers, json):
            bodies.append(json)
            return Response()

    monkeypatch.setattr(ark_web_search.settings, "ark_api_key", "k")
    monkeypatch.setattr(ark_web_search.httpx, "AsyncClient", Client)
    location = xian().web_search_location()
    await ark_web_search.generate_with_web_search(
        [{"role": "user", "content": "有推荐吗"}], model="doubao", user_location=location,
    )
    await ark_web_search.generate_with_web_search([{"role": "user", "content": "x"}], model="doubao")
    assert bodies[0]["tools"] == [{"type": "web_search", "user_location": location}]
    assert bodies[1]["tools"] == [{"type": "web_search"}]


async def test_reply_generation_threads_the_location_to_web_search(monkeypatch):
    from app.services.chat import reply_generate as rg

    calls: list = []

    async def fake_generate(messages, *, model, **kwargs):
        calls.append(kwargs)
        return "联网结果"

    monkeypatch.setattr("app.services.llm.ark_web_search.generate_with_web_search", fake_generate)
    monkeypatch.setattr("app.services.runtime_config.resolve_for_current",
                        lambda: SimpleNamespace(remote_chat_model="doubao"))
    location = xian().web_search_location()
    text, _ = await rg._run_main_llm([{"role": "user", "content": "x"}], needs_web_search=True,
                                     web_search_location=location)
    await rg._run_main_llm([{"role": "user", "content": "x"}], needs_web_search=True)
    assert text == "联网结果"
    assert calls == [{"user_location": location}, {}]


# ── 双边地点校验 ──────────────────────────────────────────────────────────


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.setattr(g, "get_prompt_text",
                        AsyncMock(return_value=defaults.PERSONA_GROUNDING_CHECK_PROMPT))
    monkeypatch.setattr(g, "get_grounding_model", lambda: object())
    call = AsyncMock(return_value={"verdicts": [{"index": 0, "about": "user", "allowed": False}]})
    monkeypatch.setattr(g, "invoke_json", call)
    return call


async def test_false_co_location_answer_is_corrected_with_both_cities(checker):
    text, corrected = await g.guard_reply(
        agent(), "转塘这家店就在我附近呀！要不要去这家试试", question="砂锅粥不错 让我看看",
        user_location=xian(),
    )
    assert corrected and text == "刚才说岔了，我在杭州，你在西安，我们不在一个城市哈"
    prompt = checker.await_args.args[1]
    assert '对方位置：{"city": "西安市雁塔区", "known": true' in prompt
    assert "顺着用户“我们在一个地方吗”的提问答“对啊”也算断言" in prompt
    # 线上验收 (qwen3.5-plus): 只写"让对方去"时模型按第 1 条"推荐地点允许"放行了转塘的店
    assert "哪怕只是推荐、没说角色本人到场" in prompt
    assert "但说对方在哪、让对方现在去哪，还要过第 7 条" in prompt


async def test_question_about_the_user_forces_a_check_even_without_a_claim(checker):
    _, corrected = await g.guard_reply(
        agent(), "杭州的吧 哈哈", question="我是哪里的？", user_location=xian(),
    )
    assert corrected and checker.await_count == 1


async def test_user_claims_are_only_checked_on_the_two_sided_path(checker):
    reply = "离你不远呀"
    assert await g.guard_reply(agent(), reply) == (reply, False)
    checker.assert_not_awaited()
    checker.return_value = {"verdicts": [{"index": 0, "about": "user", "allowed": True}]}
    assert await g.guard_reply(agent(), reply, user_location=LocationContext()) == (reply, False)
    assert "对方位置：{\"known\": false}" in checker.await_args.args[1]


async def test_single_sided_callers_tell_the_checker_to_skip_the_user_rule(checker):
    checker.return_value = {"verdicts": [{"index": 0, "allowed": True}]}
    await g.verify_generated_locations(agent(), ["我住在杭州"], kind="memory")
    assert f"对方位置：{g.USER_LOCATION_UNCHECKED}" in checker.await_args.args[1]


@pytest.mark.parametrize("about", [None, "both", ["user"]])
async def test_unlabeled_rejection_keeps_the_self_sided_correction(checker, about):
    verdict = {"index": 0, "allowed": False}
    if about is not None:
        verdict["about"] = about
    checker.return_value = {"verdicts": [verdict]}
    text, corrected = await g.guard_reply(agent(), "我住镇江", user_location=xian())
    assert corrected and text == f"抱歉，刚才地点说错了。我住在{HANGZHOU}。"


async def test_verifier_result_stays_a_plain_list_for_existing_callers(checker):
    rejected = await g.verify_generated_locations(agent(), ["我住镇江"], kind="memory")
    assert rejected == [0] and rejected.subjects == {0: "user"}
    assert await g.verify_generated_locations(agent(), [], kind="memory") == []


async def test_checker_outage_fails_closed_on_the_side_that_was_risky(checker):
    checker.side_effect = RuntimeError("offline")
    text, corrected = await g.guard_reply(agent(), "离你不远呀", user_location=xian())
    assert corrected and text.startswith("刚才说岔了")
    text, corrected = await g.guard_reply(agent(), "我住镇江", user_location=xian())
    assert corrected and text.startswith("抱歉，刚才地点说错了")


@pytest.mark.parametrize("location,expected", [
    (xian(), "刚才说岔了，我在杭州，你在西安，我们不在一个城市哈"),
    (xian(user_city="杭州市"), "刚才说岔了，我们都在杭州"),
    (LocationContext(), "刚才说岔了，我在杭州，还不知道你在哪儿呢"),
    (None, "刚才说岔了，我在杭州，还不知道你在哪儿呢"),
])
def test_user_side_correction_only_states_known_facts(location, expected):
    assert g.user_location_correction(agent(), location) == expected


def test_user_side_correction_without_agent_city():
    assert g.user_location_correction(agent(None), None) == "刚才说岔了，还不知道你在哪儿呢"


# ── 两个执行器都接上 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["legacy", "langgraph"])
async def test_both_executors_use_one_location_context(monkeypatch, executor):
    from app.config import settings
    from app.services import persona_grounding
    from app.services.chat import orchestrator as chat
    from app.services.chat.data_fetch_phase import FetchedContext
    from app.services.chat.intent_dispatcher import IntentType
    from tests.g02_harness_support import configure_pair

    io = configure_pair(monkeypatch)
    monkeypatch.setattr(settings, "chat_executor", executor)
    chat.fetch_parallel_context.return_value = FetchedContext(memory_relevance="weak")
    monkeypatch.setattr(chat, "load_location_context", AsyncMock(return_value=xian()))
    verify = AsyncMock(return_value=[])
    monkeypatch.setattr(persona_grounding, "verify_generated_locations", verify)

    events = [event async for event in chat.stream_chat_response(
        "c-1", "我们在一个城市吗", io.agent, "u-1", forced_intent=IntentType.NONE,
        reply_context={"received_at": NOW.isoformat()},
    )]

    assert sum(e["event"] == "done" for e in events) == 1
    # 问到双方在哪: 不走 tier 轻量 prompt, 主 prompt 带「你们各自在哪」段
    assert not io.tier_calls and io.main_calls
    messages, kwargs = io.main_calls[0]
    assert "## 你们各自在哪" in json.dumps(messages, ensure_ascii=False)
    assert kwargs["web_search_location"] == xian().web_search_location()
    assert verify.await_args.kwargs["user_location"] == xian().verifier_facts()
    assert verify.await_args.kwargs["always"] is True


async def test_user_only_claim_without_agent_city_is_not_sent_to_the_checker(checker):
    assert await g.guard_reply(agent(None), "离你不远呀", user_location=xian()) == ("离你不远呀", False)
    checker.assert_not_awaited()


def test_unparseable_timestamp_types_are_unknown():
    assert lc._as_datetime(None) is None and lc._as_datetime(123) is None


@pytest.mark.parametrize("reply", [
    "对啊，转塘是杭州的呀！ 我就在杭州西湖区转塘街道呢。", "是啊 太巧了", "嗯嗯 我也在这边",
])
async def test_affirming_a_known_wrong_same_place_is_corrected_without_the_model(checker, reply):
    text, corrected = await g.guard_reply(agent(), reply, question="我们在一个城市吗",
                                          user_location=xian())
    assert corrected and text == "刚才说岔了，我在杭州，你在西安，我们不在一个城市哈"
    checker.assert_not_awaited()


@pytest.mark.parametrize("location,question,reply", [
    (xian(), "我们在一个城市吗", "不在哦 我在杭州 你那边是西安吧"),
    (xian(), "你在干嘛", "对啊 刚下班"),
    (xian(user_city="杭州市"), "我们在一个城市吗", "对啊 都在杭州"),
    (LocationContext(), "我们在一个城市吗", "对啊"),
    (None, "我们在一个城市吗", "对啊"),
])
def test_same_place_affirmation_needs_known_different_cities(location, question, reply):
    assert not lc.affirms_same_place_wrongly(location, question, reply)
