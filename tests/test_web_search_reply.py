"""Web search main-reply integration tests.

Covers:
- runtime config: web_search_enabled resolve chain + admin payload mapping
  (global-only, agent endpoints must not carry the column)
- ark_web_search service: payload shape, output extraction, usage recording,
  fail-open on every error class
- reply_generate branch: gate conditions + short-circuit before the
  streaming path, fallback to streaming when search yields nothing
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.admin.runtime_config import ConfigPayload, _payload_to_data, _row_to_payload
from app.config import settings
from app.services import runtime_config
from app.services.chat import reply_generate
from app.services.llm import ark_web_search


@pytest.fixture()
def loaded_caches(monkeypatch):
    monkeypatch.setattr(runtime_config, "_CACHE_LOADED", True)
    monkeypatch.setattr(runtime_config, "_AGENT_CACHE", {})
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {})
    return runtime_config


# ─── runtime config chain ─────────────────────────────────────────────────


def test_web_search_defaults_to_env_false(loaded_caches):
    resolved = runtime_config.resolve_config_sync(agent_id=None)
    assert resolved.web_search_enabled is settings.web_search_enabled is False


def test_web_search_enabled_via_system_config(monkeypatch, loaded_caches):
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {"webSearchEnabled": True})
    assert runtime_config.resolve_config_sync(agent_id=None).web_search_enabled is True


def test_payload_to_data_web_search_global_only():
    payload = ConfigPayload(web_search_enabled=True)
    global_data = _payload_to_data(payload, include_global_only=True)
    assert global_data["webSearchEnabled"] is True
    agent_data = _payload_to_data(payload)
    assert "webSearchEnabled" not in agent_data


def test_row_to_payload_web_search_getattr_safe():
    agent_row = SimpleNamespace(
        onlineModel=None, remoteProvider=None, remoteChatProvider=None,
        remoteSmallProvider=None, localChatModel=None, localSmallModel=None,
        remoteChatModel=None, remoteSmallModel=None,
    )
    assert _row_to_payload(agent_row)["web_search_enabled"] is None
    assert _row_to_payload(None)["web_search_enabled"] is None


# ─── ark_web_search service ───────────────────────────────────────────────


def test_to_responses_input_filters_bad_roles():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": "dropped"},
        {"role": "user", "content": None},
    ]
    out = ark_web_search._to_responses_input(messages)
    assert [m["role"] for m in out] == ["system", "assistant", "user"]


def test_extract_output_captures_queries_and_sources():
    """Queries/citations are what make a search turn debuggable in the panel."""
    payload = {
        "output": [
            {
                "type": "web_search_call", "status": "completed",
                "action": {"type": "search", "query": "2026年7月新上映电影"},
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "北京今天31℃",
                        "annotations": [
                            {
                                "type": "url_citation", "title": "猫眼电影",
                                "url": "https://maoyan.example/x",
                                "summary": "观众评分 9.7  八仙！\n评分 9.6 群星闪耀时",
                            },
                        ],
                    },
                ],
            },
        ],
    }
    result = ark_web_search._extract_output(payload)
    assert result.text == "北京今天31℃"
    assert result.search_calls == 1
    assert result.queries == ["2026年7月新上映电影"]
    assert result.sources == [{
        "title": "猫眼电影",
        "url": "https://maoyan.example/x",
        "summary": "观众评分 9.7 八仙！ 评分 9.6 群星闪耀时",
    }]


def test_extract_output_caps_sources():
    payload = {
        "output": [{
            "type": "message",
            "content": [{
                "type": "output_text", "text": "x",
                "annotations": [
                    {"title": f"t{i}", "url": f"u{i}", "summary": "s" * 400}
                    for i in range(20)
                ],
            }],
        }],
    }
    result = ark_web_search._extract_output(payload)
    assert len(result.sources) == ark_web_search._MAX_TRACED_SOURCES
    assert all(len(s["summary"]) <= 120 for s in result.sources)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = "err"

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.captured: dict = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, endpoint, headers=None, json=None):
        self.captured["endpoint"] = endpoint
        self.captured["json"] = json
        if self.error:
            raise self.error
        return self.response


@pytest.mark.asyncio
async def test_generate_with_web_search_success(monkeypatch):
    payload = {
        "output": [
            {"type": "web_search_call"},
            {"type": "message", "content": [{"type": "output_text", "text": "回复"}]},
        ],
        "usage": {
            "input_tokens": 6000, "output_tokens": 100,
            "input_tokens_details": {"cached_tokens": 500},
        },
    }
    client = _FakeClient(response=_FakeResponse(200, payload))
    monkeypatch.setattr(ark_web_search.httpx, "AsyncClient", lambda **kw: client)
    monkeypatch.setattr(ark_web_search.settings, "ark_api_key", "test-key")
    recorded: dict = {}
    monkeypatch.setattr(
        ark_web_search.usage_tracker, "record",
        lambda model, i, o, cached_input_tokens=0: recorded.update(
            {"model": model, "in": i, "out": o, "cached": cached_input_tokens},
        ),
    )

    text = await ark_web_search.generate_with_web_search(
        [{"role": "user", "content": "明天天气"}], model="doubao-seed-character-260628",
    )

    assert text == "回复"
    body = client.captured["json"]
    assert body["model"] == "doubao-seed-character-260628"
    assert body["tools"] == [{"type": "web_search"}]
    # Forced: with "auto" the character model never calls the tool under the
    # production prompt (0/16 measured), it just says "我帮你查下".
    assert body["tool_choice"] == "required"
    assert body["stream"] is False
    assert client.captured["endpoint"].endswith("/responses")
    assert recorded == {
        "model": "ark/doubao-seed-character-260628",
        "in": 6000, "out": 100, "cached": 500,
    }


@pytest.mark.asyncio
async def test_generate_with_web_search_records_trace_step(monkeypatch):
    """Raw HTTP calls bypass langchain, so the step must be recorded manually
    or the main reply disappears from the trace tree."""
    payload = {
        "output": [
            {"type": "web_search_call", "action": {"query": "北京天气"}},
            {
                "type": "message",
                "content": [{
                    "type": "output_text", "text": "北京31℃",
                    "annotations": [{"title": "中国气象局", "url": "https://cma.example"}],
                }],
            },
        ],
        "usage": {
            "input_tokens": 4830, "output_tokens": 29,
            "input_tokens_details": {"cached_tokens": 0},
        },
    }
    client = _FakeClient(response=_FakeResponse(200, payload))
    monkeypatch.setattr(ark_web_search.httpx, "AsyncClient", lambda **kw: client)
    monkeypatch.setattr(ark_web_search.settings, "ark_api_key", "test-key")
    monkeypatch.setattr(ark_web_search.usage_tracker, "record", lambda *a, **k: None)
    recorded: dict = {}
    monkeypatch.setattr(
        ark_web_search, "record_manual_llm_run",
        lambda **kwargs: recorded.update(kwargs),
    )

    messages = [
        {"role": "system", "content": "人设 prompt"},
        {"role": "user", "content": "北京天气"},
    ]
    await ark_web_search.generate_with_web_search(messages, model="doubao-x")

    assert recorded["model_name"] == "doubao-x"
    assert recorded["provider"] == "ark"
    assert recorded["messages"] == messages
    assert recorded["output_text"] == "北京31℃"
    assert recorded["input_tokens"] == 4830
    assert recorded["output_tokens"] == 29
    assert recorded["cached_input_tokens"] == 0
    assert recorded["metadata"] == {
        "web_search_calls": 1, "web_search_queries": ["北京天气"],
    }
    # Queries + citations land in the trace outputs so the panel can show what
    # the search actually returned (previously only the reply text was kept).
    search_block = recorded["extra_outputs"]["web_search"]
    assert search_block["calls"] == 1
    assert search_block["queries"] == ["北京天气"]
    assert search_block["sources"][0]["title"] == "中国气象局"
    assert recorded["ended_at"] >= recorded["started_at"]


@pytest.mark.asyncio
async def test_generate_with_web_search_fail_open(monkeypatch):
    monkeypatch.setattr(ark_web_search.settings, "ark_api_key", "test-key")

    # HTTP error status → None
    client = _FakeClient(response=_FakeResponse(404))
    monkeypatch.setattr(ark_web_search.httpx, "AsyncClient", lambda **kw: client)
    assert await ark_web_search.generate_with_web_search(
        [{"role": "user", "content": "hi"}], model="m",
    ) is None

    # Transport exception → None
    client = _FakeClient(error=RuntimeError("timeout"))
    monkeypatch.setattr(ark_web_search.httpx, "AsyncClient", lambda **kw: client)
    assert await ark_web_search.generate_with_web_search(
        [{"role": "user", "content": "hi"}], model="m",
    ) is None

    # Empty output → None
    client = _FakeClient(response=_FakeResponse(200, {"output": []}))
    monkeypatch.setattr(ark_web_search.httpx, "AsyncClient", lambda **kw: client)
    assert await ark_web_search.generate_with_web_search(
        [{"role": "user", "content": "hi"}], model="m",
    ) is None


@pytest.mark.asyncio
async def test_generate_with_web_search_requires_key_and_model(monkeypatch):
    monkeypatch.setattr(ark_web_search.settings, "ark_api_key", "")
    assert await ark_web_search.generate_with_web_search(
        [{"role": "user", "content": "hi"}], model="m",
    ) is None
    monkeypatch.setattr(ark_web_search.settings, "ark_api_key", "k")
    assert await ark_web_search.generate_with_web_search(
        [{"role": "user", "content": "hi"}], model="",
    ) is None


# ─── 「联网结果使用」prompt section ─────────────────────────────────────────


def test_web_search_usage_prompt_covers_observed_failures():
    """2026-07-25 生产 trace: 搜索结果压过历史与记忆, 复现三个毛病."""
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    text = PROMPT_DEFINITION_MAP["chat.web_search_usage"].default_text
    assert "已经聊过的内容不要当成新发现再讲一遍" in text  # 重复端出刚聊过的片子
    assert "我刚搜了下" in text                            # 每轮宣告的播报腔
    assert "先用你自己的人设和记忆回答" in text            # 问偏好答成榜单播报
    # 联网结果的合法性授权必须留在**受开关控制**的这一段里 — 放进每轮都注入的
    # 反幻觉规则会让开关关闭时 AI 仍答应"我帮你查下"却查不了 (空头承诺).
    assert "反幻觉硬约束" in text
    assert "私人过往" in text
    # 无占位符 — 整段静态注入, 渲染时不需要任何变量
    assert "{" not in text


def test_web_search_usage_prompt_forbids_bare_facts():
    """2026-07-25 生产 trace: 「永乐宫建于哪一年」→「它是元代1247年动工的哦👌」.

    人设、字数上限、反播报规则全都没拦住 —— 它们管的是语气和长度, 没有一条要求
    事实必须挂在她自己的反应上. 一个只报数字的回复在这些约束下完全合法, 但读起来
    就是查询工具. 同一条 trace 里她还漏答了「有怎样的历史故事」, 只捡了好查的那半.
    """
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    text = PROMPT_DEFINITION_MAP["chat.web_search_usage"].default_text
    assert "事实不能光秃秃地丢出来" in text
    assert "你的反应、看法、好奇" in text
    assert "都接住" in text  # 多问一次答一半 = 在交检索结果


def test_web_search_decision_prompt_excludes_static_lookups():
    """静态百科考问不该走联网.

    「建于哪一年」查得到, 但查到之后她只会报一个数字 —— 联网把一次闲聊变成一次
    检索. 朋友答不上来说不知道反而更自然, 所以这类问题宁可不查.
    原【要点】的兜底「拿不准就查」会把它们全兜进来, 必须给出反向边界.
    """
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    text = PROMPT_DEFINITION_MAP["chat.web_search_decision"].default_text
    assert "建于哪一年" in text
    assert "静态数字或名词答案" in text
    # 反向边界必须在兜底规则**之后**出现, 否则读起来仍是"拿不准就查"收尾.
    assert text.index("静态数字或名词答案") > text.index("拿不准时")
    # 近期动态不能被误伤 —— 它跟静态考问是这段 prompt 要分开的两件事.
    assert "最近上了什么片" in text


def test_web_search_authorisation_is_not_in_always_on_prompt():
    """守卫: 联网授权不得回流到无条件注入的反幻觉硬约束里."""
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    always_on = PROMPT_DEFINITION_MAP["chat.anti_hallucination_hard_rule"].default_text
    assert "联网" not in always_on


def test_extract_discussed_titles_from_recent_turns():
    from app.services.llm.web_search_gate import extract_discussed_titles

    messages = [
        {"role": "user", "content": "八仙看过没"},
        {"role": "assistant", "content": "最近上映的《八仙》票房还挺高的"},
        {"role": "assistant", "content": "我还挺期待《长安三万里》的"},
        {"role": "user", "content": "这个近期上映吗"},
    ]
    # newest first, deduped
    assert extract_discussed_titles(messages) == ["长安三万里", "八仙"]
    # a title the user names right now is what they want to hear about
    assert extract_discussed_titles(
        messages, current_message="《八仙》好看吗",
    ) == ["长安三万里"]
    assert extract_discussed_titles([]) == []


def test_extract_discussed_titles_caps_and_ignores_non_text():
    from app.services.llm import web_search_gate

    messages = [{"role": "user", "content": f"《片{i}》"} for i in range(20)]
    messages.append({"role": "user", "content": None})
    titles = web_search_gate.extract_discussed_titles(messages)
    assert len(titles) == web_search_gate._MAX_TITLES


@pytest.mark.asyncio
async def test_recent_titles_section_lists_discussed_works():
    from unittest.mock import patch

    from app.services.chat.prompt_builder import build_system_prompt
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    async def _prompt_text(key: str, **_kwargs) -> str:
        definition = PROMPT_DEFINITION_MAP.get(key)
        return definition.default_text if definition else ""

    with (
        patch(
            "app.services.chat.prompt_builder.get_prompt_text",
            AsyncMock(side_effect=_prompt_text),
        ),
        patch(
            "app.services.chat.prompt_builder.get_prompt_text_or_default",
            AsyncMock(side_effect=_prompt_text),
        ),
    ):
        prompt = await build_system_prompt(
            agent=SimpleNamespace(name="小伴", values={"gender": "female"}),
            needs_web_search=True,
            discussed_titles=["八仙", "长安三万里"],
        )
    assert "## 刚聊过的作品" in prompt
    assert "《八仙》、《长安三万里》" in prompt


@pytest.mark.asyncio
async def test_usage_section_injected_only_on_search_turns():
    from unittest.mock import patch

    from app.services.chat.prompt_builder import build_system_prompt
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    async def _prompt_text(key: str, **_kwargs) -> str:
        definition = PROMPT_DEFINITION_MAP.get(key)
        return definition.default_text if definition else ""

    async def build(needs_web_search: bool) -> tuple[str, dict]:
        diagnostics: dict = {}
        with (
            patch(
                "app.services.chat.prompt_builder.get_prompt_text",
                AsyncMock(side_effect=_prompt_text),
            ),
            patch(
                "app.services.chat.prompt_builder.get_prompt_text_or_default",
                AsyncMock(side_effect=_prompt_text),
            ),
        ):
            prompt = await build_system_prompt(
                agent=SimpleNamespace(name="小伴", values={"gender": "female"}),
                needs_web_search=needs_web_search,
                diagnostics=diagnostics,
            )
        return prompt, diagnostics

    with_search, _ = await build(True)
    without_search, skipped_diag = await build(False)
    assert "## 联网结果使用" in with_search
    assert "已经聊过的内容不要当成新发现再讲一遍" in with_search
    assert "## 联网结果使用" not in without_search
    assert "联网结果使用" in skipped_diag["empty_prompt_sections_removed"]


# ─── reply_generate branch ────────────────────────────────────────────────


def _resolved(web_search=True, online=True, provider="ark"):
    return runtime_config.ResolvedConfig(
        online_model=online,
        remote_provider=provider,
        remote_chat_provider=provider,
        remote_small_provider=provider,
        local_chat_model="qwen2.5:14b",
        local_small_model="qwen2.5:7b",
        remote_chat_model="doubao-seed-character-260628",
        remote_small_model="doubao-seed-character-260628",
        vision_model="v",
        asr_model="a",
        web_search_enabled=web_search,
    )


@pytest.mark.asyncio
async def test_try_web_search_reply_uses_resolved_model(monkeypatch):
    """Gating moved to web_search_gate; this helper only issues the request."""
    calls: list = []

    async def fake_generate(messages, *, model):
        calls.append(model)
        return "搜到了"

    monkeypatch.setattr(
        "app.services.llm.ark_web_search.generate_with_web_search", fake_generate,
    )
    monkeypatch.setattr(
        "app.services.runtime_config.resolve_for_current", lambda: _resolved(),
    )
    result = await reply_generate._try_web_search_reply(
        [{"role": "user", "content": "x"}],
    )
    assert result == "搜到了"
    assert calls == ["doubao-seed-character-260628"]


# ─── gate: length floor + classifier + route check ────────────────────────


def test_length_floor_only_skips_chitchat_acks():
    """Keyword prefiltering was removed: proper nouns ("八仙看过没") match no
    topic keyword yet are exactly what needs looking up."""
    from app.services.llm.web_search_gate import is_worth_classifying

    for kept in ("八仙看过没", "最近看新电影了没", "听过告五人的歌没",
                 "你今天过得怎么样呀", "明天北京天气怎么样",
                 # 省略式追问: 短, 但话题继承自上文 — 多轮测试里「还有吗」被
                 # 4 字下限挡掉后, 模型凭参数记忆编了一部电影出来.
                 "还有吗", "然后呢", "真的吗"):
        assert is_worth_classifying(kept), kept
    for skipped in ("嗯", "好的", "在吗", "  ", ""):
        assert not is_worth_classifying(skipped), skipped


@pytest.mark.asyncio
async def test_gate_classifier_parses_verdicts(monkeypatch):
    from app.services.llm import web_search_gate

    async def reply(verdict):
        async def fake_render(key, params, invoke):
            assert key == "chat.web_search_decision"
            return verdict
        monkeypatch.setattr(web_search_gate, "render_prompt", fake_render)
        return await web_search_gate.needs_web_search("最近有什么新电影", context="c")

    assert await reply("需要联网") is True
    assert await reply("  需要联网。 ") is True
    assert await reply("不需要联网") is False
    assert await reply("") is False


@pytest.mark.asyncio
async def test_gate_classifier_fails_closed(monkeypatch):
    from app.services.llm import web_search_gate

    async def boom(*args, **kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(web_search_gate, "render_prompt", boom)
    assert await web_search_gate.needs_web_search("金价多少") is False


@pytest.mark.asyncio
async def test_decide_web_search_skips_llm_for_short_acks(monkeypatch):
    from app.services.chat import data_fetch_phase

    called = False

    async def fake_needs(*args, **kwargs):
        nonlocal called
        called = True
        return True

    monkeypatch.setattr(
        "app.services.llm.web_search_gate.needs_web_search", fake_needs,
    )
    monkeypatch.setattr(
        "app.services.runtime_config.resolve_for_current", lambda: _resolved(),
    )
    assert await data_fetch_phase._decide_web_search("嗯", "") is False
    assert called is False


@pytest.mark.asyncio
async def test_decide_web_search_classifies_proper_nouns(monkeypatch):
    """Regression: "八仙看过没" used to be dropped by the keyword prefilter."""
    from app.services.chat import data_fetch_phase

    seen: list[str] = []

    async def fake_needs(message, context=""):
        seen.append(message)
        return True

    monkeypatch.setattr(
        "app.services.llm.web_search_gate.needs_web_search", fake_needs,
    )
    monkeypatch.setattr(
        "app.services.runtime_config.resolve_for_current", lambda: _resolved(),
    )
    assert await data_fetch_phase._decide_web_search("八仙看过没", "") is True
    assert seen == ["八仙看过没"]


@pytest.mark.asyncio
async def test_decide_web_search_skips_llm_when_route_unavailable(monkeypatch):
    from app.services.chat import data_fetch_phase

    called = False

    async def fake_needs(*args, **kwargs):
        nonlocal called
        called = True
        return True

    monkeypatch.setattr(
        "app.services.llm.web_search_gate.needs_web_search", fake_needs,
    )
    for resolved in (
        _resolved(web_search=False),          # switch off
        _resolved(provider="deepseek"),       # non-ark chat route
        _resolved(online=False),              # local ollama mode
    ):
        monkeypatch.setattr(
            "app.services.runtime_config.resolve_for_current", lambda r=resolved: r,
        )
        assert await data_fetch_phase._decide_web_search("明天天气", "") is False
    assert called is False


@pytest.mark.asyncio
async def test_decide_web_search_confirms_with_classifier(monkeypatch):
    from app.services.chat import data_fetch_phase

    async def fake_needs(message, context=""):
        return "电影" in message

    monkeypatch.setattr(
        "app.services.llm.web_search_gate.needs_web_search", fake_needs,
    )
    monkeypatch.setattr(
        "app.services.runtime_config.resolve_for_current", lambda: _resolved(),
    )
    assert await data_fetch_phase._decide_web_search("最近有什么新电影", "") is True
    # classifier is the only semantic gate now — its "no" is final
    assert await data_fetch_phase._decide_web_search("今天心情不错呀", "") is False


@pytest.mark.asyncio
async def test_run_main_llm_short_circuits_on_web_search(monkeypatch):
    monkeypatch.setattr(
        reply_generate, "_try_web_search_reply",
        AsyncMock(return_value="联网回复[EMO:高兴/60]"),
    )

    def _boom():
        raise AssertionError("streaming path must not run when web search hits")

    monkeypatch.setattr(reply_generate, "get_chat_model", _boom)

    text, is_fallback = await reply_generate._run_main_llm(
        [{"role": "user", "content": "明天天气"}], needs_web_search=True,
    )
    assert text == "联网回复[EMO:高兴/60]"
    assert is_fallback is False


@pytest.mark.asyncio
async def test_run_main_llm_skips_web_search_when_gate_says_no(monkeypatch):
    """Gate off → no Responses API call at all (no tool tokens, keeps stream)."""
    called = False

    async def should_not_run(_messages):
        nonlocal called
        called = True
        return "不该走这里"

    monkeypatch.setattr(reply_generate, "_try_web_search_reply", should_not_run)
    fake_model = SimpleNamespace(astream=lambda msgs: None)
    monkeypatch.setattr(reply_generate, "get_chat_model", lambda: fake_model)
    monkeypatch.setattr(reply_generate, "get_fallback_chat_model", lambda: fake_model)
    monkeypatch.setattr(reply_generate, "provider_name", lambda m: "ark")
    monkeypatch.setattr(
        "app.services.llm.models._resolve_usage_model_key", lambda m: "ark/x",
    )

    async def fake_collect(*args, **kwargs):
        return "流式回复"

    monkeypatch.setattr(reply_generate, "collect_stream", fake_collect)

    text, _ = await reply_generate._run_main_llm([{"role": "user", "content": "hi"}])
    assert text == "流式回复"
    assert called is False


@pytest.mark.asyncio
async def test_run_main_llm_falls_back_to_stream_when_search_empty(monkeypatch):
    monkeypatch.setattr(
        reply_generate, "_try_web_search_reply", AsyncMock(return_value=None),
    )
    fake_model = SimpleNamespace(astream=lambda msgs: None)
    monkeypatch.setattr(reply_generate, "get_chat_model", lambda: fake_model)
    monkeypatch.setattr(reply_generate, "get_fallback_chat_model", lambda: fake_model)
    monkeypatch.setattr(reply_generate, "provider_name", lambda m: "ark")
    monkeypatch.setattr(
        "app.services.llm.models._resolve_usage_model_key", lambda m: "ark/x",
    )

    async def fake_collect(*args, **kwargs):
        return "流式回复"

    monkeypatch.setattr(reply_generate, "collect_stream", fake_collect)

    text, is_fallback = await reply_generate._run_main_llm(
        [{"role": "user", "content": "hi"}],
    )
    assert text == "流式回复"
    assert is_fallback is False
