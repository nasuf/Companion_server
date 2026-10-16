"""Agent nickname: L1 姓名 memory text, persona init, first greeting, backfill."""

from __future__ import annotations

import contextlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.life_story import convert_profile_to_memories, name_memory_content
from app.services.prompting.defaults import PROACTIVE_FIRST_GREETING_PROMPT


def _name_memories(profile: dict) -> list[str]:
    return [m["content"] for m in convert_profile_to_memories(profile, None)
            if m.get("sub_category") == "姓名"]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"nickname": "阿砚"}, "我叫陈砚，你可以叫我阿砚"),
        ({"nickname": None}, "我叫陈砚"),
        ({"nickname": "陈砚"}, "我叫陈砚"),   # 昵称与名字相同不重复
        ({"nickname": "  "}, "我叫陈砚"),
        ({"nickname": "阿砚", "detail": "小名：砚砚"}, "我叫陈砚，你可以叫我阿砚。小名：砚砚"),
        ({"detail": "小名：砚砚"}, "我叫陈砚。小名：砚砚"),  # 原有 name_detail 行为不变
    ],
)
def test_name_memory_content(kwargs, expected):
    assert name_memory_content("陈砚", **kwargs) == expected


def test_profile_nickname_lands_in_singleton_name_memory():
    assert _name_memories({"identity": {"name": "陈砚", "nickname": "阿砚"}}) == [
        "我叫陈砚，你可以叫我阿砚",
    ]
    assert _name_memories({"identity": {"name": "小伴"}}) == ["我叫小伴"]


def test_first_greeting_prompt_asks_for_nickname_intro():
    rendered = PROACTIVE_FIRST_GREETING_PROMPT.format(
        ai_name="陈砚", ai_nickname="阿砚", personality_brief="温和", occupation="插画师",
    )
    assert "AI昵称：阿砚" in rendered
    assert "你可以叫我阿砚" in rendered
    assert "{" not in rendered


def _backfill_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "backfill_agent_nickname_memories.py"
    spec = importlib.util.spec_from_file_location("backfill_agent_nickname_memories", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("content", "status", "after"),
    [
        ("我叫陈砚", "update", "我叫陈砚，你可以叫我阿砚"),
        ("我叫陈砚。小名：砚砚", "update", "我叫陈砚，你可以叫我阿砚。小名：砚砚"),
        ("我叫陈砚，你可以叫我阿砚", "done", None),          # 幂等
        ("我叫吴时远", "skip_unrecognized", None),            # 改过名, 不猜
        ("大家都叫我陈砚", "skip_unrecognized", None),        # 人工编辑过的形状
        ("我叫陈砚哦", "skip_unrecognized", None),
    ],
)
def test_backfill_plan_content(content, status, after):
    assert _backfill_module().plan_content(content, "陈砚", "阿砚") == (status, after)


def test_backfill_output_matches_fresh_init_text():
    # 回填与新建走同一个文案函数, 新旧 agent 的姓名记忆逐字一致。
    _, after = _backfill_module().plan_content("我叫陈砚", "陈砚", "阿砚")
    assert after == _name_memories({"identity": {"name": "陈砚", "nickname": "阿砚"}})[0]


def test_clone_copies_template_nickname(monkeypatch):
    from app.services.agent_template import clone as agent_clone

    monkeypatch.setattr(agent_clone, "pick_agent_avatar",
                        lambda gender: SimpleNamespace(key="k", url="/u"))
    template = SimpleNamespace(
        name="陈砚", nickname="阿砚", background=None, lifeOverview=None, age=None,
        occupation=None, city=None, gender="male", mbti=None, currentMbti=None, values=None,
    )
    payload = agent_clone._clone_persona_data(template, "user-id")
    assert (payload["name"], payload["nickname"]) == ("陈砚", "阿砚")


@pytest.mark.parametrize("imported", [False, True])
async def test_initialization_freezes_nickname_into_identity(monkeypatch, imported):
    from app.services import agent_initialization as init

    agent = SimpleNamespace(id="a1", userId="u1", name="陈砚", nickname=" 阿砚 ", gender="male")
    profile = {"identity": {"name": "陈砚"}}
    monkeypatch.setattr(init, "db", SimpleNamespace(aiagent=SimpleNamespace(update=AsyncMock())))
    for name in ("set_progress", "activate_agent", "dispatch_first_greeting_for_agent",
                 "generate_daily_schedule"):
        monkeypatch.setattr(init, name, AsyncMock())
    monkeypatch.setattr(init, "seven_dim_to_mbti", AsyncMock(return_value={"EI": 50, "summary": ""}))
    monkeypatch.setattr(init, "build_mbti", AsyncMock(return_value={"EI": 50}))
    from app.services.speech_output import voices
    monkeypatch.setattr(voices, "ensure_agent_voice", AsyncMock())
    monkeypatch.setattr(init, "pick_random_active_career", AsyncMock(return_value=None))
    monkeypatch.setattr(init, "generate_full_profile", AsyncMock(return_value=profile))
    monkeypatch.setattr(init, "_apply_postprocess_overrides", lambda value, **kwargs: value)
    monkeypatch.setattr(init, "_safe_life_overview", AsyncMock(return_value=""))
    captured = []

    async def memories(**kwargs):
        snapshot = json.loads(kwargs["origin"].payload)
        captured.append(snapshot)
        # 冻结的人设来源里要能找到昵称, 姓名记忆才有出处。
        assert snapshot["profile"]["identity"]["nickname"] == "阿砚"
        assert snapshot["invocation_inputs"]["nickname"] == "阿砚"
        return 1

    monkeypatch.setattr(init, "generate_l1_coverage", memories)
    await init._run_agent_initialization_inner(
        agent, "u1", "ws1", {"lively": 50},
        profile_override={"identity": {"name": "陈砚"}} if imported else None,
    )
    assert len(captured) == 1


async def test_first_greeting_renders_nickname(monkeypatch):
    from app.services.admin_takeover import service as takeover
    from app.services import runtime_config
    from app.services.llm import usage_tracker
    from app.services.proactive import sender

    agent = SimpleNamespace(id="a1", name="陈砚", nickname="阿砚", status="active",
                            occupation="插画师")
    monkeypatch.setattr(takeover, "is_conversation_taken_over", AsyncMock(return_value=False))
    monkeypatch.setattr(runtime_config, "bind_agent_context", AsyncMock())
    monkeypatch.setattr(sender, "db", SimpleNamespace(
        message=SimpleNamespace(count=AsyncMock(return_value=0)),
        aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=agent)),
    ))
    redis = SimpleNamespace(set=AsyncMock(return_value=True), delete=AsyncMock())
    monkeypatch.setattr(sender, "get_redis", AsyncMock(return_value=redis))

    @contextlib.asynccontextmanager
    async def session(**kwargs):
        yield SimpleNamespace(safe_trace_id=None)

    monkeypatch.setattr(usage_tracker, "traced_usage_session", session)
    monkeypatch.setattr(sender, "get_prompt_text",
                        AsyncMock(return_value=PROACTIVE_FIRST_GREETING_PROMPT))
    monkeypatch.setattr(sender, "build_personality_brief", lambda agent: "温和")
    monkeypatch.setattr(sender, "get_chat_model", lambda: None)
    prompts = []

    async def invoke(model, prompt):
        prompts.append(prompt)
        return ""  # 生成为空 → 不落库, 只看渲染

    monkeypatch.setattr(sender, "invoke_text", invoke)
    assert await sender.send_first_greeting(
        conversation_id="c1", user_id="u1", agent_id="a1") is False
    assert "AI昵称：阿砚" in prompts[0] and "你可以叫我阿砚" in prompts[0]

    agent.nickname = None  # 姓名库外的名字: 用全名兜底
    prompts.clear()
    await sender.send_first_greeting(conversation_id="c1", user_id="u1", agent_id="a1")
    assert "AI昵称：陈砚" in prompts[0]


async def test_auth_response_carries_agent_nickname(monkeypatch):
    from app.api.public import auth as auth_api
    import app.services.phone_auth as phone_auth_mod

    user = SimpleNamespace(id="u1", username="ph_x", role="user")
    agent = SimpleNamespace(id="a1", name="陈砚", nickname="阿砚", avatarKey=None,
                            avatarUrl=None, city=None)
    monkeypatch.setattr(auth_api, "get_active_workspace",
                        AsyncMock(return_value=SimpleNamespace(id="w1", agentId="a1")))
    monkeypatch.setattr(auth_api, "db", SimpleNamespace(
        aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=agent)),
        conversation=SimpleNamespace(find_first=AsyncMock(return_value=None)),
    ))
    monkeypatch.setattr(auth_api, "resolve_display_identity", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(phone_auth_mod, "get_identity_summary", AsyncMock(return_value=(None, False)))

    response = await auth_api._build_auth_response(user, "jwt")
    assert (response.agent_name, response.agent_nickname) == ("陈砚", "阿砚")

    agent.nickname = None
    response = await auth_api._build_auth_response(user, "jwt")
    assert response.agent_nickname is None and response.agent_name == "陈砚"


async def test_chat_identity_anchor_includes_nickname():
    from app.services.persona_grounding import canonical_facts, grounding_context

    agent = SimpleNamespace(name="陈砚", nickname="阿砚", city="杭州", occupation="插画师", age=26)
    assert canonical_facts(agent)["nickname"] == "阿砚"
    text = await grounding_context(agent, template="档案：{facts}")
    assert '"nickname": "阿砚"' in text

    agent.nickname = None  # 没有昵称就不出现该键, 不注入空值
    assert "nickname" not in canonical_facts(agent)
