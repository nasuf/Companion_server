"""特殊日期祝福的真实发送函数体 (之前只有把整个函数 mock 掉的 admin 测试).

回归: 09-14 起 maybe_prepare_proactive_link_recommendation 改为返回 (卡片, 原因)
元组, 这里没跟上, 每次祝福都在 .component_card 上崩溃 —— LLM 已调用, 消息永远
发不出去, 而测试全绿。
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.proactive import special_dates as sd


@pytest.fixture
def send_env(monkeypatch):
    monkeypatch.setattr(sd, "can_send_proactive", AsyncMock(return_value=True))
    monkeypatch.setattr(sd, "ensure_proactive_state_for_workspace", AsyncMock(return_value=None))
    monkeypatch.setattr(
        sd, "get_active_workspace_context", AsyncMock(return_value={"conversation_id": "c1"}),
    )
    monkeypatch.setattr(
        sd, "db",
        SimpleNamespace(aiagent=SimpleNamespace(find_unique=AsyncMock(
            return_value=SimpleNamespace(name="小伴", mbti=None, currentMbti=None),
        ))),
    )
    monkeypatch.setattr(sd, "_pick_prompt_key_and_fields", AsyncMock(return_value=("k", {})))
    monkeypatch.setattr(sd, "get_prompt_text", AsyncMock(return_value="hello"))
    monkeypatch.setattr(sd, "render_template", lambda *a, **k: "prompt")
    monkeypatch.setattr(
        "app.services.proactive.trending_context.resolve_trending_context",
        AsyncMock(return_value=("", False, None)),
    )
    monkeypatch.setattr("app.services.runtime_config.ensure_loaded", AsyncMock())

    @contextlib.asynccontextmanager
    async def _session(**_kwargs):
        yield SimpleNamespace(safe_trace_id=None)

    monkeypatch.setattr("app.services.llm.usage_tracker.traced_usage_session", _session)
    monkeypatch.setattr(sd, "get_chat_model", lambda: None)
    monkeypatch.setattr(sd, "invoke_text", AsyncMock(return_value="生日快乐呀今天"))
    link = AsyncMock(return_value=(None, "no_preselected_no_force"))
    monkeypatch.setattr("app.services.chat_links.maybe_prepare_proactive_link_recommendation", link)
    env = SimpleNamespace(
        emit=AsyncMock(return_value="m1"),
        count=AsyncMock(),
        close=AsyncMock(),
        link=link,
    )
    monkeypatch.setattr(sd, "emit_proactive_message", env.emit)
    monkeypatch.setattr(sd, "increment_proactive_count", env.count)
    monkeypatch.setattr("app.services.interaction.topic_continuity.close_session", env.close)
    return env


async def _send():
    return await sd.send_special_date_proactive(
        agent_id="a1", user_id="u1", workspace_id="w1",
        occasions=[sd.Occasion(type="birthday", name="生日", owner="user")],
    )


async def test_birthday_greeting_is_sent(send_env):
    assert await _send() is True
    kwargs = send_env.emit.await_args.kwargs
    assert kwargs["trigger_type"] == "special_date"
    assert kwargs["abort_if_user_replied_since"] is not None
    assert "component_card" not in kwargs["extra_metadata"]
    send_env.count.assert_awaited_once()
    # 祝福和 A 模式开场一样是新话题 → 下一条用户消息开新会话
    send_env.close.assert_awaited_once_with("c1")


async def test_link_card_is_attached_when_recommended(send_env, monkeypatch):
    link = SimpleNamespace(
        component_card={"type": "link"}, link_card_metadata={"url": "u"},
        link=SimpleNamespace(id="lk"),
    )
    send_env.link.return_value = (link, None)
    bind = AsyncMock()
    monkeypatch.setattr("app.services.chat_links.bind_link_card_to_message", bind)
    assert await _send() is True
    assert send_env.emit.await_args.kwargs["extra_metadata"]["component_card"] == {"type": "link"}
    bind.assert_awaited_once()


async def test_user_back_during_generation_aborts_without_counting(send_env):
    send_env.emit.return_value = ""
    assert await _send() is False
    send_env.count.assert_not_awaited()
    send_env.close.assert_not_awaited()
