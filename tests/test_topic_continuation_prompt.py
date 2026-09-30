"""被动回复高级承接 (《主动聊天机制（新增）》+《主动交流提示词（新增）》).

承接短句 / 跳话题过渡句单独生成、作为独立气泡排在正常回复前; 主回复 prompt 只注入
「话题接续」段告诉模型别重复, 并压掉重逢感知短档。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat import topic_continuation as tcont
from app.services.chat.topic_continuation import (
    TopicContinuation,
    await_topic_continuation,
    build_topic_continuation,
)
from app.services.interaction.topic_continuity import (
    FOLLOWUP_TRIGGER_TYPE,
    ContinuationCue,
    ContinuityState,
    TopicVerdict,
)

UTC = timezone.utc


# ── prompt_builder: 「话题接续」段 ────────────────────────────────────

async def _prompt_text(key: str, **_kwargs) -> str:
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    definition = PROMPT_DEFINITION_MAP.get(key)
    return definition.default_text if definition else ""


async def _build(**kwargs) -> str:
    from app.services.chat.prompt_builder import build_system_prompt

    with (
        patch("app.services.chat.prompt_builder.get_prompt_text", AsyncMock(side_effect=_prompt_text)),
        patch(
            "app.services.chat.prompt_builder.get_prompt_text_or_default",
            AsyncMock(side_effect=_prompt_text),
        ),
    ):
        return await build_system_prompt(
            agent=SimpleNamespace(name="Hillow", values={"gender": "female"}),
            memory_relevance="weak",
            reply_count=1,
            reply_total=150,
            **kwargs,
        )


def _continuation(*lines: str, unfinished: bool = True) -> TopicContinuation:
    cue = ContinuationCue(unfinished=unfinished, return_line=bool(lines), gap_seconds=40 * 60)
    return TopicContinuation(cue, list(lines))


@pytest.mark.asyncio
async def test_sent_lines_are_listed_and_reengagement_suppressed():
    prompt = await _build(
        reengagement_gap_seconds=40 * 60,
        topic_continuation=_continuation("忙完啦～", "对了面试那事等会儿再聊"),
    )
    assert "## 话题接续" in prompt
    assert "「忙完啦～」\n「对了面试那事等会儿再聊」" in prompt
    assert "## 重逢感知" not in prompt  # 不叠两层"回来了怎么接"


@pytest.mark.asyncio
async def test_finished_topic_suppresses_reengagement_without_section():
    """spec: 上一轮已完结 → 无论隔多久都不带承接话术."""
    prompt = await _build(
        reengagement_gap_seconds=90 * 60,
        topic_continuation=_continuation(unfinished=False),
    )
    assert "## 话题接续" not in prompt
    assert "## 重逢感知" not in prompt


@pytest.mark.asyncio
async def test_unfinished_without_lines_keeps_reengagement():
    """承接句没生成出来 (失败 / 被口吻过滤): 至少让主回复知道隔了多久."""
    prompt = await _build(
        reengagement_gap_seconds=40 * 60,
        topic_continuation=_continuation(unfinished=True),
    )
    assert "## 重逢感知" in prompt
    assert "## 话题接续" not in prompt


@pytest.mark.asyncio
async def test_no_continuation_keeps_existing_reengagement():
    prompt = await _build(reengagement_gap_seconds=90 * 60)
    assert "## 重逢感知" in prompt
    assert "## 话题接续" not in prompt


@pytest.mark.asyncio
async def test_offering_turn_ignores_continuation():
    """红包/礼物本身就是新话题."""
    prompt = await _build(
        topic_continuation=_continuation("忙完啦～"),
        red_packet_context={"offering_id": "rp-1", "amount": 5},
    )
    assert "## 话题接续" not in prompt


@pytest.mark.asyncio
async def test_disabled_template_drops_section(monkeypatch):
    from app.services.chat import prompt_builder

    monkeypatch.setattr(prompt_builder, "_get_optional_prompt", AsyncMock(return_value=None))
    assert await prompt_builder._build_topic_continuation_section(_continuation("x")) is None


# ── build_topic_continuation ─────────────────────────────────────────

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


def _continuity(**kw) -> ContinuityState:
    base = dict(
        verdict=TopicVerdict("unfinished", "llm"),
        anchor_message_id="ai-1",
        followup_sent_at=None,
        last_user_at=None,
        session_closed=False,
    )
    base.update(kw)
    return ContinuityState(**base)  # type: ignore[arg-type]


def _msg(id_: str, role: str, content: str, minutes_ago: float, metadata=None):
    return SimpleNamespace(
        id=id_, role=role, content=content, metadata=metadata or {},
        createdAt=NOW - timedelta(minutes=minutes_ago),
    )


def _history():
    return [  # 时间正序, 同 orchestrator 的 recent_messages
        _msg("u-0", "user", "我明天面试好紧张", 30),
        _msg("g-1", "assistant", "黑棋落子", 29, {"kind": "game_status"}),
        _msg("ai-1", "assistant", "什么岗位呀？", 25),
        _msg("u-1", "user", "今天吃了火锅", 0),
    ]


class FakeLLM:
    """按 prompt key 返回; 记录每次调用的参数."""

    def __init__(self, **outputs):
        self.outputs = {
            "chat.topic_continuation_return": "忙完啦～",
            "chat.topic_jump_detect": "新话题",
            "chat.topic_continuation_jump": "面试的事等会儿再跟我说说",
            **outputs,
        }
        self.calls: dict[str, dict] = {}

    async def __call__(self, key, params, *, utility=False):
        self.calls[key] = {"params": params, "utility": utility}
        out = self.outputs[key]
        if isinstance(out, BaseException):
            raise out
        return out


@pytest.fixture
def llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(tcont, "_generate", fake)
    return fake


def _load(state):
    return AsyncMock(return_value=state)


async def _run(monkeypatch, *, continuity=None, previous=None, replied_minutes_ago=0.0, **kw):
    monkeypatch.setattr(tcont, "load_continuity", _load(continuity or _continuity()))
    history = _history()
    params = dict(
        conversation_id="c-1",
        previous_assistant=previous or history[2],
        replied_at=NOW - timedelta(minutes=replied_minutes_ago),
        user_message="今天吃了火锅",
        history=history,
        current_turn_ids={"u-1"},
        agent=SimpleNamespace(mbti=None, currentMbti=None),
        offering_turn=False,
        patience_low=False,
    )
    params.update(kw)
    return await build_topic_continuation(**params)


@pytest.mark.asyncio
async def test_return_line_then_transition_line(monkeypatch, llm):
    result = await _run(monkeypatch)
    assert result.lines == ["忙完啦～", "面试的事等会儿再跟我说说"]
    ret = llm.calls["chat.topic_continuation_return"]["params"]
    assert ret["user_msg"] == "今天吃了火锅"
    assert ret["ai_last_send_time"].endswith("19:35")  # UTC+8
    assert ret["current_time"].endswith("20:00")
    assert "25" in ret["time_gap"]
    # 上下文: 带时间戳, 不含本轮消息和游戏播报
    conversation = ret["conversation"]
    assert "[09-29 19:35] AI: 什么岗位呀？" in conversation
    assert "黑棋落子" not in conversation and "火锅" not in conversation
    # 跳话题判定用小模型; 两句生成用大模型
    assert llm.calls["chat.topic_jump_detect"]["utility"] is True
    assert llm.calls["chat.topic_continuation_jump"]["utility"] is False


@pytest.mark.asyncio
async def test_same_topic_gets_no_transition(monkeypatch, llm):
    llm.outputs["chat.topic_jump_detect"] = "接续"
    result = await _run(monkeypatch)
    assert result.lines == ["忙完啦～"]
    assert "chat.topic_continuation_jump" not in llm.calls


@pytest.mark.asyncio
async def test_quick_reply_only_considers_transition(monkeypatch, llm):
    """≤10 分钟回来: 不说"回来了", 只在跳话题时过渡."""
    history = _history()
    previous = SimpleNamespace(**{**vars(history[2]), "createdAt": NOW - timedelta(minutes=6)})
    result = await _run(monkeypatch, previous=previous)
    assert result.lines == ["面试的事等会儿再跟我说说"]
    assert "chat.topic_continuation_return" not in llm.calls


@pytest.mark.asyncio
async def test_gap_is_measured_to_user_reply_not_generation_time(monkeypatch, llm):
    """聚合/延迟队列下生成比用户发消息晚: 用户 8 分钟就回的不能算成 >10 分钟."""
    history = _history()
    previous = SimpleNamespace(**{**vars(history[2]), "createdAt": NOW - timedelta(minutes=12)})
    result = await _run(monkeypatch, previous=previous, replied_minutes_ago=4)
    assert result.cue.return_line is False
    assert 7 * 60 < result.cue.gap_seconds < 9 * 60


@pytest.mark.asyncio
async def test_followup_used_drops_return_line(monkeypatch, llm):
    result = await _run(monkeypatch, continuity=_continuity(followup_sent_at=NOW))
    assert result.lines == ["面试的事等会儿再跟我说说"]


@pytest.mark.asyncio
async def test_finished_topic_generates_nothing(monkeypatch, llm):
    result = await _run(monkeypatch, continuity=_continuity(verdict=TopicVerdict("finished", "llm")))
    assert result is not None and result.cue.unfinished is False and result.lines == []
    assert llm.calls == {}


@pytest.mark.asyncio
async def test_reply_to_followup_generates_nothing(monkeypatch, llm):
    history = _history()
    previous = SimpleNamespace(**{
        **vars(history[2]), "id": "b-1", "metadata": {"trigger_type": FOLLOWUP_TRIGGER_TYPE},
    })
    result = await _run(monkeypatch, previous=previous)
    assert result.cue.unfinished is False and result.lines == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kw", [{"offering_turn": True}, {"patience_low": True}])
async def test_offering_or_annoyed_is_not_applicable(monkeypatch, llm, kw):
    assert await _run(monkeypatch, **kw) is None
    assert llm.calls == {}


@pytest.mark.asyncio
async def test_unknown_continuity_keeps_old_behaviour(monkeypatch, llm):
    monkeypatch.setattr(tcont, "load_continuity", _load(None))  # Redis 挂了
    history = _history()
    result = await build_topic_continuation(
        conversation_id="c-1", previous_assistant=history[2], replied_at=NOW,
        user_message="x", history=history, current_turn_ids=set(), agent=None,
        offering_turn=False, patience_low=False,
    )
    assert result is None


@pytest.mark.asyncio
async def test_failed_or_needy_lines_are_dropped(monkeypatch, llm):
    llm.outputs["chat.topic_continuation_return"] = RuntimeError("llm down")
    llm.outputs["chat.topic_continuation_jump"] = "怎么才回呀"  # 提示词禁止的口吻
    result = await _run(monkeypatch)
    assert result.lines == []
    assert result.cue.unfinished is True


@pytest.mark.asyncio
async def test_welcome_back_line_may_ask_if_user_is_done_being_busy(monkeypatch, llm):
    """对方已经回来了: "忙完了吗" 是承接时间差, 不是查岗 (B 追问才拦这类口吻)."""
    llm.outputs["chat.topic_continuation_return"] = "忙完了吗？刚说到面试那儿"
    llm.outputs["chat.topic_jump_detect"] = "接续"
    result = await _run(monkeypatch)
    assert result.lines == ["忙完了吗？刚说到面试那儿"]


@pytest.mark.asyncio
async def test_lines_respect_single_bubble_length(monkeypatch, llm):
    from app.services.prompts.system_prompts import MAX_PER_REPLY

    llm.outputs["chat.topic_continuation_return"] = "忙完啦。" + "刚才说到的那件事我还一直惦记着呢" * 6
    llm.outputs["chat.topic_jump_detect"] = "接续"
    result = await _run(monkeypatch)
    assert result.lines and len(result.lines[0]) <= MAX_PER_REPLY


@pytest.mark.asyncio
async def test_generate_timeout_returns_none(monkeypatch):
    async def _slow(*_a, **_k):
        await asyncio.sleep(1)

    monkeypatch.setattr(tcont, "render_prompt", _slow)
    monkeypatch.setattr(tcont, "_LLM_TIMEOUT_S", 0.01)
    assert await tcont._generate("chat.topic_jump_detect", {}) is None


# ── await_topic_continuation ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_await_records_diagnostics():
    async def _ok():
        return _continuation("忙完啦～")

    diagnostics: dict = {}
    result = await await_topic_continuation(asyncio.ensure_future(_ok()), diagnostics)
    assert result.lines == ["忙完啦～"]
    assert diagnostics["topic_continuation"] == {"unfinished": True, "n_lines": 1}


@pytest.mark.asyncio
async def test_await_tolerates_missing_failed_and_cancelled_tasks():
    async def _boom():
        raise RuntimeError("redis down")

    assert await await_topic_continuation(None, {}) is None
    assert await await_topic_continuation(asyncio.ensure_future(_boom()), {}) is None

    cancelled = asyncio.ensure_future(asyncio.sleep(10))
    cancelled.cancel()
    await asyncio.sleep(0)
    assert await await_topic_continuation(cancelled, {}) is None


@pytest.mark.asyncio
async def test_await_gives_up_after_budget(monkeypatch):
    """承接句还没生成好就不等了 —— 不能把整轮回复拖慢."""
    monkeypatch.setattr(tcont, "_AWAIT_BUDGET_S", 0.01)
    slow = asyncio.ensure_future(asyncio.sleep(1))
    assert await await_topic_continuation(slow, {}) is None
    await asyncio.sleep(0)
    assert slow.cancelled()


def test_turn_started_at_takes_earliest_fragment():
    from app.services.chat.orchestrator import _turn_started_at

    early = datetime(2026, 9, 29, 4, 0, tzinfo=UTC)
    messages = [
        {"id": "old", "role": "user", "createdAt": (early - timedelta(hours=1)).isoformat()},
        {"id": "f1", "role": "user", "createdAt": early.isoformat()},
        {"id": "f2", "role": "user", "createdAt": (early + timedelta(seconds=3)).isoformat()},
    ]
    assert _turn_started_at(messages, {"f1", "f2"}) == early
    assert _turn_started_at(messages, set()) is None
