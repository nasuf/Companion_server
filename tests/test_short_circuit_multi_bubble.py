"""短路回复里模型写的 || 必须切成多条气泡, 不能只留首段.

生产复现 (会话 4de5372b, 2026-10-14 23:40): intent.current_state_reply 输出
「刚醒||还懵懵的||你在干嘛呀」, messages 表只落了「刚醒」—— render_prompt 对非主动
模板一律 split("||")[0]。意图模板自己写「单条输出」, 但拼在前面的 Web 定制版
【通用回复规则】要求用 || 分句, 模型常照后者写, 后半句 (常是追问) 就被静默丢掉。
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.prompts.system_prompts import MAX_REPLY_COUNT

PROD_OUTPUT = "刚醒||还懵懵的||你在干嘛呀"


async def _drain(agen):
    return [evt async for evt in agen]


def _reply_payloads(events: list[dict]) -> list[dict]:
    return [json.loads(e["data"]) for e in events if e["event"] == "reply"]


def _run_background(tasks: list):
    def _capture(coro):
        task = asyncio.ensure_future(coro)
        tasks.append(task)
        return task
    return _capture


def _short_circuit_ctx(save_replies, **overrides):
    from app.services.chat.intent_handlers import ShortCircuitCtx

    fields: dict[str, Any] = dict(
        conversation_id="c1", agent_id=None, user_id="u1",
        agent=SimpleNamespace(name="A"),
        reply_context=None,
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=save_replies,
        pending_sub_fragments={},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=100,
        response_diagnostics={"version": 1},
    )
    fields.update(overrides)
    return ShortCircuitCtx(**fields)


# ═══════════════════════════════════════════════════════════════════
# 生成侧: render_prompt 保留 ||
# ═══════════════════════════════════════════════════════════════════


def test_normalize_reply_bubbles_keeps_every_bubble():
    from app.services.prompting.utils import normalize_reply_bubbles

    assert normalize_reply_bubbles(PROD_OUTPUT, 120) == PROD_OUTPUT
    assert normalize_reply_bubbles(" 刚醒 || ||还懵懵的|| ", None) == "刚醒||还懵懵的"
    assert normalize_reply_bubbles("", 120) == ""


def test_normalize_reply_bubbles_budgets_visible_chars_only():
    """预算不含分隔符: 三条加起来恰好 max_chars 时一条都不能丢."""
    from app.services.prompting.utils import normalize_reply_bubbles

    raw = "a" * 10 + "||" + "b" * 10 + "||" + "c" * 10
    assert normalize_reply_bubbles(raw, 30) == raw
    # 超预算: 前面整条保留, 最后放得下的那条按句末截, 其后丢弃
    raw = "第一条。||第二条有点长。后半句超了。||第三条"
    assert normalize_reply_bubbles(raw, 12) == "第一条。||第二条有点长。"
    # 剩余预算 ≤5 时不再硬塞碎片
    assert normalize_reply_bubbles("一二三四五六七八||第二条很长很长", 12) == "一二三四五六七八"
    # 首条本身超长照旧截断 (与旧行为一致)
    assert normalize_reply_bubbles("今天天气真好。我想出去走走。", 10) == "今天天气真好。"


@pytest.mark.asyncio
async def test_render_prompt_keep_bubbles_only_changes_opted_in_callers():
    from app.services.prompting.utils import render_prompt

    async def invoke(_prompt):
        return PROD_OUTPUT

    with patch(
        "app.services.prompting.utils.get_prompt_text",
        new=AsyncMock(return_value="tmpl"),
    ):
        kept = await render_prompt(
            "intent.current_state_reply", {}, invoke, max_chars=120, keep_bubbles=True,
        )
        legacy = await render_prompt("intent.current_state_reply", {}, invoke, max_chars=120)
        proactive = await render_prompt(
            "proactive.silence_plain", {}, invoke, max_chars=120, keep_bubbles=True,
        )

    assert kept == PROD_OUTPUT
    # 不传 keep_bubbles 的调用方 (出口不拆 ||) 行为不变
    assert legacy == "刚醒"
    # 主动出口不拆 ||, 仍合并成一句
    assert proactive == "刚醒 还懵懵的 你在干嘛呀"


@pytest.mark.asyncio
async def test_current_state_reply_returns_all_bubbles():
    """生产复现: 意图回复 helper 不再只返回「刚醒」."""
    from app.services.chat import intent_replies

    with (
        patch(
            "app.services.prompting.utils.get_prompt_text",
            new=AsyncMock(return_value="{message}"),
        ),
        patch.object(intent_replies, "invoke_text", new=AsyncMock(return_value=PROD_OUTPUT)),
        patch.object(intent_replies, "get_chat_model", return_value=object()),
    ):
        reply = await intent_replies.current_state_reply(message="在干嘛")

    assert reply == PROD_OUTPUT


@pytest.mark.asyncio
async def test_delay_explanation_stays_single_bubble():
    """延迟解释由 emit_replies 单独推一条, 不经短路出口 —— 保留 || 会原样漏给用户."""
    from app.services.chat import intent_replies

    with (
        patch(
            "app.services.prompting.utils.get_prompt_text",
            new=AsyncMock(return_value="{activity}"),
        ),
        patch.object(
            intent_replies, "invoke_text",
            new=AsyncMock(return_value="刚在忙||现在回你"),
        ),
        patch.object(intent_replies, "get_chat_model", return_value=object()),
    ):
        reply = await intent_replies.delay_explanation_reply(
            received_time="10:00", current_time="10:05",
            activity="开会", status="busy", delay_minutes=5,
        )

    assert reply == "刚在忙"


@pytest.mark.asyncio
async def test_boundary_reply_keeps_bubbles():
    from app.services.interaction import boundary

    with (
        patch.object(boundary, "get_prompt_text", new=AsyncMock(return_value="{message}")),
        patch.object(
            boundary, "invoke_text",
            new=AsyncMock(return_value="你这么说我有点难过||先缓缓吧"),
        ),
        patch.object(boundary, "get_chat_model", return_value=object()),
    ):
        reply = await boundary.generate_boundary_reply_llm(zone="medium", message="哼")

    assert reply == "你这么说我有点难过||先缓缓吧"


# ═══════════════════════════════════════════════════════════════════
# 出口侧: short_circuit_reply 切气泡
# ═══════════════════════════════════════════════════════════════════


def test_split_short_circuit_reply_bubbles_and_guards():
    from app.services.chat.reply_formatting import split_short_circuit_reply

    assert split_short_circuit_reply(PROD_OUTPUT) == ["刚醒", "还懵懵的", "你在干嘛呀"]
    # 切出来的气泡尾巴上悬空的逗号去掉; 系统标记逐条剥
    assert split_short_circuit_reply("刚醒，||你呢？[EMO:高兴/60]") == ["刚醒", "你呢？"]
    # 条数溢出合并进最后一条, 不丢
    many = "||".join(str(i) for i in range(MAX_REPLY_COUNT + 2))
    parts = split_short_circuit_reply(many)
    assert len(parts) == MAX_REPLY_COUNT
    assert " ".join(parts) == " ".join(str(i) for i in range(MAX_REPLY_COUNT + 2))
    # 没有 || 的程序拼接文案 (删除候选列表) 换行排版原样保留
    confirm = "你想把这些挪到 明天 对吗?\n- 交房租\n- 买菜\n\n回我「对」我就改"
    assert split_short_circuit_reply(confirm) == [confirm]
    assert split_short_circuit_reply("[EMO:中性/0]||") == ["..."]


@pytest.mark.asyncio
async def test_finalize_emits_every_bubble_with_continuous_indices():
    save_replies = AsyncMock(return_value="m1")
    ctx = _short_circuit_ctx(save_replies, reply_index_offset=2)
    tasks: list = []

    with (
        patch("app.services.chat.multi_intent._fire_background", side_effect=_run_background(tasks)),
        patch(
            "app.services.chat.reply_count_state.save_last_reply_count",
            new=AsyncMock(),
        ) as save_count,
    ):
        events = await _drain(ctx.finalize(PROD_OUTPUT, kind="current_state"))
        await asyncio.gather(*tasks)

    payloads = _reply_payloads(events)
    assert [p["text"] for p in payloads] == ["刚醒", "还懵懵的", "你在干嘛呀"]
    assert [p["index"] for p in payloads] == [2, 3, 4]
    assert events[-1]["event"] == "done"
    # 后台记忆/情绪管线拿到的是完整回复, 不是首条
    assert ctx.last_short_circuit_reply == "刚醒 还懵懵的 你在干嘛呀"
    # 上一轮气泡数 = offset + 气泡数 (图灵测试条数变化约束)
    save_count.assert_awaited_once_with("c1", 5)

    saved = save_replies.call_args.args[1]
    assert [r["text"] for r in saved] == ["刚醒", "还懵懵的", "你在干嘛呀"]
    # 判定窗 / 重复追问读最后一条 AI 消息的 metadata, 每条都要带短路标记
    assert all(
        r["response_diagnostics"]["short_circuit_kind"] == "current_state"
        for r in saved
    )


@pytest.mark.asyncio
async def test_boundary_metadata_reaches_last_bubble():
    """边界回复后不追问 (_blocks_followup) 看的是最后一条, 拆气泡后不能漏标."""
    from app.services.chat.multi_intent import short_circuit_reply
    from app.services.proactive.followup import _blocks_followup

    save_replies = AsyncMock(return_value="m1")
    tasks: list = []
    with (
        patch("app.services.chat.multi_intent._fire_background", side_effect=_run_background(tasks)),
        patch("app.services.chat.multi_intent.finish_assistant_turn", new=AsyncMock()),
        patch("app.services.chat.reply_count_state.save_last_reply_count", new=AsyncMock()),
    ):
        events = await short_circuit_reply(
            "你这么说我有点难过||先缓缓吧", "c1", None, "u1", save_replies,
            extra_metadata={"boundary": True, "zone": "medium"},
        )
        await asyncio.gather(*tasks)

    assert [p["text"] for p in _reply_payloads(events)] == ["你这么说我有点难过", "先缓缓吧"]
    saved = save_replies.call_args.args[1]
    assert _blocks_followup(saved[-1])


@pytest.mark.asyncio
async def test_one_emoji_per_turn_across_bubbles():
    from app.services.chat.multi_intent import short_circuit_reply

    save_replies = AsyncMock(return_value="m1")
    with (
        patch("app.services.chat.multi_intent._fire_background", side_effect=lambda c: c.close()),
        patch("app.services.chat.multi_intent.finish_assistant_turn", new=AsyncMock()),
    ):
        events = await short_circuit_reply(
            "刚醒😴||你在干嘛呀😊", "c1", None, "u1", save_replies,
        )

    assert [p["text"] for p in _reply_payloads(events)] == ["刚醒😴", "你在干嘛呀"]


@pytest.mark.asyncio
async def test_voice_reply_stays_one_bubble_with_full_text(monkeypatch):
    from app.services.chat import multi_intent
    from app.services.speech_output import delivery, policy
    from app.services.speech_output.policy import VoiceContext

    spoken: list[str] = []

    async def fake_prepare(**kwargs):
        spoken.append(kwargs["text"])
        return SimpleNamespace(transcript=kwargs["text"], metadata={"id": "att-1"})

    async def always_voice(**kwargs):
        return True

    async def passthrough_guard(_agent, text, **_kwargs):
        return text, False

    monkeypatch.setattr(policy, "should_generate_voice", always_voice)
    monkeypatch.setattr(delivery, "prepare_voice_output", fake_prepare)
    monkeypatch.setattr(delivery, "bind_prepared_voice_output", AsyncMock())
    monkeypatch.setattr("app.services.persona_grounding.guard_reply", passthrough_guard)
    monkeypatch.setattr(multi_intent, "_fire_background", lambda c: c.close())
    monkeypatch.setattr(multi_intent, "finish_assistant_turn", AsyncMock())

    save_replies = AsyncMock(return_value="m1")
    events = await multi_intent.short_circuit_reply(
        "晚安||早点休息", "c1", "a1", "u1", save_replies,
        agent=SimpleNamespace(id="a1"),
        reply_context={"client_supports_voice": True},
        voice_context=VoiceContext.NORMAL_CHAT,
    )

    assert spoken == ["晚安 早点休息"]
    payloads = _reply_payloads(events)
    assert len(payloads) == 1
    assert payloads[0]["text"] == "晚安 早点休息"
    assert payloads[0]["display_mode"] == "voice"
    assert len(save_replies.call_args.args[1]) == 1


@pytest.mark.asyncio
async def test_sub_intents_start_after_all_primary_bubbles():
    from app.services.chat import multi_intent

    captured: dict = {}

    async def fake_sub_intents(*args, start_index, **kwargs):
        captured["start_index"] = start_index
        return
        yield  # pragma: no cover - async generator marker

    with (
        patch.object(multi_intent, "process_sub_intents", fake_sub_intents),
        patch.object(multi_intent, "_fire_background", side_effect=lambda c: c.close()),
        patch.object(multi_intent, "finish_assistant_turn", new=AsyncMock()),
    ):
        events = await _drain(multi_intent.finalize_short_circuit(
            PROD_OUTPUT,
            conversation_id="c1", agent_id=None, user_id="u1", agent=None,
            reply_context=None,
            tracer=MagicMock(safe_trace_id=None),
            save_replies_fn=AsyncMock(return_value="m1"),
            pending_sub_fragments={"日常交流": "顺便问下"},
            sub_intent_mode=False,
            reply_index_offset=1,
            cached_patience=100,
        ))

    assert [p["index"] for p in _reply_payloads(events)] == [1, 2, 3]
    assert captured["start_index"] == 4


@pytest.mark.asyncio
async def test_boundary_phase_records_delivered_text():
    from app.services.chat.boundary_phase import _emit_short_circuit
    from app.services.chat.multi_intent import short_circuit_reply

    async def short_circuit_fn(reply, conversation_id, agent_id, user_id, **kwargs):
        kwargs.pop("trace_id", None)
        return await short_circuit_reply(
            reply, conversation_id, agent_id, user_id,
            AsyncMock(return_value="m1"), **kwargs,
        )

    ctx = SimpleNamespace(
        short_circuit_fn=short_circuit_fn,
        conversation_id="c1", agent_id=None, user_id="u1",
        tracer=MagicMock(safe_trace_id=None),
        last_short_circuit_reply=None,
    )
    with (
        patch("app.services.chat.multi_intent._fire_background", side_effect=lambda c: c.close()),
        patch("app.services.chat.multi_intent.finish_assistant_turn", new=AsyncMock()),
    ):
        await _drain(_emit_short_circuit(ctx, "好啦||不生气了", {"boundary": True}))

    assert ctx.last_short_circuit_reply == "好啦 不生气了"
