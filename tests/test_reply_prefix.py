"""回复类指令固定前置 (reply_prefix) 测试.

2026-07-08 产品决策: 聊天回复规则 + 反幻觉硬约束作为所有 AI 用户可见消息
(含主动消息) 的固定前置. 覆盖: key 集合卫生 / 前置构建与停用语义 /
store 注入点 / EMO 标记拆分 / 主动消息 || 防御.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.prompting.reply_prefix import (
    PREFIX_SOURCE_KEYS,
    REPLY_PROMPT_KEYS,
    build_reply_prefix,
)


# ═══════════════════════════════════════════════════════════════════
# Key 集合卫生
# ═══════════════════════════════════════════════════════════════════


def test_reply_keys_all_registered():
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    unknown = [k for k in REPLY_PROMPT_KEYS if k not in PROMPT_DEFINITION_MAP]
    assert not unknown, f"REPLY_PROMPT_KEYS 含未注册 key: {unknown}"


def test_prefix_sources_not_in_reply_keys():
    """前置来源模板绝不能在集合内 — 否则 get_prompt_text 无限递归."""
    assert not (set(PREFIX_SOURCE_KEYS) & REPLY_PROMPT_KEYS)


def test_classifier_and_section_keys_excluded():
    """分类器/JSON 抽取/section 片段不许进集合 — 前置会污染结构化输出.
    这里列典型代表作守卫 (新增误加时至少一个会命中)."""
    forbidden = {
        "intent.unified", "intent.split",
        "memory.relevance", "memory.l3_trigger",
        "memory.extraction_user", "memory.extraction_ai",
        "memory.judgement_user", "memory.deletion_intent",
        "memory.contradiction_detection", "memory.contradiction_analysis",
        "boundary.attack_target", "boundary.attack_level",
        "boundary.banned_word", "boundary.apology",
        "emotion.user_label", "reply.emotion_detection",
        "proactive.reminder_pre_check", "proactive.memory_topic_rerank",
        "chat.system_base", "chat.reply_emotion_marker",
        "chat.session_recap", "expression.learn_style",
        "offline.gift_selection", "offline.gift_candidate_pick",
        "music.user_pause_followup_decision",
        "schedule.daily_schedule", "character.generation",
    }
    hit = forbidden & REPLY_PROMPT_KEYS
    assert not hit, f"非回复类 key 被误加进 REPLY_PROMPT_KEYS: {hit}"


# ═══════════════════════════════════════════════════════════════════
# 前置构建
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_build_reply_prefix_renders_constants():
    from app.services.prompting.defaults import (
        ANTI_HALLUCINATION_HARD_RULE_PROMPT,
        RESPONSE_INSTRUCTION_PROMPT,
    )

    async def fake_get(key, **_kwargs):
        return {
            "chat.response_instruction": RESPONSE_INSTRUCTION_PROMPT,
            "chat.anti_hallucination_hard_rule": ANTI_HALLUCINATION_HARD_RULE_PROMPT,
        }[key]

    with patch(
        "app.services.prompting.store.get_prompt_text",
        AsyncMock(side_effect=fake_get),
    ):
        prefix = await build_reply_prefix()

    assert prefix.startswith("【通用回复规则】")
    assert "回答前的硬约束" in prefix
    # 装饰性占位符已渲染成常量, 不留 {max_per}/{total} 残渣
    assert "{max_per}" not in prefix and "{total}" not in prefix
    assert "60" in prefix and "150" in prefix
    # EMO 标记指令绝不在前置里 (只有主回复管线剥标记)
    assert "EMO" not in prefix


@pytest.mark.asyncio
async def test_build_reply_prefix_respects_disabled():
    from app.services.prompting.store import PromptDisabledError

    async def fake_get(key, **_kwargs):
        if key == "chat.response_instruction":
            raise PromptDisabledError(key)
        return "【回答前的硬约束】…"

    with patch(
        "app.services.prompting.store.get_prompt_text",
        AsyncMock(side_effect=fake_get),
    ):
        prefix = await build_reply_prefix()
    assert "通用回复规则" not in prefix
    assert "回答前的硬约束" in prefix

    async def all_disabled(key, **_kwargs):
        raise PromptDisabledError(key)

    with patch(
        "app.services.prompting.store.get_prompt_text",
        AsyncMock(side_effect=all_disabled),
    ):
        assert await build_reply_prefix() == ""


# ═══════════════════════════════════════════════════════════════════
# store 注入点
# ═══════════════════════════════════════════════════════════════════


class _FakeRedis:
    def __init__(self, store: dict[str, str]):
        self.store = store

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


@pytest.mark.asyncio
async def test_get_prompt_text_injects_prefix_for_reply_keys(monkeypatch):
    from app.services.prompting import store

    redis_store = {
        "prompt_template:intent.end_reply": "【任务】生成道别语 {message}",
        "prompt_template:intent.unified": "【任务】意图分类 {message}",
        "prompt_template:chat.response_instruction": "像朋友发微信那样回复，每条不超过{max_per}个字。",
        "prompt_template:chat.anti_hallucination_hard_rule": "【回答前的硬约束】不许编造。",
    }
    monkeypatch.setattr(store, "get_redis", AsyncMock(return_value=_FakeRedis(redis_store)))
    monkeypatch.setattr(store, "is_prompt_enabled", AsyncMock(return_value=True))

    reply_text = await store.get_prompt_text("intent.end_reply")
    assert str(reply_text).startswith("【通用回复规则】")
    assert "不许编造" in str(reply_text)
    assert str(reply_text).endswith("【任务】生成道别语 {message}")
    # ManagedPromptText 语义保留: prompt_key 仍是原模板 (trace 归属不变)
    assert reply_text.prompt_key == "intent.end_reply"

    classifier_text = await store.get_prompt_text("intent.unified")
    assert "通用回复规则" not in str(classifier_text)
    assert str(classifier_text) == "【任务】意图分类 {message}"


@pytest.mark.asyncio
async def test_prefix_failure_falls_back_to_bare_template(monkeypatch):
    """前置构建整体失败 (编程错误/Redis 半路挂) → 退回裸模板, 回复链路不断."""
    from app.services.prompting import store

    redis_store = {"prompt_template:intent.end_reply": "【任务】道别"}
    monkeypatch.setattr(store, "get_redis", AsyncMock(return_value=_FakeRedis(redis_store)))
    monkeypatch.setattr(store, "is_prompt_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "app.services.prompting.reply_prefix.build_reply_prefix",
        AsyncMock(side_effect=RuntimeError("redis down")),
    )
    text = await store.get_prompt_text("intent.end_reply")
    assert str(text) == "【任务】道别"


# ═══════════════════════════════════════════════════════════════════
# EMO 标记拆分
# ═══════════════════════════════════════════════════════════════════


def test_emotion_marker_split_out_of_response_instruction():
    from app.services.prompting.defaults import (
        CHAT_REPLY_EMOTION_MARKER_PROMPT,
        RESPONSE_INSTRUCTION_PROMPT,
    )

    assert "EMO" not in RESPONSE_INSTRUCTION_PROMPT, (
        "EMO 标记指令必须拆出 response_instruction — 它现在是所有回复类"
        "指令的前置, 非主回复路径不剥标记会漏给用户"
    )
    assert "[EMO:标签/强度]" in CHAT_REPLY_EMOTION_MARKER_PROMPT
    # 12 类标签仍齐全 (与 EMOJI_MAP 对齐由 test_emotion_marker 守卫)
    assert "高兴" in CHAT_REPLY_EMOTION_MARKER_PROMPT


# ═══════════════════════════════════════════════════════════════════
# 主动消息 || 防御
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_emit_proactive_collapses_multi_segment(monkeypatch):
    from app.services.proactive import emit as emit_mod

    created = SimpleNamespace(id="msg-1", createdAt=None)
    fake_db = MagicMock()
    fake_db.message.create = AsyncMock(return_value=created)
    fake_db.proactivechatlog.create = AsyncMock()
    monkeypatch.setattr(emit_mod, "db", fake_db)
    monkeypatch.setattr(
        emit_mod, "manager", SimpleNamespace(send_to_workspace=AsyncMock()),
    )

    await emit_mod.emit_proactive_message(
        conversation_id="c1", user_id="u1", agent_id="a1", workspace_id="w1",
        message="今晚月色真好||想到你了||早点休息",
        trigger_type="silence_wakeup",
    )

    stored = fake_db.message.create.call_args.kwargs["data"]["content"]
    assert stored == "今晚月色真好"
    assert "||" not in stored
