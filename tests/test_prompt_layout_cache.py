"""Main reply prompt layout: stable system + history + per-turn system.

2026-10-15 trace: of 8075 input tokens only 3896 hit the provider prefix cache.
The per-turn sections (state, time, reply timing with a microsecond timestamp,
topic turn count, …) sat at the end of the leading system prompt, so the ~4000
tokens of history after them could never be cached; and once the token budget
was full the window start moved every turn, which breaks the prefix as well.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat import prompt_builder as pb
from app.services.chat.repetition_guard import find_repeated_ai_phrases
from app.services.memory.retrieval.context_selector import estimate_tokens

_T0 = datetime(2026, 10, 15, 7, 0, tzinfo=UTC)


def _prompt_text(key: str) -> str:
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    definition = PROMPT_DEFINITION_MAP.get(key)
    return definition.default_text if definition else ""


def _patch_prompts():
    return (
        patch.object(pb, "get_prompt_text", AsyncMock(side_effect=_prompt_text)),
        patch.object(pb, "get_prompt_text_or_default", AsyncMock(side_effect=_prompt_text)),
    )


def _agent():
    return type("A", (), {"name": "小伴", "gender": "female", "occupation": "客服", "city": "普洱"})()


async def _system_prompt(**kwargs):
    p1, p2 = _patch_prompts()
    with p1, p2:
        return await pb.build_system_prompt(agent=_agent(), **kwargs)


class TestSplitLayout:
    async def test_per_turn_sections_go_after_the_history(self):
        prompt = await _system_prompt(time_context="当前时间：2026年10月15日 15时", last_reply_count=3)
        history = [
            {"id": "u1", "role": "user", "content": "你好", "createdAt": _T0.isoformat()},
            {"id": "a1", "role": "assistant", "content": "嗨", "createdAt": _T0.isoformat()},
            {"id": "u2", "role": "user", "content": "在干嘛",
             "createdAt": (_T0 + timedelta(minutes=1)).isoformat()},
        ]

        messages = pb.build_chat_messages(prompt, history)

        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "system"]
        lead, turn = messages[0]["content"], messages[-1]["content"]
        assert "## 回复要求" in lead and "## 对话一致性" in lead
        assert "## 时间" not in lead and "## 条数变化" not in lead
        assert turn.startswith("## 时间") and "## 条数变化" in turn
        # Trace spans were recorded over the joined text; it must stay byte-identical.
        assert f"{lead}\n\n{turn}" == str(prompt)

    async def test_lead_is_identical_across_turns(self):
        first = await _system_prompt(time_context="当前时间：15时", last_reply_count=3)
        second = await _system_prompt(time_context="当前时间：16时", last_reply_count=2)
        assert pb.split_system_prompt(first)[0] == pb.split_system_prompt(second)[0]

    def test_plain_string_prompt_keeps_single_system(self):
        messages = pb.build_chat_messages("系统", [{"role": "user", "content": "嗨"}])
        assert [m["role"] for m in messages] == ["system", "user"]

    def test_fold_restores_the_single_system_layout(self):
        prompt = pb.SystemPromptText("稳定段\n\n本轮段", turn_offset=len("稳定段") + 2)
        messages = pb.build_chat_messages(prompt, [{"role": "user", "content": "嗨"}])
        folded = pb.fold_trailing_system(messages)
        assert folded == [
            {"role": "system", "content": "稳定段\n\n本轮段"},
            {"role": "user", "content": "嗨"},
        ]


class TestHistoryWindow:
    def test_window_start_moves_in_steps_not_every_turn(self):
        rows = []
        for n in range(400):
            role = "user" if n % 2 == 0 else "assistant"
            rows.append({"id": f"m{n:04d}", "role": role, "content": "这是一句普通的聊天内容" * 2,
                         "createdAt": (_T0 + timedelta(minutes=n)).isoformat()})

        starts = []
        for end in range(200, 400, 2):  # one turn = one user + one AI row
            messages = pb.build_chat_messages("系统", rows[:end], token_budget=1500)
            starts.append(messages[1]["content"])
            assert sum(estimate_tokens(m["content"]) for m in messages[1:]) <= 1500

        changes = sum(a != b for a, b in zip(starts, starts[1:]))
        # Greedy filling moves the start on every turn; cut points hold it for several.
        assert changes < len(starts) * 0.4

    def test_short_history_is_kept_whole(self):
        rows = [{"id": f"m{n}", "role": "user" if n % 2 == 0 else "assistant", "content": "嗨",
                 "createdAt": (_T0 + timedelta(minutes=n)).isoformat()} for n in range(6)]
        messages = pb.build_chat_messages("系统", rows)
        assert len(messages) == 1 + 6


class TestBubbleTimestamps:
    def test_late_ai_message_keeps_its_own_time(self):
        rows = [
            {"role": "user", "content": "手机忘带了 你拿什么发的消息", "createdAt": "2026-10-11T11:28:00+00:00"},
            {"role": "assistant", "content": "啊我骗你的", "createdAt": "2026-10-11T11:28:05+00:00"},
            {"role": "assistant", "content": "就是不想拍啦", "createdAt": "2026-10-11T11:28:06+00:00"},
            {"role": "assistant", "content": "好久没聊啦，最近怎么样？",
             "createdAt": "2026-10-13T00:10:00+00:00"},
        ]
        merged = pb._coalesce_bubbles(rows)
        assert merged[1]["content"] == "啊我骗你的 就是不想拍啦 [2026-10-13 08:10] 好久没聊啦，最近怎么样？"


class TestRepetitionGuard:
    _ROWS = [
        ("assistant", "对呀 上班偷偷刷手机呢😜 你也在摸鱼吗？"),
        ("user", "你在上班？"),
        ("assistant", "没有啦 我刚整理完酿造的东西 正歇着刷手机呢 你今天还忙吗？"),
        ("user", "你不是在睡觉吗"),
        ("assistant", "没睡呀"),
        ("assistant", "正歇着刷手机呢 你今天还忙吗？"),
        ("user", "你在干嘛"),
    ]

    def test_detects_phrases_repeated_across_replies(self):
        msgs = [{"role": r, "content": c} for r, c in self._ROWS]
        assert find_repeated_ai_phrases(msgs) == ["正歇着刷手机呢", "你今天还忙吗"]

    def test_bubbles_of_one_reply_count_once(self):
        msgs = [{"role": "assistant", "content": "你今天还忙吗"},
                {"role": "assistant", "content": "你今天还忙吗"}]
        assert find_repeated_ai_phrases(msgs) == []

    def test_laughter_is_not_a_repetition(self):
        msgs = [{"role": "assistant", "content": "哈哈哈哈 好呀"}, {"role": "user", "content": "嗯"},
                {"role": "assistant", "content": "哈哈哈哈 行吧"}]
        assert find_repeated_ai_phrases(msgs) == []

    async def test_section_is_injected_in_the_turn_part(self):
        prompt = await _system_prompt(repeated_phrases=["你今天还忙吗"])
        _, turn = pb.split_system_prompt(prompt)
        assert "## 避免重复" in turn and "「你今天还忙吗」" in turn

    async def test_no_section_without_repetition(self):
        assert "## 避免重复" not in await _system_prompt(repeated_phrases=[])


async def test_claude_route_folds_the_trailing_system():
    from app.services.chat import reply_generate as rg

    seen = {}

    def convert(messages):
        seen["messages"] = messages
        raise RuntimeError("stop after conversion")

    messages = [{"role": "system", "content": "稳定"}, {"role": "user", "content": "嗨"},
                {"role": "system", "content": "本轮"}]
    with patch.object(rg, "get_chat_model", lambda: object()), \
            patch.object(rg, "provider_name", lambda _m: "claude"), \
            patch.object(rg, "convert_messages", convert), \
            pytest.raises(RuntimeError, match="stop after conversion"):
        await rg._run_main_llm(messages)

    assert seen["messages"] == [{"role": "system", "content": "稳定\n\n本轮"},
                                {"role": "user", "content": "嗨"}]


def test_trace_matches_components_over_the_split_prompt():
    from app.services.chat.trace_enrich import apply_prompt_render_traces
    from app.services.prompting.trace_components import prompt_hash

    lead, turn = "## 回复要求\n短句", "## 时间\n15时"
    step = {
        "run_type": "llm",
        "inputs": {"messages": [[
            {"id": ["langchain", "schema", "messages", "SystemMessage"],
             "kwargs": {"type": "system", "content": lead}},
            {"id": ["langchain", "schema", "messages", "HumanMessage"],
             "kwargs": {"type": "human", "content": "嗨"}},
            {"id": ["langchain", "schema", "messages", "SystemMessage"],
             "kwargs": {"type": "system", "content": turn}},
        ]]},
    }
    component = {"prompt_key": "chat.time_context_section", "start": len(lead) + 2 + 6,
                 "end": len(lead) + 2 + len(turn), "editable": True}
    apply_prompt_render_traces([step], [{
        "prompt_hash": prompt_hash(f"{lead}\n\n{turn}"), "source": "chat.system_prompt",
        "prompt_key": "chat.system_base", "components": [component],
    }])
    assert step["prompt_render_source"] == "chat.system_prompt"
    assert step["prompt_components"][0]["prompt_key"] == "chat.time_context_section"


@pytest.mark.asyncio
async def test_temporal_sections_have_separate_editable_spans_after_stable_prefix():
    from app.services.prompting.trace_components import (
        start_prompt_render_trace, reset_prompt_render_trace, snapshot_prompt_render_traces,
    )
    token = start_prompt_render_trace()
    try:
        prompt = await _system_prompt(temporal_instructions="当前时间：2026-10-19 19:54",
                                      temporal_evidence="今天没有该事件的来源证据")
        lead, turn = pb.split_system_prompt(prompt)
        assert "2026-10-19" not in lead
        assert "当前时间：2026-10-19 19:54" in turn
        components = [c for t in snapshot_prompt_render_traces() for c in t.get("components", [])]
        selected = {c["prompt_key"]: str(prompt)[c["start"]:c["end"]]
                    for c in components if c["prompt_key"] in {"chat.temporal_context", "chat.temporal_evidence"}}
        assert selected == {"chat.temporal_context": "当前时间：2026-10-19 19:54",
                            "chat.temporal_evidence": "今天没有该事件的来源证据"}
    finally:
        reset_prompt_render_trace(token)
