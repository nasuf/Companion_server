"""对话上下文按用户回合截取, 不按消息行数.

AI 一次回复拆成 1-4 个气泡, 每个气泡在库里是独立一行 (全库 user:assistant ≈ 1:2.56).
2026-10-15 的 trace (用户说"我喜欢你"): 记忆相关度判定的 4 行窗口里装的是上一轮
AI 的 3 个气泡加当前消息本身, 一句此前的用户发言都没有; 意图识别的 6 行只剩 1.5 轮;
记忆预筛只拿到最后一个气泡 "你更喜欢那种呀？", 不知道问的是软甜还是酷甜.
"""

from __future__ import annotations

import pytest

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.chat.data_fetch_phase import format_recent_context
from app.services.chat.message_utils import last_user_turns

_T0 = datetime(2026, 10, 15, 7, 24, tzinfo=UTC)


def _trace_dialogue() -> list[dict]:
    """The tail of the 2026-10-15 conversation, one row per bubble."""
    rows = [
        ("u1", "user", "还以为你长这样呢"),
        ("a1", "assistant", "哈哈 那你觉得我长啥样呀？"),
        ("a2", "assistant", "是偏甜的还是酷的呀？🤪"),
        ("u2", "user", "甜吧"),
        ("a3", "assistant", "被你说甜还挺开心的✨"),
        ("a4", "assistant", "你觉得我是软甜还是酷甜呀？"),
        ("u3", "user", "有什么区别啊"),
        ("a5", "assistant", "软甜就是说话温温的👌"),
        ("a6", "assistant", "酷甜是带点小拽的那种"),
        ("a7", "assistant", "你更喜欢那种呀？"),
        ("cur", "user", "我喜欢你"),
    ]
    return [
        {"id": i, "role": r, "content": c, "createdAt": (_T0 + timedelta(seconds=n)).isoformat()}
        for n, (i, r, c) in enumerate(rows)
    ]


class TestLastUserTurns:
    def test_counts_user_rows_not_rows(self):
        items = ["u", "a", "a", "a", "u", "a", "a", "u", "a"]
        assert last_user_turns(items, 2, lambda x: "user" if x == "u" else "assistant") == [
            "u", "a", "a", "u", "a",
        ]

    def test_fewer_user_turns_returns_everything(self):
        items = ["a", "u", "a"]
        assert last_user_turns(items, 3, lambda x: "user" if x == "u" else "assistant") == items

    def test_zero_turns_is_empty(self):
        assert last_user_turns(["u"], 0, lambda x: "user") == []


class TestRecentContext:
    def test_trace_window_keeps_three_user_turns(self):
        text = format_recent_context(_trace_dialogue(), exclude_message_ids={"cur"})
        assert text.splitlines() == ["[10-15 15:24] " + line for line in [
            "用户: 还以为你长这样呢",
            "AI: 哈哈 那你觉得我长啥样呀？ 是偏甜的还是酷的呀？🤪",
            "用户: 甜吧",
            "AI: 被你说甜还挺开心的✨ 你觉得我是软甜还是酷甜呀？",
            "用户: 有什么区别啊",
            "AI: 软甜就是说话温温的👌 酷甜是带点小拽的那种 你更喜欢那种呀？",
        ]]

    def test_current_message_is_not_shown_twice(self):
        assert "我喜欢你" not in format_recent_context(
            _trace_dialogue(), exclude_message_id="cur",
        )

    def test_over_budget_drops_oldest_whole_lines(self):
        text = format_recent_context(
            _trace_dialogue(), exclude_message_ids={"cur"}, max_chars=60,
        )
        assert len(text) <= 60
        assert text.splitlines() == ["[10-15 15:24] AI: 软甜就是说话温温的👌 酷甜是带点小拽的那种 你更喜欢那种呀？"]

    def test_long_ai_reply_keeps_its_final_question(self):
        msgs = [
            {"id": "u", "role": "user", "content": "在吗"},
            {"id": "a", "role": "assistant", "content": "铺垫" * 1000 + "你周末去哪玩？"},
        ]
        line = format_recent_context(msgs).splitlines()[-1]
        assert line.startswith("AI: …") and line.endswith("你周末去哪玩？")

    def test_empty_history_uses_placeholder(self):
        assert format_recent_context([]) == "（无）"


async def test_intent_context_survives_multi_bubble_replies():
    from app.services.chat.orchestrator import _fetch_intent_context

    rows = [
        SimpleNamespace(id=m["id"], role=m["role"], content=m["content"])
        for m in reversed(_trace_dialogue())  # DB desc
    ]
    with patch("app.services.chat.orchestrator.db") as mock_db:
        mock_db.message = MagicMock()
        mock_db.message.find_many = AsyncMock(return_value=rows)
        context = await _fetch_intent_context("conv", exclude_id="cur")

    assert context.splitlines()[0] == "用户: 还以为你长这样呢"
    assert "我喜欢你" not in context


async def test_relevance_and_web_search_exclude_the_current_turn():
    from app.services.chat import data_fetch_phase as dfp
    from app.services.memory.retrieval.relevance import RelevanceResult

    seen: dict[str, str] = {}

    async def relevance(message, context=""):
        seen["relevance"] = context
        return RelevanceResult(level="weak", enhanced_query="")

    async def web_search(message, context):
        seen["web"] = context
        return False

    with patch.object(dfp, "_classify_relevance", relevance), \
            patch.object(dfp, "_decide_web_search", web_search), \
            patch.object(dfp, "_should_fast_weak_relevance", lambda _m: False), \
            patch.object(dfp, "analyze_user_emotion", AsyncMock(return_value={})), \
            patch.object(dfp, "get_latest_portrait", AsyncMock(return_value=None)), \
            patch.object(dfp, "get_cached_schedule", AsyncMock(return_value=None)), \
            patch.object(dfp, "get_topic_intimacy", AsyncMock(return_value=50.0)):
        await dfp.fetch_parallel_context(
            user_id="u", agent_id="a", workspace_id=None,
            user_message="我喜欢你", messages_dicts=_trace_dialogue(), parsed_times=[],
            exclude_message_ids={"cur"},
        )

    for key in ("relevance", "web"):
        assert "我喜欢你" not in seen[key]
        assert seen[key].startswith("[10-15 15:24] 用户: 还以为你长这样呢")


class TestTwentyTurnWindows:
    """Prompt docs ask for 前20轮; 40 rows used to be only ~11 turns."""

    async def test_followup_judge_keeps_twenty_user_turns(self, monkeypatch):
        from app.services.proactive import followup

        rows = []
        for n in range(30):  # 30 turns, AI answers in 3 bubbles
            rows.append({"id": f"u{n}", "role": "user", "content": f"用户{n}", "metadata": None})
            for b in range(3):
                rows.append({"id": f"a{n}{b}", "role": "assistant", "content": f"AI{n}-{b}",
                             "metadata": {}})
        monkeypatch.setattr(followup.db, "query_raw", AsyncMock(return_value=list(reversed(rows))))

        turns = await followup._load_recent_turns("c-1")

        assert sum(t.role == "user" for t in turns) == 20
        assert turns[0].id == "u10" and turns[-1].id == "a292"

    def test_topic_continuation_uses_five_turns_without_changing_followup_judge(self):
        from app.services.chat.topic_continuation import _history_text

        history = []
        for n in range(30):
            history.append(SimpleNamespace(id=f"u{n}", role="user", content=f"用户说{n}",
                                           metadata=None, createdAt=None))
            for b in range(3):
                history.append(SimpleNamespace(id=f"a{n}{b}", role="assistant",
                                               content=f"AI说{n}-{b}", metadata=None,
                                               createdAt=None))

        lines = _history_text(history, set()).splitlines()

        assert sum(line.startswith("用户") for line in lines) == 5
        assert lines[0] == "用户: 用户说25"


class TestMemoryPipelineContext:
    def test_prefilter_gets_the_whole_preceding_reply(self):
        from app.services.chat.post_process import _ai_line_before

        msgs = _trace_dialogue()
        assert _ai_line_before(msgs, msgs[-1]) == (
            "软甜就是说话温温的👌 酷甜是带点小拽的那种 你更喜欢那种呀？"
        )

    def test_prefilter_skips_user_fragments_after_the_reply(self):
        """Aggregated fragments: the AI line still precedes the first fragment."""
        from app.services.chat.post_process import _ai_line_before

        msgs = [
            {"id": "a", "role": "assistant", "content": "你今天还好吗"},
            {"id": "u1", "role": "user", "content": "还行"},
            {"id": "u2", "role": "user", "content": "就是有点累"},
        ]
        assert _ai_line_before(msgs, msgs[2]) == "你今天还好吗"

    async def _run_side(self, side: Literal["user", "ai"]):
        from app.services.chat import post_process as pp

        history = _trace_dialogue()
        wm = datetime.fromisoformat(history[-2]["createdAt"])
        reply_ts = _T0 + timedelta(minutes=1)
        bubbles = [
            {"id": "r1", "role": "assistant", "content": "啊？真的假的", "createdAt": reply_ts},
            {"id": "r2", "role": "assistant", "content": "我没看错吧？😮",
             "createdAt": reply_ts + timedelta(seconds=1)},
        ]
        backlog = (
            [{**history[-1], "createdAt": datetime.fromisoformat(history[-1]["createdAt"])}]
            if side == "user" else bubbles
        )
        recent = (history + [{"role": "assistant", "content": "啊？真的假的 我没看错吧？😮"}])[-6:]
        pipeline = AsyncMock(return_value=[])
        with patch.object(pp, "get_watermark", AsyncMock(return_value=wm)), \
                patch.object(pp, "set_watermark", AsyncMock()), \
                patch.object(pp, "_fetch_new_side_messages", AsyncMock(return_value=backlog)), \
                patch.object(pp, "process_memory_pipeline", pipeline):
            await pp._pipeline_with_watermark("u", recent, "conv", side=side)
        return pipeline.await_args.kwargs

    async def test_ai_side_does_not_extract_the_reply_twice(self):
        kwargs = await self._run_side("ai")
        assert kwargs["new_conversation"].splitlines() == [
            "assistant: 啊？真的假的",
            "assistant: 我没看错吧？😮",
        ]

    async def test_user_side_history_stops_before_the_ai_reply(self):
        kwargs = await self._run_side("user")
        assert kwargs["new_conversation"] == "user: 我喜欢你"
        assert "真的假的" not in kwargs["context_conversation"]
        assert kwargs["context_conversation"].splitlines()[-1] == (
            "assistant: 软甜就是说话温温的👌 / 酷甜是带点小拽的那种 / 你更喜欢那种呀？"
        )

    def test_untimestamped_reply_keeps_the_user_turn_it_answers(self):
        """First run (no watermark): old persisted bubbles plus the fresh reply are all
        targets; the fresh reply has no timestamp, so the user line it answers stays."""
        from app.services.chat.post_process import _context_before_targets

        old_reply = {"role": "assistant", "content": "旧回复", "createdAt": _T0.isoformat()}
        user = {"role": "user", "content": "我喜欢你",
                "createdAt": (_T0 + timedelta(minutes=1)).isoformat()}
        fresh = {"role": "assistant", "content": "啊？真的假的"}
        recent = [old_reply, user, fresh]

        assert _context_before_targets([user], [old_reply, fresh], recent, "assistant") == [user]


    def test_ai_reply_between_two_new_user_rows_stays(self):
        """Only rows after the newest target are hidden; a reply in between is real history."""
        from app.services.chat.post_process import _context_before_targets

        def row(role, text, minutes):
            return {"role": role, "content": text,
                    "createdAt": (_T0 + timedelta(minutes=minutes)).isoformat()}

        u1, between, u2, after = row("user", "a", 0), row("assistant", "b", 1), \
            row("user", "c", 2), row("assistant", "d", 3)
        kept = _context_before_targets([between, after], [u1, u2], [u1, between, u2, after], "user")
        assert kept == [between]


class TestSharedDialogueGrouping:
    def test_template_and_memory_share_groups_for_normal_bubbles(self):
        from app.services.chat.message_utils import group_dialogue_turns, TURN_BUBBLE_MAX_GAP
        from app.services.chat.dialogue_context import render_dialogue
        from app.services.memory.recording.context_window import history_rows, format_history
        rows = _trace_dialogue()[:-1]
        groups = group_dialogue_turns(rows)
        memory_groups = group_dialogue_turns(rows, user_max_gap=TURN_BUBBLE_MAX_GAP)
        assert groups == memory_groups
        assert render_dialogue(rows, turns=3).metrics['injected_rounds'] == 3
        assert history_rows(rows) == rows
        assert format_history(rows).count('assistant:') == render_dialogue(rows, turns=3).text.count('AI:') == 3
        assert all(any(row is original for original in rows) for group in groups for row in group)

    def test_user_silence_boundary_is_explicit_and_predictable(self):
        from app.services.chat.message_utils import group_dialogue_turns, TURN_BUBBLE_MAX_GAP
        from app.services.chat.dialogue_context import render_dialogue
        from app.services.memory.recording.context_window import history_rows
        rows = [dict(id=str(i), role='user', content=f'独立陈述{i}', createdAt=_T0+timedelta(hours=i)) for i in range(4)]
        assert len(group_dialogue_turns(rows)) == 1
        assert len(group_dialogue_turns(rows, user_max_gap=TURN_BUBBLE_MAX_GAP)) == 4
        assert render_dialogue(rows, turns=3).metrics['injected_rounds'] == 1
        assert history_rows(rows) == rows[1:]

    @pytest.mark.parametrize('gap,expected', [(0,1),(600,1),(601,2),(-1,2),(None,2)])
    def test_finite_user_gap_edges(self, gap, expected):
        from app.services.chat.message_utils import group_dialogue_turns, TURN_BUBBLE_MAX_GAP
        rows = [dict(role='user',createdAt=_T0),dict(role='user',createdAt=_T0+timedelta(seconds=gap) if gap is not None else None)]
        assert len(group_dialogue_turns(rows, user_max_gap=TURN_BUBBLE_MAX_GAP)) == expected
        assert len(group_dialogue_turns(rows)) == 1

    def test_unknown_roles_are_boundaries_and_assistant_identity_stays_last(self):
        from app.services.chat.message_utils import group_dialogue_turns, previous_assistant_turn
        rows = _trace_dialogue()[:3]
        groups = group_dialogue_turns(rows)
        assert previous_assistant_turn(rows, None).id == groups[-1][-1]['id'] == 'a2'
        rows.insert(2, dict(role='system',content='boundary'))
        assert len(group_dialogue_turns(rows)) == 4
