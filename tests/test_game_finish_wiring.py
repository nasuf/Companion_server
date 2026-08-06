"""完局伴聊接线: LLM → 消息 → 记忆 的顺序与降级.

这段最容易出错的不是逻辑而是**时序**: LLM 要跑几秒, 而记忆写入原本是与聊天副作用
并发 fire 的另一个后台任务 —— 并发的话记忆几乎必然在 worth_remembering 落库之前
就写完了, 那个字段等于白算。
"""

from __future__ import annotations

import inspect

from app.services.games import native


class TestOrdering:
    def test_finished_memory_sync_is_not_fired_concurrently(self):
        """完局的 memory sync 不能和聊天副作用并发 —— 会读不到 LLM 写的笔记."""
        src = inspect.getsource(native.handle_event)
        # 只有 aborted 走并发 fire; finished 由副作用任务串行触发
        assert 'if event_type == "game_aborted":' in src
        assert 'fire_background(sync_session_memory' in src
        # 不应再有同时覆盖 finished 的并发 fire
        assert 'event_type in {"game_finished", "game_aborted"}:\n            fire_background(sync_session_memory' not in src

    def test_side_effects_runs_llm_before_persisting_reply(self):
        """先拿到 LLM 结果再落消息, 消息只写一次 —— 不做"先发再改"."""
        src = inspect.getsource(native._persist_chat_side_effects)
        assert src.index("_llm_finish_reply") < src.index("_persist_reply_to_chat_if_needed")

    def test_side_effects_triggers_memory_sync_for_finished(self):
        src = inspect.getsource(native._persist_chat_side_effects)
        assert "sync_session_memory" in src

    def test_memory_sync_runs_even_if_reply_persist_failed(self):
        """记忆和聊天消息是两件独立的事, 一个挂了不该连累另一个."""
        src = inspect.getsource(native._persist_chat_side_effects)
        # sync 在 for-retry 循环之外, 不被 break/except 短路
        after = src[src.index("sync_session_memory") - 400:]
        assert "try:" in after


class TestAbortedPathUnchanged:
    def test_aborted_does_not_call_the_llm(self):
        """745 局里只有 233 局完局, 中位时长 11-85 秒 —— 给秒退的局生成走心复盘
        本身就是错的, 真朋友不会为你点开又关掉说一段话。"""
        src = inspect.getsource(native._persist_chat_side_effects)
        assert 'event_type == "game_finished" and reply' in src


class TestReplayIdempotence:
    """客户端会重投同一个 game_finished (网络抖动 / outbox 重发)."""

    def test_llm_is_not_called_again_on_replay(self):
        """无门控的话每次重投都白调一次 LLM, 而且两次产出不同 —— 用户会看到同一
        局出现措辞不一样的两条伴聊。"""
        src = inspect.getsource(native._persist_chat_side_effects)
        assert "already_written" in src
        assert "not already_written" in src

    def test_replay_path_only_syncs_memory_for_aborted(self):
        """完局由副作用任务串行触发; 两边都 fire 会并发跑两次, 并发那次读不到笔记."""
        src = inspect.getsource(native.handle_event)
        idx = src.index("_ensure_idempotent") if "_ensure_idempotent" in src else 0
        assert idx >= 0
        ensure = inspect.getsource(native._ensure_idempotent_side_effects)
        assert 'if event_type == "game_aborted":' in ensure


class TestNotePriority:
    def test_llm_note_wins_over_the_template(self):
        """LLM 那句是"回忆"口吻, 模板是"台账"口吻 —— 模板化正是记忆挤成一坨的来源."""
        src = inspect.getsource(native._remember_shared_experience)
        assert "remembered_note" in src
        assert src.index("if remembered_note:") < src.index("rarity_text")

    def test_note_alone_passes_the_notability_gate(self):
        src = inspect.getsource(native._remember_shared_experience)
        assert "not remembered_note and not rarity.is_notable" in src


class TestNoteStorage:
    def test_note_is_merged_not_overwriting_the_row(self):
        """整行覆盖会把并发写入的 memory_sync 状态冲掉."""
        src = inspect.getsource(native._attach_remembered_note)
        assert "||" in src and "jsonb_build_object" in src

    def test_note_length_is_capped(self):
        src = inspect.getsource(native._attach_remembered_note)
        assert "note[:200]" in src


class TestScheduleColouring:
    def test_agent_state_never_blocks_the_game(self):
        """作息只染色不设卡.

        用户点开游戏就是现在想玩, 以"我在睡觉"为由让他等, 体感是"我想找你玩你不理
        我" —— 对陪伴产品这是最糟的失败。
        """
        src = inspect.getsource(native._agent_state_text)
        for blocking in ("return None  # sleeping", "raise", "sleep("):
            assert blocking not in src

    def test_agent_state_failure_degrades_to_empty(self):
        src = inspect.getsource(native._agent_state_text)
        assert "except Exception:" in src
        assert 'return ""' in src
