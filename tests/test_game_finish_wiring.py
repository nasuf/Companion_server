"""完局伴聊接线: LLM → 消息 → 记忆 的顺序与降级.

这段最容易出错的不是逻辑而是**时序**: LLM 要跑几秒, 而记忆写入原本是与聊天副作用
并发 fire 的另一个后台任务 —— 并发的话记忆几乎必然在 worth_remembering 落库之前
就写完了, 那个字段等于白算。
"""

from __future__ import annotations

import inspect

from app.services.games import native


class TestOrdering:
    def test_memory_sync_is_never_fired_concurrently(self):
        """memory sync 不能和聊天副作用并发 —— 会读不到 LLM 写的笔记.

        中断局原来走并发 fire (那时它不调 LLM 所以没有笔记可读)。现在玩起来了的
        中断局也上 LLM 也可能产出笔记 —— 一局 420 步玩了 34 分钟的棋值得记, 跟它
        有没有下完无关 —— 所以两条路都改成串行。
        """
        src = inspect.getsource(native.handle_event)
        assert "fire_background(sync_session_memory" not in src

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


class TestAbortedPathGetsNarratedToo:
    def test_the_split_is_substance_not_completion(self):
        """分叉依据是"有没有玩起来", 不是"有没有下完".

        原来只对 game_finished 调 LLM, 于是"玩了很久但中途停"的局全落在硬编码上:
        实测这类局 88 个, 89% 有高光、85% 有 AI 决策快照, 而一局 420 步 34 分钟的
        数字合并收到的话跟一局 204 步的**一字不差**。
        """
        src = inspect.getsource(native._persist_chat_side_effects)
        # 只看两个 if/elif 的条件本身, 不看注释 —— 注释里会提到这段历史。
        conditions = [
            line for line in src.splitlines()
            if line.strip().startswith(("if _is_quick_exit", "elif "))
        ]
        assert conditions, "找不到分叉条件"
        for line in conditions:
            assert "game_finished" not in line, f"不该再按完局与否分叉: {line}"

    def test_quick_exits_still_skip_the_llm(self):
        """0 步 2 秒的空局没有素材, 两个模型实测都只会说同一句话."""
        src = inspect.getsource(native._persist_chat_side_effects)
        assert src.index("_quick_exit_reply") < src.index("_llm_finish_reply")


class TestReplayIdempotence:
    """客户端会重投同一个 game_finished (网络抖动 / outbox 重发)."""

    def test_llm_is_not_called_again_on_replay(self):
        """无门控的话每次重投都白调一次 LLM, 而且两次产出不同 —— 用户会看到同一
        局出现措辞不一样的两条伴聊。"""
        src = inspect.getsource(native._persist_chat_side_effects)
        assert "already_written" in src
        assert "not already_written" in src

    def test_replay_path_does_not_fire_memory_sync_concurrently(self):
        """重投路径也不能旁路串行化.

        原来中断局在这里补 fire 一次 (那时中断局不调 LLM, 没有笔记可读, 并发无害)。
        现在玩起来了的中断局也上 LLM 也可能产出 worth_remembering, 并发那次会读不到
        笔记。sync_session_memory 自带锁 + claim, 本身幂等, 所以交给副作用任务末尾
        那一次串行调用就够了。
        """
        ensure = inspect.getsource(native._ensure_idempotent_side_effects)
        assert "fire_background(sync_session_memory" not in ensure
        # 副作用任务仍会跑 sync —— 否则重投时记忆就彻底没人补了
        side_effects = inspect.getsource(native._persist_chat_side_effects)
        assert "await sync_session_memory" in side_effects


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
