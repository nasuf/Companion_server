"""Guards that stop unretrievable memories from being written or cloned.

Background (2026-08): a memory over MAX_MEMORY_TOKENS_PER_ITEM is skipped
whole by select_context — it sits in the table, counts in the stats, and no
conversation ever uses it. The 2026-07-30 fix closed the *generation* side,
but three write paths still had no check at all (public PATCH, repair edit,
repair merge) and a fourth actively multiplied the damage: agent cloning
copies memory rows verbatim, so one dirty template became one dirty agent
per signup. These tests pin each of those shut.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.memory.retrieval.context_selector import (
    MAX_MEMORY_TOKENS_PER_ITEM,
    estimate_tokens,
    exceeds_injection_limit,
)


class TestExceedsInjectionLimit:
    """The shared yardstick. Every guard below must agree with select_context."""

    def test_agrees_with_the_selector_constant(self):
        just_under = "测" * 119
        assert estimate_tokens(just_under) <= MAX_MEMORY_TOKENS_PER_ITEM
        assert exceeds_injection_limit(just_under) is False

        clearly_over = "测" * 200
        assert estimate_tokens(clearly_over) > MAX_MEMORY_TOKENS_PER_ITEM
        assert exceeds_injection_limit(clearly_over) is True

    def test_cjk_punctuation_counts_as_ascii_not_as_a_han_character(self):
        """The trap that made a hand-rolled `len(content) > 120` check wrong.

        CJK punctuation (，。；) lives outside the \\u4e00-\\u9fff block, so it
        is billed at 0.25 token, not 1.5. A heavily punctuated sentence can run
        well past 120 *characters* while staying under the real cap — measuring
        by character count reports memories as broken that are perfectly fine.
        """
        # 132 chars, but 15 of them are punctuation → under the cap.
        heavily_punctuated = ("测" * 7 + "，") * 15 + "测" * 12
        assert len(heavily_punctuated) > 120
        assert exceeds_injection_limit(heavily_punctuated) is False

        # Same character count, no punctuation → over the cap.
        all_han = "测" * len(heavily_punctuated)
        assert exceeds_injection_limit(all_han) is True

    def test_empty_and_ascii_text_are_never_over(self):
        assert exceeds_injection_limit("") is False
        assert exceeds_injection_limit("a" * 500) is False


class TestRepairActionGuards:
    """repair_actions exists to *improve* memory quality — it was the last
    place that should be able to mint an unretrievable row."""

    def test_edit_and_merge_reject_oversized_content(self):
        from app.services.memory.repair_actions import (
            MemoryRepairActionError,
            _assert_within_injection_limit,
        )

        with pytest.raises(MemoryRepairActionError) as exc:
            _assert_within_injection_limit("测" * 200)
        assert exc.value.detail == "memory_content_too_long"
        assert exc.value.status_code == 400

    def test_normal_content_passes(self):
        from app.services.memory.repair_actions import _assert_within_injection_limit

        _assert_within_injection_limit("用户的妈妈叫王秀兰")  # must not raise

    def test_llm_merge_and_manual_merge_use_the_same_yardstick(self):
        """reconciliation refused over-long LLM merges long before the manual
        repair path checked anything; the two must not drift apart."""
        from app.services.memory.storage.reconciliation import _exceeds_injection_limit

        long_text = "测" * 200
        assert _exceeds_injection_limit(long_text) == exceeds_injection_limit(long_text)


class TestTemplatePromotionGuard:
    """The choke point: cloning copies rows verbatim, so a dirty template does
    not stay one bad agent — it becomes one per signup, forever."""

    @pytest.mark.asyncio
    async def test_refuses_to_promote_an_agent_with_oversized_memories(self, monkeypatch):
        from app.services.agent_template import registry

        monkeypatch.setattr(
            registry.db, "query_raw",
            AsyncMock(return_value=[{"content": "测" * 200}, {"content": "短记忆"}]),
        )
        execute = AsyncMock()
        monkeypatch.setattr(registry.db, "execute_raw", execute)

        with pytest.raises(ValueError, match="不能设为默认模板"):
            await registry.set_default_template_agent_id("agent-1")
        execute.assert_not_awaited()  # pointer must stay on the old template

    @pytest.mark.asyncio
    async def test_allows_promoting_a_clean_agent(self, monkeypatch):
        from app.services.agent_template import registry

        monkeypatch.setattr(
            registry.db, "query_raw",
            AsyncMock(return_value=[{"content": "我叫林昕"}, {"content": "我今年22岁"}]),
        )
        execute = AsyncMock()
        monkeypatch.setattr(registry.db, "execute_raw", execute)

        await registry.set_default_template_agent_id("agent-1")
        execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_clearing_the_pointer_skips_the_check(self, monkeypatch):
        """Unsetting the default must never be blocked — there is no agent to
        validate, and an admin clearing a bad pointer is the fix, not the bug."""
        from app.services.agent_template import registry

        query = AsyncMock()
        monkeypatch.setattr(registry.db, "query_raw", query)
        monkeypatch.setattr(registry.db, "execute_raw", AsyncMock())

        await registry.set_default_template_agent_id(None)
        query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_counts_only_unarchived_rows_of_that_agent(self, monkeypatch):
        from app.services.agent_template import registry

        captured: dict = {}

        async def _capture(sql, *args):
            captured["sql"] = sql
            captured["args"] = args
            return []

        monkeypatch.setattr(registry.db, "query_raw", _capture)
        assert await registry.count_oversized_memories("agent-9") == 0
        assert "is_archived = false" in captured["sql"]
        assert captured["args"] == ("agent-9",)


class TestSplitFailureIsObservable:
    """_split_for_storage is best-effort, not a gate: when the splitter cannot
    break a long single-sentence narrative it returns the original. That is a
    deliberate trade-off (losing the text is worse), but it used to happen
    silently, which is how the problem stayed invisible."""

    def test_unsplittable_oversized_content_is_logged(self, caplog):
        from types import SimpleNamespace

        from app.services.memory.storage import persistence

        # A long narrative with no full-width semicolon and no sentence break —
        # exactly the shape the splitter cannot help with.
        content = "大橘是我刚工作时在公司园区附近发现的一只瘦弱流浪小奶猫" * 6
        assert exceeds_injection_limit(content)

        taxonomy = SimpleNamespace(main_category="生活", sub_category="宠物")
        with caplog.at_level("WARNING"):
            pieces = persistence._split_for_storage(content, taxonomy)

        assert pieces == [content]  # unchanged — content is never dropped
        assert "SPLIT-FAILED" in caplog.text

    def test_short_content_is_passed_through_without_warning(self, caplog):
        from types import SimpleNamespace

        from app.services.memory.storage import persistence

        taxonomy = SimpleNamespace(main_category="身份", sub_category="姓名")
        with caplog.at_level("WARNING"):
            pieces = persistence._split_for_storage("我叫林昕", taxonomy)

        assert pieces == ["我叫林昕"]
        assert "SPLIT-FAILED" not in caplog.text
