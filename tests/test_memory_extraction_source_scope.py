"""类目归一化必须按 source 走对应的类目表, 以及"抽取没跑"不得推进水位线.

两个都是静默失效型缺陷 —— 不报错, 只是记忆悄悄变形或消失.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.normalization import normalize_memory_category
from app.services.memory.taxonomy import allowed_sub_categories


class TestSourceScopedTaxonomy:
    @pytest.mark.asyncio
    async def test_ai_only_subcategory_survives_for_ai_side(self):
        """「生活/交互」只存在于 AI 表. 不传 source 会按 user 解析 → 打成「其他」,
        而抽取 prompt 专门要求 AI 用这个子类记录关系里程碑."""
        res = await normalize_memory_category(
            main_category="生活", sub_category="交互",
            legacy_type=None, summary="我和用户第一次聊到深夜", source="ai",
        )
        assert (res.main_category, res.sub_category) == ("生活", "交互")

    @pytest.mark.asyncio
    async def test_same_subcategory_falls_back_for_user_side(self):
        """用户表里没有「交互」, 落「其他」是正确行为, 不是 bug."""
        res = await normalize_memory_category(
            main_category="生活", sub_category="交互",
            legacy_type=None, summary="用户和朋友聊到深夜", source="user",
        )
        assert res.sub_category != "交互"

    @pytest.mark.asyncio
    async def test_default_source_stays_user_for_back_compat(self):
        """老调用方不传 source, 行为必须跟以前一模一样."""
        explicit = await normalize_memory_category(
            main_category="身份", sub_category="姓名",
            legacy_type=None, summary="用户叫小明", source="user",
        )
        implicit = await normalize_memory_category(
            main_category="身份", sub_category="姓名",
            legacy_type=None, summary="用户叫小明",
        )
        assert (implicit.main_category, implicit.sub_category) == (
            explicit.main_category, explicit.sub_category
        )

    def test_taxonomy_tables_actually_differ(self):
        """这个修复的前提: 两侧类目表确实不同. 若哪天合并了, 这个测试会提醒."""
        assert "交互" in allowed_sub_categories("生活", "ai")
        assert "交互" not in allowed_sub_categories("生活", "user")

    @pytest.mark.asyncio
    async def test_extraction_passes_its_side_as_source(self):
        """回归守卫: extraction 必须把 side 透传下去."""
        from app.services.memory.recording import extraction as ex

        captured: dict = {}

        async def fake_normalize(**kwargs):
            captured.update(kwargs)

            class _R:
                main_category, sub_category, legacy_type = "生活", "交互", "life"
            return _R()

        payload = {"memories": [{"summary": "我和用户聊到深夜",
                                 "main_category": "生活", "sub_category": "交互"}]}
        with patch.object(ex, "get_prompt_text", AsyncMock(return_value="{new_conversation}")), \
             patch.object(ex, "get_chat_model", lambda: object()), \
             patch.object(ex, "invoke_json", AsyncMock(return_value=payload)), \
             patch("app.services.memory.normalization.normalize_memory_category",
                   fake_normalize):
            await ex.extract_memories("我和用户聊到深夜", side="ai")

        assert captured.get("source") == "ai"

    @pytest.mark.asyncio
    async def test_prompt_summary_field_is_translated_to_content(self):
        """The extraction prompt still emits a `summary` field because its rules
        are written around that name; the stored column is `content`.
        extract_memories is the single translation point — downstream code that
        keeps reading `summary` would silently get empty memory text."""
        from app.services.memory.recording import extraction as ex

        async def fake_normalize(**kwargs):
            class _R:
                main_category, sub_category, legacy_type = "偏好", "饮食喜好", "preference"
            return _R()

        payload = {"memories": [{"summary": "用户喜欢手冲咖啡",
                                 "main_category": "偏好", "sub_category": "饮食喜好"}]}
        with patch.object(ex, "get_prompt_text", AsyncMock(return_value="{new_conversation}")), \
             patch.object(ex, "get_chat_model", lambda: object()), \
             patch.object(ex, "invoke_json", AsyncMock(return_value=payload)), \
             patch("app.services.memory.normalization.normalize_memory_category",
                   fake_normalize):
            result = await ex.extract_memories("user: 我爱手冲咖啡", side="user")

        mem = result["memories"][0]
        assert mem["content"] == "用户喜欢手冲咖啡"
        assert "summary" not in mem


class TestImportanceCoercion:
    """LLM 的 importance 字段类型不可靠, 而它现在参与更早的判断 (身份兜底).

    坏值若抛异常会被上层当成抽取失败 → 水位线卡住 → 整批消息反复重试同一条
    坏数据, 永远出不去.
    """

    @pytest.mark.parametrize("raw,want", [
        ("0.8", 0.8), (0.8, 0.8), (1, 1.0),
        (None, 0.5), ("abc", 0.5), ("", 0.5), ([], 0.5),
        (-1, 0.0), (2.5, 1.0),
        (float("nan"), 0.5), (float("inf"), 0.5), (float("-inf"), 0.5),
    ])
    def test_coerces_without_raising(self, raw, want):
        from app.services.memory.recording.pipeline import _coerce_importance

        assert _coerce_importance(raw) == want

    def test_bad_value_lands_in_l2_not_dropped(self):
        """退回值必须落 L2: 抽取已判定这条值得记, 不该因字段格式丢整条."""
        from app.services.memory.recording.pipeline import _coerce_importance

        assert 0.50 <= _coerce_importance("垃圾") < 0.85


class TestWatermarkHeldWhenExtractionDidNotRun:
    @pytest.mark.asyncio
    async def test_prompt_disabled_reports_an_error_not_an_empty_result(self):
        """停用模板是运维状态, 不是"抽完了没内容".

        少了这个标记, 调用方会推进水位线, 这批消息永不重试 —— admin 临时关一下
        抽取模板就等于静默丢掉那段时间的全部记忆.
        """
        from app.services.memory.recording import extraction as ex
        from app.services.prompting.store import PromptDisabledError

        with patch.object(ex, "get_prompt_text",
                          AsyncMock(side_effect=PromptDisabledError("disabled"))):
            result = await ex.extract_memories("随便一句话", side="user")

        assert result["memories"] == []
        assert result["_extraction_error"] is True
        assert result["_extraction_error_kind"] == "prompt_disabled"

    @pytest.mark.asyncio
    async def test_pipeline_raises_so_the_caller_holds_the_watermark(self):
        from app.services.memory.recording import pipeline as pl
        from app.services.memory.recording.pipeline import (
            MemoryExtractionError,
            process_memory_pipeline,
        )

        disabled = {
            "memories": [], "entities": [], "preferences": [], "topics": [],
            "_extraction_error": True, "_extraction_error_kind": "prompt_disabled",
        }
        with patch.object(pl, "resolve_workspace_id", AsyncMock(return_value="ws")), \
             patch.object(pl, "should_extract_memory", lambda _: True), \
             patch.object(pl, "should_memorize", AsyncMock(return_value=True)), \
             patch.object(pl, "extract_memories", AsyncMock(return_value=disabled)):
            with pytest.raises(MemoryExtractionError) as exc:
                await process_memory_pipeline(
                    user_id="u", new_conversation="话", side="user",
                )
        # 异常文本要能区分成因: LLM 失败会自愈, 模板停用要人去开回来.
        assert "prompt_disabled" in str(exc.value)

    @pytest.mark.asyncio
    async def test_genuine_empty_extraction_still_advances(self):
        """抽取真的跑了但没内容 → 正常返回空列表, 水位线该推进."""
        from app.services.memory.recording import pipeline as pl
        from app.services.memory.recording.pipeline import process_memory_pipeline

        empty = {"memories": [], "entities": [], "preferences": [], "topics": []}
        with patch.object(pl, "resolve_workspace_id", AsyncMock(return_value="ws")), \
             patch.object(pl, "should_extract_memory", lambda _: True), \
             patch.object(pl, "should_memorize", AsyncMock(return_value=True)), \
             patch.object(pl, "extract_memories", AsyncMock(return_value=empty)):
            assert await process_memory_pipeline(
                user_id="u", new_conversation="话", side="user",
            ) == []
