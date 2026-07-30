"""后台上传 profile 文档建模板时, 超长字段要被识别出来.

这条路径 (`POST /admin/agent-templates/from-document`) 用 profile_override 直接喂
CharacterProfile, **完全绕过 LLM** —— 所以 character.generation 里收紧字数的约束对
它一点作用都没有。管理员在 txt 里写多长就是多长, 而超过检索单条上限的记忆存进去
之后任何对话都不会用到它, 他却只看到"模板创建成功"。

刻意不拦创建: 整份五维档案不该因为一个字段太长就作废。报给管理员, 由他决定改不改。
"""

from __future__ import annotations

from app.services.agent_template.document_import import oversized_profile_fields
from app.services.memory.recording.splitting import unsplittable_oversized
from app.services.memory.retrieval.context_selector import (
    MAX_MEMORY_TOKENS_PER_ITEM,
    estimate_tokens,
)

# 真正拆不动的形态: 整段就是一个句子, 没有句号也没有分号可切。
# (带句号的长叙事现在会被句级拆分救回来, 不再算"有问题" —— 见 splitting.py 的
#  2026-07-30 修订。所以这里不能拿它当样本, 否则测的是已经不存在的场景。)
_LONG_NARRATIVE = "那个下着冬雨的黄昏她在公司楼下等了我很久很久始终没有离开" * 8
_MULTI_FACT = "；".join(
    f"我的工作是第{i}项职责的详细说明内容占位文字若干这里再补一些字数" for i in range(1, 8)
)


class TestUnsplittableDetection:
    def test_long_single_narrative_is_reported(self):
        """单段叙事没有分隔符, 拆不动 —— 必须报出来."""
        assert estimate_tokens(_LONG_NARRATIVE) > MAX_MEMORY_TOKENS_PER_ITEM
        assert unsplittable_oversized(_LONG_NARRATIVE) == [_LONG_NARRATIVE]

    def test_splittable_multi_fact_is_not_reported(self):
        """能拆好的不该报.

        这是这个检查最容易做错的地方: 只按原文长度判断的话, 多事实拼接会被误报,
        而它转换时会被拆成几条正常记忆。噪音会让管理员忽略真正的问题。
        """
        assert estimate_tokens(_MULTI_FACT) > MAX_MEMORY_TOKENS_PER_ITEM
        assert unsplittable_oversized(_MULTI_FACT) == []

    def test_normal_text_is_clean(self):
        assert unsplittable_oversized("我在苏州长大, 大学读的是设计。") == []

    def test_empty_and_blank_are_safe(self):
        assert unsplittable_oversized("") == []
        assert unsplittable_oversized("   ") == []

    def test_reported_pieces_are_actually_oversized(self):
        """报出来的每一条都得真的超限, 否则就是误报."""
        for piece in unsplittable_oversized(_LONG_NARRATIVE):
            assert estimate_tokens(piece) > MAX_MEMORY_TOKENS_PER_ITEM


class TestProfileWalk:
    def test_finds_nested_list_entries_with_path(self):
        profile = {
            "identity": {"name": "小明"},
            "life_events": {"work": ["正常的一条经历", _LONG_NARRATIVE]},
        }
        found = oversized_profile_fields(profile)
        assert len(found) == 1
        path, piece = found[0]
        assert path == "life_events.work[1]", f"路径不对, 管理员定位不到: {path}"
        assert piece == _LONG_NARRATIVE

    def test_clean_profile_reports_nothing(self):
        profile = {
            "identity": {"name": "小明", "age": 28},
            "life_events": {"work": ["在设计公司做了三年视觉"], "travel": []},
        }
        assert oversized_profile_fields(profile) == []

    def test_non_string_leaves_do_not_crash(self):
        """profile 里混有 int/None/bool, 遍历不能炸."""
        profile = {"identity": {"age": 28, "pet": None, "married": False}}
        assert oversized_profile_fields(profile) == []

    def test_empty_profile(self):
        assert oversized_profile_fields({}) == []

    def test_multiple_hits_are_all_reported(self):
        profile = {
            "life_events": {"work": [_LONG_NARRATIVE]},
            "emotion_events": {"loss": [_LONG_NARRATIVE]},
        }
        paths = [p for p, _ in oversized_profile_fields(profile)]
        assert len(paths) == 2
        assert "life_events.work[0]" in paths
        assert "emotion_events.loss[0]" in paths
