"""V3 主动·话题源分类器 (proactive/topic_source.py) 单测.

分类器是 V3 的决策枢纽: 一次分错就走错 prompt → 完全跑错朋友感表达模式.
这里锁住:
  - 用户兴趣命中 → user_interest_match
  - AI 人设命中 → ai_persona_match
  - 都不命中但有质量内容 → socially_hot
  - 都不满足 → none
  - 内容黑名单硬拒 (八卦/事故/死讯/UI 残余)
  - 优先级: user > ai > socially_hot (确定性, 非概率抽签)
"""

from __future__ import annotations

from types import SimpleNamespace

from app.services.proactive.topic_source import (
    _extract_interests,
    _is_hot_quality,
    classify_topic_source,
)


class TestExtractInterests:
    def test_hooks_after_喜欢(self):
        assert "摄影" in _extract_interests("25岁上海白领, 喜欢摄影和露营")

    def test_hooks_after_爱好(self):
        assert "户外" in _extract_interests("爱好户外和阅读")

    def test_hooks_after_迷上(self):
        assert "腰乐队" in _extract_interests("最近迷上腰乐队")

    def test_splits_on_commas_and_slash(self):
        # 主要断言: "、和 ," 都能切出多个词
        terms = _extract_interests("喜欢摄影、露营、咖啡")
        assert "摄影" in terms and "露营" in terms and "咖啡" in terms

    def test_ignores_too_short_or_too_long(self):
        # "A" 只 1 字被过滤; 超过 12 字的一大段也被过滤
        terms = _extract_interests("喜欢A")
        assert "A" not in terms

    def test_empty_input(self):
        assert _extract_interests("") == []
        assert _extract_interests("完全无兴趣关键词的一段话") == []

    def test_dedup(self):
        # 同一词多次出现只保留一次
        terms = _extract_interests("喜欢摄影, 平常也爱好摄影, 迷上摄影")
        assert terms.count("摄影") == 1


class TestQualityGate:
    def test_blocks_death(self):
        assert not _is_hot_quality({"title": "小七龄童去世 热度259w", "snippet": "..."})

    def test_blocks_accident(self):
        assert not _is_hot_quality({"title": "韩安冉出车祸", "snippet": ""})

    def test_blocks_api_ui_junk(self):
        assert not _is_hot_quality({"title": "热榜聚合", "snippet": "keyword_pinyin xxx"})

    def test_blocks_politics(self):
        assert not _is_hot_quality({"title": "某地抗议", "snippet": ""})

    def test_allows_normal_content(self):
        assert _is_hot_quality({"title": "iPhone 17 首销", "snippet": "秒空"})
        assert _is_hot_quality({"title": "《漫长的季节 2》定档", "snippet": "..."})

    def test_rejects_empty_title(self):
        assert not _is_hot_quality({"title": "", "snippet": "some text"})


class TestClassifier:
    _AGENT = SimpleNamespace(
        background="苏州皮具制作师, 爱好独立音乐, 常做手工皮包",
        lifeOverview="",
        occupation="皮具制作师",
    )

    def test_user_interest_wins_when_matched(self):
        cls = classify_topic_source(
            trending_candidates=[
                {"title": "手机拍夜景技巧", "snippet": "摄影小白也能拍出电影感"},
                {"title": "iPhone 17 首销", "snippet": "秒空"},
            ],
            user_portrait="25岁上海白领, 喜欢摄影和露营",
            agent=self._AGENT,
        )
        assert cls.kind == "user_interest_match"
        # 应挑第一条 (title/snippet 命中"摄影"), 不是 iPhone 那条
        picked = cls.selected_candidate["title"] + cls.selected_candidate["snippet"]
        assert "摄影" in picked

    def test_ai_persona_wins_when_only_ai_matches(self):
        cls = classify_topic_source(
            trending_candidates=[
                {"title": "腰乐队新专辑发布", "snippet": "独立音乐十年沉淀"},
                {"title": "iPhone 17 首销", "snippet": "秒空"},
            ],
            user_portrait="25岁上海白领",  # 无摄影/露营等 AI 匹配之外的兴趣
            agent=self._AGENT,
        )
        assert cls.kind == "ai_persona_match"
        assert "腰乐队" in cls.selected_candidate["title"]

    def test_user_beats_ai_when_both_match(self):
        # 优先级: user > ai. 都有条命中时选 user 档
        cls = classify_topic_source(
            trending_candidates=[
                {"title": "腰乐队新专辑发布", "snippet": "独立音乐"},   # 命中 AI
                {"title": "手机拍夜景技巧", "snippet": "摄影"},         # 命中 user
            ],
            user_portrait="喜欢摄影",
            agent=self._AGENT,
        )
        assert cls.kind == "user_interest_match"

    def test_socially_hot_when_no_matches_but_quality_ok(self):
        cls = classify_topic_source(
            trending_candidates=[
                {"title": "iPhone 17 首销秒空", "snippet": "全国排队"},
            ],
            user_portrait="喜欢摄影",  # 不 match
            agent=self._AGENT,  # 不 match
        )
        assert cls.kind == "socially_hot"

    def test_none_when_all_candidates_blocked(self):
        cls = classify_topic_source(
            trending_candidates=[
                {"title": "小七龄童去世", "snippet": ""},
                {"title": "热榜聚合", "snippet": "keyword_pinyin"},
            ],
            user_portrait="",
            agent=self._AGENT,
        )
        assert cls.kind == "none"
        assert cls.selected_candidate is None

    def test_none_when_no_candidates(self):
        cls = classify_topic_source(
            trending_candidates=[],
            user_portrait="喜欢摄影",
            agent=self._AGENT,
        )
        assert cls.kind == "none"

    def test_none_when_agent_missing(self):
        # agent=None 应不炸
        cls = classify_topic_source(
            trending_candidates=[{"title": "iPhone 17", "snippet": "首销"}],
            user_portrait="",
            agent=None,
        )
        # user/ai 都无兴趣词但内容是 hot quality → socially_hot
        assert cls.kind == "socially_hot"

    def test_reason_populated(self):
        cls = classify_topic_source(
            trending_candidates=[{"title": "iPhone 17", "snippet": "首销"}],
            user_portrait="", agent=self._AGENT,
        )
        assert cls.reason  # 非空字符串
