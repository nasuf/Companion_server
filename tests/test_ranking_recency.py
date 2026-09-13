"""求近查询按 occur_time 事件新近度排序 (P3)。

失败形态 (evals/temporal_recall): "我最近一次去健身" 这类问最新的题, 一组语义
几乎相同、只有 occur_time 不同的事件 (三次面试/多次健身), 基础排序按 30/90 天
粗桶算新鲜度分不开同桶内的两次, 于是最新那条排不到前面。这里用连续的 occur_time
衰减补上分辨率, 只在求近查询上生效 (避免重蹈 importance 一刀切乘进排序的覆辙)。

用确定性单测证明机制 (无 embedding、无 select_context 的安全保护槽混淆) ——
temporal_recall 的 top-1 聚合被安全槽 (恐惧类记忆无条件占首槽) 盖住, 反映不出
纯排序的改善, 这里直接比 rank_score。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.services.memory.retrieval.ranking import (
    _is_recency_seeking_query,
    rank_memory_candidate,
)

_NOW = datetime.now(timezone.utc)


def _event(mem_id: str, text: str, occur_days_ago: int, sim: float = 0.62,
           last_accessed_days: int | None = None) -> dict:
    occ = _NOW - timedelta(days=occur_days_ago)
    la = _NOW - timedelta(days=occur_days_ago if last_accessed_days is None
                          else last_accessed_days)
    return {
        "id": mem_id, "content": text, "similarity": sim, "importance": 0.5,
        "occur_time": occ, "last_accessed_at": la, "created_at": la,
        "main_category": "生活", "sub_category": "运动", "source": "user",
    }


class TestRecencyQueryDetection:
    @pytest.mark.parametrize("q", [
        "我最近一次去健身是什么时候", "我这个月去过几次健身房",
        "最近一个月我面试过几家", "你现在在干嘛", "最新的消息是啥", "上次那家店",
    ])
    def test_recency_queries_detected(self, q):
        assert _is_recency_seeking_query(q) is True

    @pytest.mark.parametrize("q", [
        "两个多月前我去面的那家是做什么的",  # 指向具体过去, 要旧的那条, 绝不能当求近
        "去年夏天我们去哪玩了", "我喜欢什么颜色", "我老板叫什么名字",
    ])
    def test_non_recency_queries_not_detected(self, q):
        assert _is_recency_seeking_query(q) is False


class TestRecencyRanking:
    def test_most_recent_event_ranks_highest_among_siblings(self, monkeypatch):
        # 同相似度、同类目, 只有 occur_time 不同 —— 最新那条必须 rank_score 最高.
        # 显式把权重打开 (0.5), 因为 2026-09-13 起模块默认 = 0.0 (feature soft-disable):
        # temporal_recall v3 显示 P3 在真实候选压力下净负. 但**机制本身**仍是可用的,
        # 未来加了"L1 保护槽 / 查询意图分类"能修上述失败机制后可以重新激活权重, 这条
        # 测试守住的就是"权重开启时排序确实按 occur_time 拉开次序"这个基本机制.
        import app.services.memory.retrieval.ranking as ranking_mod
        monkeypatch.setattr(ranking_mod, "_RECENCY_BOOST_WEIGHT", 0.5)

        q = "我最近一次去健身是什么时候"
        cands = [
            _event("gym_1", "今天去健身房练了腿", 45),
            _event("gym_2", "又去健身了，练的背", 20),
            _event("gym_3", "昨天去健身房了，这次练胸", 2),
        ]
        scored = sorted(
            ((c["id"], rank_memory_candidate(c, q)[0]) for c in cands),
            key=lambda x: -x[1],
        )
        assert [cid for cid, _ in scored] == ["gym_3", "gym_2", "gym_1"]

    def test_default_weight_is_disabled(self):
        # 模块默认权重是 0 (feature soft-disable). 若哪天有人不加解释地改回非零,
        # 这条会挂 —— 提醒对方: 重启 P3 前必须先看 evals/temporal_recall/standard.py
        # 的证据链, 修好 A/B/C 三类失败机制, 或者提供新的正 delta 证据.
        from app.services.memory.retrieval.ranking import _RECENCY_BOOST_WEIGHT
        assert _RECENCY_BOOST_WEIGHT == 0.0, (
            "P3 求近排序权重非零 —— 见 evals/temporal_recall/standard.py 的证据链, "
            "重启前必须先修好 3 类失败机制或提供新的正 delta 证据"
        )

    def test_no_boost_on_non_recency_query(self):
        # 非求近查询: occur_time 不参与排序, 三条按基础分 (同相似度→约等), 不因
        # 事件新近被重排 —— 证明信号只在求近查询上生效, 不污染普通查询。
        q = "我健身一般都练哪些部位比较多"  # 无求近词
        # last_accessed 固定 5 天(新鲜度相同), 只让 occur_time 差 —— 隔离出
        # "事件新近"这一维: 非求近查询下 occur_time 不该参与, 分数应相等。
        s3 = rank_memory_candidate(
            _event("gym_3", "昨天去健身房了，这次练胸", 2, last_accessed_days=5), q)[0]
        s1 = rank_memory_candidate(
            _event("gym_1", "今天去健身房练了腿", 45, last_accessed_days=5), q)[0]
        assert abs(s3 - s1) < 1e-9

    def test_no_occur_time_means_no_boost(self):
        # 求近查询但候选没有 occur_time (人设偏好类记忆) → boost 不触发, 不炸
        q = "我最近一次去健身是什么时候"
        mem = {"id": "x", "content": "用户喜欢健身", "similarity": 0.6,
               "importance": 0.5, "main_category": "生活", "sub_category": "运动",
               "source": "user"}  # 无 occur_time
        score, reasons = rank_memory_candidate(mem, q)
        assert "求近:事件新近" not in reasons
        assert score > 0

    def test_recency_reason_recorded_only_when_fired(self):
        fired = rank_memory_candidate(
            _event("g", "昨天去健身房了，这次练胸", 1), "我最近一次去健身是什么时候")[1]
        assert "求近:事件新近" in fired
        not_fired = rank_memory_candidate(
            _event("g", "昨天去健身房了，这次练胸", 1), "我健身一般练哪些部位")[1]
        assert "求近:事件新近" not in not_fired
