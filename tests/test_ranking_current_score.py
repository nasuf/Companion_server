"""L2 动态分与 importance 都不再参与检索排序 —— 只有相似度和新鲜度参与.

原本 `rank_memory_candidate` 会拿 current_score (夜间 cron 算的 L2 动态分,
缺失时退回 importance) 去乘 display_score. 拿 423 条真实候选逐条判定"这条记忆
对回复有没有用"之后, 这个乘法被证明是有害的 —— 有用记忆落进 top3 的比例:

    只用相似度              44%
    相似度 × importance     29%
    随机洗牌                31%

按权重递减是单调的 (^0.5 落在 39%), 所以不是采样噪声. 机制是 importance 高的
多为「我是汉族」「我属猴」这类身份事实, 它们跟任何一句话都弱相关, 于是把真正
切题的那条挤下去.

设计上的理由: 检索已经按 level=[1,2] 过滤过了, 层级本身就是用这个分数划出来的.
再拿它当排序乘数等于同一个信号计两次权, 而实测这次重复计权是负收益. 分层决定
「哪些记忆有资格参与」, 相似度决定「这一句该用哪条」, 两件事不该共用一个分数.

⚠️ 证据边界: 那 1180 条真实候选里 current_score 全为 NULL (夜间 cron 没写过这些
行), 所以实测只覆盖了 importance 这条路径. current_score 是跟着一起去掉的 ——
它与 importance 同量纲同性质, 保留它等于埋一颗雷: cron 哪天开始写这一列, 排序
就会按上面测到的方式静默劣化.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.services.memory.retrieval.ranking import rank_memory_candidate


def _mem(**kwargs):
    base = {
        "id": "m1",
        "content": "用户在做一个副业项目",
        "importance": 0.8,
        "similarity": 0.7,
        "source": "user",
        "main_category": "生活",
        "sub_category": "工作",
        "last_accessed_at": datetime.now(UTC).isoformat(),
        "mention_count": 0,
    }
    base.update(kwargs)
    return base


_QUERY = "副业项目怎么样了"


def test_current_score_no_longer_moves_the_ranking():
    decayed, _ = rank_memory_candidate(_mem(current_score=0.4), _QUERY)
    fresh, _ = rank_memory_candidate(_mem(current_score=None), _QUERY)
    assert decayed == fresh


def test_importance_no_longer_moves_the_ranking():
    low, _ = rank_memory_candidate(_mem(importance=0.3), _QUERY)
    high, _ = rank_memory_candidate(_mem(importance=0.9), _QUERY)
    assert low == high


def test_bad_current_score_still_does_not_raise():
    """current_score 已不参与打分, 但脏值不能把整条检索炸掉."""
    baseline, _ = rank_memory_candidate(_mem(), _QUERY)
    garbage, _ = rank_memory_candidate(_mem(current_score="oops"), _QUERY)
    assert garbage == baseline


def test_similarity_still_drives_the_ranking():
    """去掉两个乘数之后, 相似度必须仍然是有效的排序信号 —— 否则就不是"去掉
    噪声", 而是把排序整个弄平了."""
    weak, _ = rank_memory_candidate(_mem(similarity=0.55), _QUERY)
    strong, _ = rank_memory_candidate(_mem(similarity=0.85), _QUERY)
    assert strong > weak


def test_l1_identity_still_survives_a_long_silence():
    """去掉 importance 后仍要保住原来的护栏: 用户半年没提自己的名字, 身份记忆
    不能被新鲜的琐事压下去. 这里靠的是 L1 的新鲜度下限, 不是 importance 乘数."""
    long_ago = datetime(2025, 1, 1, tzinfo=UTC).isoformat()
    identity, _ = rank_memory_candidate(
        _mem(content="用户叫小明", importance=0.95, similarity=0.75,
             main_category="身份", sub_category="姓名",
             last_accessed_at=long_ago),
        "我叫什么名字",
    )
    trivia, _ = rank_memory_candidate(
        _mem(content="用户昨天吃了火锅", importance=0.4, similarity=0.55,
             main_category="生活", sub_category="饮食"),
        "我叫什么名字",
    )
    assert identity > trivia
