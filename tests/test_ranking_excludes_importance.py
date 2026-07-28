"""排序公式里不带 importance —— 这是量出来的结论, 不是口味.

importance 看起来天经地义该参与排序 (Generative Agents 给它 2 倍权重), 所以很容易
被"顺手加回去"。这个文件记录为什么在**我们的数据上**不该加:

    2026-07-28 实测 (1180 对配对, 103 对有人工/模型判定):
      仅相似度              有用记忆平均排名 0.407  (随机 0.486, 完美 0.091)
      相似度 × importance   0.495  —— 比随机还差
      相似度 × importance²  0.507  —— 更差
      相似度 + 0.2×imp      0.428  —— 仍不如纯相似度

原因不是原理错, 是我们的 importance 坏了: 56% 的值挤在 0.84-0.87, 而处在高位的
恰恰是建号人设 —— 那批记忆的检索有用率只有 11-20%, 远低于聊天中学到的 L2 的
29-37%。于是 importance 在我们这里跟"有用"是**反相关**的, 乘进去等于主动帮倒忙。

要推翻这个结论, 重跑 evals/memory_lifecycle/importance_signal.py, 而不是改这里。
存量人设重新分层之后值得再测一次。
"""

from __future__ import annotations

import inspect

from app.services.memory.retrieval import ranking, relevance


def test_display_score_does_not_multiply_by_importance():
    """禁的是**乘法因子**, 不是所有对 importance 的引用。

    它还有一处合法用途: 给 L1 事实设时间新鲜度下限 —— 姓名不会因为半年没提就
    过期。那是定向的、有据的, 跟"无差别乘一遍"是两回事。
    """
    body = inspect.getsource(relevance.compute_display_score).split('"""')[-1]
    for pattern in ("* importance", "importance *", "* imp)", "importance**"):
        assert pattern not in body, (
            f"display_score 里出现了 {pattern} —— importance 又被当成乘法因子。"
            "实测它与有用性反相关, 先跑 "
            "evals/memory_lifecycle/importance_signal.py 拿出反证。"
        )


def test_l1_facts_keep_their_freshness_floor():
    """摘掉乘法因子不能顺手把这个也摘了。

    曾经出过的 bug: L1 身份事实的新鲜度掉到 0.4, 被新鲜的琐事盖过去。姓名半年
    没被提起, 不代表它过期了。
    """
    from datetime import UTC, datetime, timedelta

    from app.services.memory.retrieval.relevance import compute_display_score

    long_ago = datetime.now(UTC) - timedelta(days=400)
    l1 = compute_display_score(
        importance=0.9, last_accessed_at=long_ago, similarity=0.7,
    )
    ordinary = compute_display_score(
        importance=0.6, last_accessed_at=long_ago, similarity=0.7,
    )
    assert l1 > ordinary, "L1 事实的新鲜度下限没生效"


def test_ranking_does_not_multiply_by_importance():
    source = inspect.getsource(ranking.rank_memory_candidate)
    body = source.split('"""')[-1]
    for pattern in ("* importance", "* imp", "importance *"):
        assert pattern not in body, f"排序里出现了 {pattern}"


def test_display_score_still_uses_freshness_and_similarity():
    """摘掉 importance 不等于把公式掏空 —— 另外两个因子仍然在用。"""
    from app.services.memory.retrieval.relevance import compute_display_score
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    fresh = compute_display_score(
        importance=0.5, last_accessed_at=now, similarity=0.8,
    )
    stale = compute_display_score(
        importance=0.5, last_accessed_at=now - timedelta(days=800), similarity=0.8,
    )
    assert fresh > stale, "时间新鲜度不起作用了"

    weak = compute_display_score(
        importance=0.5, last_accessed_at=now, similarity=0.3,
    )
    assert fresh > weak, "相似度不起作用了"


def test_importance_does_not_scale_the_score_of_recent_memories():
    """新鲜记忆本来就在新鲜度上限之上, 下限用不着 —— 此时 importance 应当完全
    不影响得分。这一条和上面的下限测试合起来划清了边界: 定向兜底可以, 无差别
    加权不行。"""
    from datetime import UTC, datetime

    from app.services.memory.retrieval.relevance import compute_display_score

    now = datetime.now(UTC)
    low = compute_display_score(importance=0.1, last_accessed_at=now, similarity=0.8)
    high = compute_display_score(importance=0.99, last_accessed_at=now, similarity=0.8)
    assert low == high
