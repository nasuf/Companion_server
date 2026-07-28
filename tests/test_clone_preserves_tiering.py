"""克隆必须原样带走分层, 且不带走衰减进度.

模板是每个新用户的源头: H5 登录时 `ensure_default_agent_for_user` 把模板的记忆
整份复制过去。所以模板的分层就是全体新用户的分层 —— 这条链路上任何一处退化都是
静默的, 不会报错, 只会让新用户拿到一份错误分层的记忆。

两个方向都要钉住:

    level / importance 必须复制    否则新用户的分层跟模板对不上, 人设分层白做
    衰减进度必须**不**复制         新号从零开始计时; 复制的话相当于继承了模板
                                   积累的闲置时长, 一上来就在半路上
"""

from __future__ import annotations

from app.services.agent_template.clone import _MEMORY_COPY_FIELDS


def test_tiering_travels_with_the_clone():
    """人设分层是在模板上做的, 靠这两列传给每个新用户。"""
    assert "level" in _MEMORY_COPY_FIELDS
    assert "importance" in _MEMORY_COPY_FIELDS


def test_provenance_travels_with_the_clone():
    """provenance 决定后续维护任务怎么对待这条记忆 (整合排除人设、hygiene 排除
    摘要)。丢了它, 克隆出来的记忆会被当成来历不明的旧数据。"""
    assert "provenance" in _MEMORY_COPY_FIELDS


def test_decay_progress_does_not_travel_with_the_clone():
    """新号的衰减时钟从建号那一刻起算。

    复制 value_updated_at 会让克隆继承模板积累的闲置时长 —— 模板放了半年, 新用户
    第一天就拿到一份"半年没人碰过"的记忆, 首次兜底扫描直接把它们打下去。
    复制 current_score 同理: 那是模板的使用历史, 跟这个新用户无关。
    """
    for field in ("currentScore", "valueUpdatedAt"):
        assert field not in _MEMORY_COPY_FIELDS, (
            f"{field} 被复制了 —— 新号会继承模板的衰减进度"
        )


def test_mention_count_travels_but_is_harmless():
    """mentionCount 会被复制。它只喂旧 cron 的频率因子, 而效用值现在由
    current_score 承载 (不复制) —— 所以继承一个计数不会让新号的记忆凭空显得
    "常被提起"。这条测试是把这个判断写下来, 免得下次有人纠结。"""
    assert "mentionCount" in _MEMORY_COPY_FIELDS
