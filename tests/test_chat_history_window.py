"""历史窗口: 让 token 预算成为唯一的收口条件.

生产事故 2026-07-22 (conv cec9b75a): 用户和 AI 聊了十几分钟 MBTI, 52 条消息
之后再问「这下能判定其他的MBTI了吗」, AI 答「我之前没研究过MBTI具体分类哎」,
用户指出「你说了」, AI 回「我真没说过呀 会不会是你记错啦？」并把锅甩给平台.

trace 显示那一刻主回复 prompt 里的历史只覆盖 7 分钟 30 条, ISFJ 出现 0 次 ——
`take=30` 这个行数上限先于 4000 token 预算生效, 实测只用掉预算的 17%.
"""

from __future__ import annotations

from app.services.chat.prompt_builder import (
    _coalesce_bubbles,
    build_chat_messages,
)


def _msg(role: str, content: str, created: str | None = None) -> dict:
    return {"role": role, "content": content, "createdAt": created}


class TestCoalesceBubbles:
    def test_consecutive_same_role_merge_into_one(self):
        """一次回复拆出的气泡是同一次发言, 不该在历史里显示成连说三次."""
        out = _coalesce_bubbles([
            _msg("user", "在吗"),
            _msg("assistant", "在的"),
            _msg("assistant", "怎么啦"),
            _msg("assistant", "刚忙完"),
            _msg("user", "没事"),
        ])
        assert [m["role"] for m in out] == ["user", "assistant", "user"]
        assert out[1]["content"] == "在的 怎么啦 刚忙完"

    def test_keeps_first_timestamp_of_the_group(self):
        """时间前缀对齐这轮发言的起点, 而不是最后一个气泡."""
        out = _coalesce_bubbles([
            _msg("assistant", "第一句", "2026-07-22T10:19:44"),
            _msg("assistant", "第二句", "2026-07-22T10:19:46"),
        ])
        assert out[0]["createdAt"] == "2026-07-22T10:19:44"

    def test_alternating_roles_unchanged(self):
        rows = [_msg("user", "a"), _msg("assistant", "b"), _msg("user", "c")]
        assert [m["content"] for m in _coalesce_bubbles(rows)] == ["a", "b", "c"]

    def test_empty_bubbles_do_not_add_separators(self):
        out = _coalesce_bubbles([
            _msg("assistant", "有内容"),
            _msg("assistant", ""),
        ])
        assert out[0]["content"] == "有内容"

    def test_does_not_mutate_input(self):
        rows = [_msg("assistant", "一"), _msg("assistant", "二")]
        _coalesce_bubbles(rows)
        assert rows[0]["content"] == "一"


class TestBudgetIsTheOnlyLimit:
    def test_coalescing_fits_more_turns_in_the_same_budget(self):
        """同样预算下, 合并气泡后能装下更多轮 —— 这正是拟人化拆句的代价所在."""
        split, merged = [], []
        for i in range(60):
            split.append(_msg("user", f"用户第{i}句"))
            merged.append(_msg("user", f"用户第{i}句"))
            for j in range(3):  # AI 每轮 3 个气泡
                split.append(_msg("assistant", f"回复{i}-{j}"))
                merged.append(_msg("assistant", f"回复{i}-{j}"))

        got = build_chat_messages("sys", split, token_budget=300)
        user_turns = sum(1 for m in got if m["role"] == "user")
        # 合并前每轮占 4 行, 合并后占 2 行; 同预算下轮数应该明显更多.
        assert user_turns >= 8, f"只装下 {user_turns} 轮"

    def test_budget_still_caps_长对话(self):
        rows = [_msg("user", "字" * 400) for _ in range(50)]
        got = build_chat_messages("sys", rows, token_budget=200)
        assert len(got) < 51  # system + 少量历史

    def test_always_keeps_at_least_the_latest_message(self):
        rows = [_msg("user", "字" * 5000)]
        got = build_chat_messages("sys", rows, token_budget=10)
        assert len(got) == 2
        assert got[0]["role"] == "system"

    def test_system_prompt_stays_first(self):
        rows = [_msg("user", "a"), _msg("assistant", "b")]
        got = build_chat_messages("SYS", rows)
        assert got[0] == {"role": "system", "content": "SYS"}


def test_fetch_limit_exceeds_what_the_budget_can_hold():
    """行数上限必须宽于预算, 否则预算永远不生效 —— 这是事故的根因.

    生产实测均值约 22 token/条, 4000 预算 ≈ 180 条. 上限低于这个数, 收口的就
    又变回行数了.
    """
    from app.services.chat.orchestrator import _HISTORY_FETCH_LIMIT
    from app.services.prompts.system_prompts import CHAT_HISTORY_TOKEN_BUDGET

    avg_tokens_per_message = 22  # 生产实测 (679 token / 30 条)
    assert _HISTORY_FETCH_LIMIT * avg_tokens_per_message >= CHAT_HISTORY_TOKEN_BUDGET


def test_mbti_incident_would_now_be_in_window():
    """回归: 事故当时 ISFJ 那句在 52 条之前, 约 1200 token, 应落在预算内."""
    rows = [_msg("assistant", "ISFJ大多很细心靠谱")]
    # 之后 52 条消息, 按生产实测每条约 22 token
    rows += [_msg("user" if i % 3 == 0 else "assistant", "字" * 15) for i in range(52)]
    rows.append(_msg("user", "所以，这下能判定其他的MBTI了吗"))

    got = build_chat_messages("sys", rows, token_budget=4000)
    joined = "".join(m["content"] for m in got)
    assert "ISFJ" in joined
