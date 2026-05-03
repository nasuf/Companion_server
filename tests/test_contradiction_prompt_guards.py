"""Regression: contradiction detection + reply prompt 必须含 anti-hallucination
+ anti-common-sense-conflict 守门.

生产 bug 复现 (2026-05-03 trace 019decdc):
  用户 turn 1: "明天跟我一起去钓鱼?"
  AI turn 1: 用常识反问 "诶等等明天不是你生日吗? 你怎么突然想跑去钓鱼了?"
  用户 turn 2: "是啊就是我生日所以想邀请你一起去钓鱼"
  AI turn 2: "...是去**上次说的那个水库**, 还是换了个新地方?" ← hallucinate

根因 2 层:
1. memory.contradiction_detection prompt 让 LLM 用**常识推断**造矛盾:
   "生日通常不去钓鱼" 当 conflict, 但 L1 里没这条 → 误触发 spec §4 矛盾流程
2. memory.contradiction_reply prompt 没禁止编造细节, LLM 自由发挥编"水库"

修: 两个 prompt 都加严格反例 + 显式禁区. 本测试守门防 future 编辑漏掉.
"""

from __future__ import annotations


def test_contradiction_detection_forbids_common_sense_conflicts():
    """contradiction_detection prompt 必须明确禁止"用常识推断造矛盾".

    防 LLM 拿 "生日通常不去钓鱼" 这种 social common sense 当 L1 conflict 触发
    spec §4 矛盾流程 (生产 bug 2026-05-03)."""
    from app.services.prompting.defaults import MEMORY_CONTRADICTION_DETECTION_PROMPT

    assert "常识" in MEMORY_CONTRADICTION_DETECTION_PROMPT, (
        "prompt 必须显式提到'常识'禁区, 否则 LLM 拿社会常识造虚假 conflict"
    )
    # 必须有反例: 生日 + 钓鱼 不矛盾 (生产 bug 直接 repro)
    assert "钓鱼" in MEMORY_CONTRADICTION_DETECTION_PROMPT or "邀请" in MEMORY_CONTRADICTION_DETECTION_PROMPT, (
        "必须有反例覆盖 '生日 + 邀请活动' 类场景, 防 LLM 误判"
    )
    # 必须明示问号 (邀请) 通常不矛盾
    assert "问号" in MEMORY_CONTRADICTION_DETECTION_PROMPT or "邀请" in MEMORY_CONTRADICTION_DETECTION_PROMPT


def test_contradiction_detection_requires_literal_conflict():
    """conflict 必须是 L1 字面事实 vs 用户字面事实 同维度对立, 不允许 LLM
    自由解读不同维度信息为冲突."""
    from app.services.prompting.defaults import MEMORY_CONTRADICTION_DETECTION_PROMPT

    # "字面" 关键词 (字面事实, 字面对立) 必须在 prompt 里
    assert "字面" in MEMORY_CONTRADICTION_DETECTION_PROMPT, (
        "必须强调'字面'对立, 防 LLM 用语义近似/含义推断造矛盾"
    )
    # "同一类属性" / "同维度" 类约束
    assert any(kw in MEMORY_CONTRADICTION_DETECTION_PROMPT for kw in ("同一类属性", "同维度", "同一类")), (
        "必须约束'同一类属性'对立 (住址 vs 住址), 不能跨维度对比"
    )


def test_contradiction_reply_forbids_fabricated_details():
    """contradiction_reply prompt 必须禁止编造未提及的细节 (生产 bug 复现:
    LLM 编了'上次说的那个水库', 用户从没说过)."""
    from app.services.prompting.defaults import MEMORY_CONTRADICTION_REPLY_PROMPT

    assert "编" in MEMORY_CONTRADICTION_REPLY_PROMPT, (
        "prompt 必须含'不能编 / 不许编'类约束"
    )
    # 必须显式提"上次说的 X" 这种隐式记忆词的禁区
    assert "上次" in MEMORY_CONTRADICTION_REPLY_PROMPT, (
        "必须显式禁'上次说的 X' 这种带入未提及细节的措辞 (生产 bug 直接 repro)"
    )


def test_contradiction_reply_warns_on_short_context():
    """对话上下文是'(无)' 或很短时, 必须强调只基于当前消息展开 — 防 LLM
    在缺历史时凭空编造."""
    from app.services.prompting.defaults import MEMORY_CONTRADICTION_REPLY_PROMPT

    assert any(kw in MEMORY_CONTRADICTION_REPLY_PROMPT for kw in ("(无)", "（无）", "上下文是")), (
        "必须显式 cover '上下文是空' 的边缘 case, 防 LLM 凭空补'以前 X'"
    )
