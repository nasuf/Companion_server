"""RECORD_REQUEST 历史回归 phrase 清单 + prompt 结构守门.

背景: INTENT_UNIFIED_PROMPT 早期把 12 个具体 phrase 例子塞进 prompt 里防 LLM
误判 (commit af5e22a / 444ac7d 各打过一仗). prompt 越长 LLM 越当 lookup
table, 反而对未列出变体识别变差. 已收敛 prompt 至 3 行 + 4 个原型例子,
historical phrases 搬到本文件作:

1. **回归清单**: 想跑 LLM 集成验证时, 用 RECORD_REQUEST_MUST_HIT /
   RECORD_REQUEST_MUST_MISS 喂给真模型确认分类正确率.
2. **prompt 结构守门**: 单测验证 prompt 至少含 ontology hooks (设置/取消/改期 +
   disambiguation), 防 future 编辑误删核心信息让 LLM 退化.
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════
# 历史 phrase 清单 — 必须分类为 RECORD_REQUEST
# 每条括注是首次发现误判的 commit 或场景, 方便 future 反查.
# ═══════════════════════════════════════════════════════════════════

RECORD_REQUEST_MUST_HIT: tuple[tuple[str, str], ...] = (
    # 设置类
    ("提醒我喝水", "原型, 关键词级"),
    ("一分钟后提醒我喝水", "短期闹钟"),
    ("明天 8 点提醒我看升旗", "长期"),
    ("记得帮我盯着报告", "记得X"),
    ("X月X日是我生日", "重要日期"),
    ("下周二我面试", "无显式'提醒我'但有事项+时间"),
    ("每周一帮我盯着Y", "周期性"),
    ("每月 1 号交房租", "周期性 月"),
    # 取消类
    ("算了别提醒了", "af5e22a — 误归 RECORD_REQUEST + sub='计划查询'"),
    ("不用提醒X了", "显式取消"),
    ("取消那个提醒", "显式取消"),
    ("我吃过了", "隐式取消 (用户已做事项)"),
    ("已经做了", "隐式取消"),
    # 改期类
    ("把明天的提醒改到后天", "显式改期"),
    ("推迟到 9 点", "显式改期"),
    ("提前到 8 点", "显式改期"),
)


# ═══════════════════════════════════════════════════════════════════
# Disambiguation — 这些**不该**归 RECORD_REQUEST
# (容易跟"记录请求"混淆但语义不同的句子)
# ═══════════════════════════════════════════════════════════════════

RECORD_REQUEST_MUST_MISS: tuple[tuple[str, str, str], ...] = (
    # (phrase, expected_intent, why)
    ("你还记得我之前说的X吗", "日常交流", "普通记忆查询, 不是让 AI 记, 也不是久远记忆"),
    ("你记得我家在哪吗", "日常交流", "stable fact 查询 (走记忆路径), 不是 RECORD_REQUEST"),
    ("你明天有空吗", "计划查询", "问 AI 自己日程, 不是设提醒"),
    ("你周末忙吗", "计划查询", "问 AI 日程"),
    ("我累了想睡了", "终结意图", "结束对话, 不是设提醒"),
    ("你在干嘛", "询问当前状态", "问 AI 当前活动"),
)


def test_current_state_fast_path_is_narrow():
    """常见当前状态问句走本地 fast path；带未来/过去时间的仍交给 LLM。"""
    from app.services.chat.intent_dispatcher import (
        detect_current_state_fast_path,
        is_explicit_current_state_query,
    )

    assert detect_current_state_fast_path("你在干嘛呢")
    assert detect_current_state_fast_path("忙啥？")
    assert not detect_current_state_fast_path("你明天在干嘛")
    assert not detect_current_state_fast_path("你刚才在干嘛")

    assert is_explicit_current_state_query("忙吗")
    assert is_explicit_current_state_query("你最近怎么样")
    assert is_explicit_current_state_query("你心情怎么样")
    assert not is_explicit_current_state_query("这么晚还能看到云啊")
    assert not is_explicit_current_state_query("你刚才说在看云？")


def test_l3_recall_requires_explicit_oldness():
    """普通 "上次/之前说过" 记忆查询走日常交流; 只有明确久远才进 L3。"""
    from app.services.chat.intent_dispatcher import is_explicit_l3_recall_query

    assert not is_explicit_l3_recall_query("你记得我上次和你说的那家书店吗")
    assert not is_explicit_l3_recall_query("你还记得我之前说的X吗")
    assert not is_explicit_l3_recall_query("你记得我家在哪吗")
    assert not is_explicit_l3_recall_query("去年上映的那个电影你看了吗")

    assert is_explicit_l3_recall_query("你还记得半年前我说的X吗")
    assert is_explicit_l3_recall_query("你能想起更早之前我说的事吗")
    assert is_explicit_l3_recall_query("第一次见面我说过什么")


# ═══════════════════════════════════════════════════════════════════
# Prompt 结构守门 — 防 future 编辑误删 ontology hooks
# ═══════════════════════════════════════════════════════════════════


def test_prompt_keeps_record_request_ontology_hooks():
    """RECORD_REQUEST 把 设置/取消/改期 三种语义统一到一个 intent 是非显然
    的设计决策 — LLM 默认会拆成 3 个不同 intent. prompt 必须显式把三类提到,
    否则 LLM 退化."""
    from app.services.prompting.defaults import INTENT_UNIFIED_PROMPT

    # 三类语义都必须在 prompt 里露面 (允许同义词)
    assert "设置" in INTENT_UNIFIED_PROMPT, (
        "RECORD_REQUEST 必须明示包含'设置'语义, 否则 LLM 不会归类设提醒类消息"
    )
    assert "取消" in INTENT_UNIFIED_PROMPT, (
        "RECORD_REQUEST 必须明示包含'取消'语义 — 用户'算了别提醒了'类消息"
        "默认会被归'日常交流', 必须 prompt 显式 cover"
    )
    assert "改期" in INTENT_UNIFIED_PROMPT, (
        "RECORD_REQUEST 必须明示包含'改期'语义"
    )


def test_prompt_keeps_disambiguation_against_recall():
    """'你还记得我X吗' 容易被错归 RECORD_REQUEST (字面有'记得'). prompt 必须
    带 disambiguation 把这类区分出去."""
    from app.services.prompting.defaults import INTENT_UNIFIED_PROMPT

    # 对应 RECORD_REQUEST 选项行附近必须有"不是"/"区分"类的反向 hint
    record_idx = INTENT_UNIFIED_PROMPT.find("记录请求")
    assert record_idx >= 0
    record_block = INTENT_UNIFIED_PROMPT[record_idx:record_idx + 400]
    has_disambig = any(
        kw in record_block for kw in ("不是这里", "区分", "调用久远记忆")
    )
    assert has_disambig, (
        "RECORD_REQUEST 选项必须带 disambiguation, 防 LLM 把'你还记得我X吗'"
        "类查询误归这里"
    )


def test_unified_prompt_record_request_is_concise():
    """防 prompt 退化为 lookup table: RECORD_REQUEST 选项段 ≤ 6 行 + ≤ 8 个例子.
    历史教训 (af5e22a / 444ac7d): 例子越多 LLM 越当 lookup, 反而对未列出
    变体识别变差. 维护时若想加例子, 先把对应的回归 phrase 加到本文件
    RECORD_REQUEST_MUST_HIT/MISS, 再决定是否真值得放 prompt."""
    from app.services.prompting.defaults import INTENT_UNIFIED_PROMPT

    record_idx = INTENT_UNIFIED_PROMPT.find("- 记录请求")
    assert record_idx >= 0
    # 找到下一个 "- " 顶级选项作为段结束
    next_opt = INTENT_UNIFIED_PROMPT.find("\n- ", record_idx + 1)
    block = INTENT_UNIFIED_PROMPT[record_idx:next_opt if next_opt > 0 else None]

    line_count = len([ln for ln in block.split("\n") if ln.strip()])
    assert line_count <= 6, (
        f"RECORD_REQUEST 选项段 {line_count} 行, 超过 6 行说明又在堆补丁 — "
        f"先把例子搬 tests/test_intent_record_request_phrases.py 再考虑加 prompt"
    )

    # 引号包围的例子数 (中文/英文引号都算)
    example_count = block.count('"') // 2 + block.count("'") // 2
    assert example_count <= 8, (
        f"RECORD_REQUEST 选项段含 {example_count} 个例子, 超过 8 个 — 同上"
    )


def test_historical_phrase_fixtures_non_empty():
    """sanity: 回归 phrase 清单不能因误编辑被清空, 否则失去回归价值."""
    assert len(RECORD_REQUEST_MUST_HIT) >= 10, (
        "回归清单至少应保留 10 个历史误判 phrase, 防止 prompt 收敛过度时漏覆盖"
    )
    assert len(RECORD_REQUEST_MUST_MISS) >= 4, (
        "disambiguation 反例至少 4 个 (覆盖 调用久远记忆/日常交流/计划查询/终结意图)"
    )


def test_intent_prompt_distinguishes_invitation_from_schedule_query():
    """生产 bug 复现 (2026-05-03 trace 019decd3): 用户 "你明天一起跟我去看电影?"
    被 LLM 错归 计划查询 → 路由到 handle_schedule_query → AI 跑去念叨自己一天活动
    + 邀请回答被 max_chars=120 截断.

    根因: prompt 里 计划查询 例子 "你明天有空吗"/"你周末忙吗" 都是查可用性,
    LLM 看到 "你+明天+问号" 表面相似就误归. 邀请 ("跟我一起去X") 跟可用性查询
    语义不同 — 邀请重点是"约你"不是"查你".

    修: 计划查询定义加 disambiguation, 显式说邀请归"日常交流"."""
    from app.services.prompting.defaults import INTENT_UNIFIED_PROMPT

    schedule_idx = INTENT_UNIFIED_PROMPT.find("- 计划查询")
    assert schedule_idx >= 0
    next_opt = INTENT_UNIFIED_PROMPT.find("\n- ", schedule_idx + 1)
    block = INTENT_UNIFIED_PROMPT[schedule_idx:next_opt if next_opt > 0 else None]

    # 必须含"邀请" disambiguation 关键词, 让 LLM 区分 query vs invite
    assert "邀请" in block, (
        "计划查询定义必须显式提到'邀请', 否则 '你明天跟我一起去X' 之类邀请会被"
        "错归 (生产 bug 2026-05-03)"
    )
    # 必须明示邀请归"日常交流"
    assert "日常交流" in block, (
        "计划查询的邀请 disambiguation 必须显式说邀请归'日常交流', 不然 LLM "
        "知道'不归这里'但不知道该归哪里"
    )
