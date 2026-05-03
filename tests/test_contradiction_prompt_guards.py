"""Regression: contradiction detection + reply prompt 必须含 anti-hallucination
+ anti-common-sense-conflict 守门.

生产 bug 复现 (2026-05-03 trace 019decdc):
  spec §4 矛盾流程被 LLM **常识推断**误触发, contradiction_reply 又凭空编
  了用户从没说过的具体地点细节. 双重 hallucinate.

修: 两个 prompt 都加 general principle 而非具体反例 — 抓本质 (字面 vs 推断,
同维度 vs 跨维度, 字面引用 vs 隐式记忆词), 不是 lift 具体生产 case.

本测试只 assert general principle 关键词在 prompt 里, 不 assert 任何具体反例
(避免跟之前 RECORD_REQUEST prompt overfit 同样的过拟合错误重蹈). 实际 LLM
行为效果需手动 / trace 复盘.
"""

from __future__ import annotations


def test_contradiction_detection_forbids_inference_as_conflict_source():
    """detection prompt 必须显式禁止把 LLM **推断/常识**当 conflict 的来源.

    生产 bug 根因: LLM 拿"生日通常不去钓鱼"这种社会常识当 L1 conflict 触发
    spec §4 矛盾流程. 修复: prompt 显式说"两端都必须是字面事实", 排除常识/
    推断/合理化作为 conflict 来源."""
    from app.services.prompting.defaults import MEMORY_CONTRADICTION_DETECTION_PROMPT
    p = MEMORY_CONTRADICTION_DETECTION_PROMPT

    # 必须强调"字面事实" — 抓 inference vs literal 的本质区分
    assert "字面" in p, "必须强调'字面事实', 排除 LLM 推断"
    # 必须禁止"常识 / 通常 / 一般" 类推论
    assert any(kw in p for kw in ("常识", "通常", "一般")), (
        "必须显式禁止用'常识/通常/一般' 推论造矛盾"
    )


def test_contradiction_detection_requires_same_attribute():
    """conflict 必须是**同一属性**不同取值, 不允许跨维度造矛盾.
    (e.g. 'L1 住北京' + '用户是程序员' — 不同属性, 不矛盾)"""
    from app.services.prompting.defaults import MEMORY_CONTRADICTION_DETECTION_PROMPT
    p = MEMORY_CONTRADICTION_DETECTION_PROMPT

    # "同一属性" / "同维度" / "不同维度" 任一表达都行
    assert any(kw in p for kw in ("同一属性", "同维度", "不同维度", "不同属性")), (
        "必须约束 conflict 必须是同一属性的不同取值, 防跨维度造假矛盾"
    )


def test_contradiction_detection_question_mark_not_conflict():
    """用户消息含问号 (邀请/求助/询问) 不是事实断言, 通常不矛盾."""
    from app.services.prompting.defaults import MEMORY_CONTRADICTION_DETECTION_PROMPT
    p = MEMORY_CONTRADICTION_DETECTION_PROMPT

    assert "问号" in p or "问句" in p, (
        "必须显式说问句/邀请通常不算矛盾, 防 LLM 把所有'你 X 吗?' 当事实断言对比"
    )


def test_contradiction_reply_forbids_implicit_memory_phrases():
    """contradiction_reply 必须显式禁'上次 / 之前 / 那个 X / 你说的 X' 这类
    隐式记忆引用 — LLM 顺手用这些措辞就会编出未提及的具体细节."""
    from app.services.prompting.defaults import MEMORY_CONTRADICTION_REPLY_PROMPT
    p = MEMORY_CONTRADICTION_REPLY_PROMPT

    # 必须显式列出禁用措辞
    assert "上次" in p, "必须禁'上次 X' 这种隐式记忆引用措辞"
    assert "之前" in p, "必须禁'之前 X' 同上"


def test_contradiction_reply_requires_literal_reference_only():
    """所有具体名词 (地点/人/物/时间) 必须 字面出现在【参考信息】里, 否则不能说."""
    from app.services.prompting.defaults import MEMORY_CONTRADICTION_REPLY_PROMPT
    p = MEMORY_CONTRADICTION_REPLY_PROMPT

    assert "字面" in p, (
        "必须强调'字面出现', 不允许带入未在参考信息里的具体名词"
    )


def test_contradiction_reply_open_vs_closed_question_guidance():
    """不确定细节时用开放问题, 不要用假定细节的封闭问题. 这是 anti-fabrication
    的核心 patten — 把 LLM 引向 '问' 而不是 '编'."""
    from app.services.prompting.defaults import MEMORY_CONTRADICTION_REPLY_PROMPT
    p = MEMORY_CONTRADICTION_REPLY_PROMPT

    assert "开放" in p or "开放问题" in p, (
        "必须给 LLM '不确定就开放问' 的指导, 否则 LLM 倾向'封闭问' (假定细节)"
    )


def test_contradiction_reply_handles_empty_context():
    """对话上下文是'(无)' 或很短时, 必须强调只基于当前消息展开."""
    from app.services.prompting.defaults import MEMORY_CONTRADICTION_REPLY_PROMPT
    p = MEMORY_CONTRADICTION_REPLY_PROMPT

    assert any(kw in p for kw in ("(无)", "（无）")), (
        "必须显式 cover '上下文是空' 的边缘 case, 防 LLM 凭空补'以前 X'"
    )


def test_no_specific_production_case_phrases_in_prompts():
    """守门: 这两个 prompt 不该含生产 trace 的具体 phrase ('钓鱼'/'水库'/'生日'
    等). 这些是抽 general principle 的反例时容易 lift 进 prompt 的痕迹, 但
    会 overfit (跟之前 RECORD_REQUEST prompt 例子堆叠 bug 同根).

    维护者: 加新规则时, 用 X/Y 抽象变量 (e.g. 'X 属性=A vs X 属性=B') 而非
    具体名词 ('住址=北京 vs 住址=上海'). 防 prompt 退化为 lookup table."""
    from app.services.prompting.defaults import (
        MEMORY_CONTRADICTION_DETECTION_PROMPT,
        MEMORY_CONTRADICTION_REPLY_PROMPT,
    )
    case_phrases = ("钓鱼", "水库", "爬山")  # 不允许出现的生产 case phrase
    for prompt_name, prompt in [
        ("detection", MEMORY_CONTRADICTION_DETECTION_PROMPT),
        ("reply", MEMORY_CONTRADICTION_REPLY_PROMPT),
    ]:
        for phrase in case_phrases:
            assert phrase not in prompt, (
                f"{prompt_name} prompt 含生产 case phrase '{phrase}' — 这是过拟合, "
                f"应该用抽象表述 (X/Y 属性) 替代具体名词"
            )
