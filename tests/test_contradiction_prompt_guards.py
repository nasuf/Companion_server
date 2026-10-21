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

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


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


def test_contradiction_grounding_rejects_single_weak_overlap():
    """后验 grounding 不能因一个泛化短词重叠就放行。"""
    from app.services.memory.interaction.contradiction import _conflict_new_info_grounded

    assert not _conflict_new_info_grounded(
        {"new_info": "用户喜欢上海"},
        "我讨厌上海，但现在住杭州",
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


def _l1(mid, content, sub, main="身份"):
    return SimpleNamespace(
        id=mid, content=content, mainCategory=main, subCategory=sub,
        createdAt=datetime(2026, 10, 10, 13, 0, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_contradiction_detector_discards_l1_to_l1_conflict_not_in_current_message():
    """LLM 把两条旧 L1 互相冲突误报为当前矛盾时, 后验校验必须丢弃。"""
    from app.services.memory.interaction.contradiction import detect_l1_contradiction

    memories = [
        _l1("m-old", "用户是独生女，无兄弟姐妹", "亲属关系"),
        _l1("m-new", "用户有个妹妹叫辽能", "亲属关系"),
    ]
    llm_result = {
        "has_conflict": True,
        "conflicting_memory_id": "m-old",
        "old_content": "用户是独生女，无兄弟姐妹",
        "new_info": "用户有个妹妹叫辽能",
        "conflict_description": "有妹妹与独生女矛盾",
    }

    with (
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.find_many",
            new=AsyncMock(return_value=memories),
        ),
        patch(
            "app.services.memory.interaction.contradiction.get_prompt_text",
            new=AsyncMock(return_value="{user_message}\n{existing_l1_memory}"),
        ),
        patch(
            "app.services.memory.interaction.contradiction.invoke_json",
            new=AsyncMock(return_value=llm_result),
        ),
    ):
        result = await detect_l1_contradiction(
            "我身高184，体重54公斤，最喜欢的明星是艾克",
            "u1",
            workspace_id="w1",
        )

    assert result is None


@pytest.mark.asyncio
async def test_contradiction_detector_keeps_conflict_grounded_in_current_message():
    """当前用户消息字面支撑 new_info 时, 不应误杀真实矛盾。"""
    from app.services.memory.interaction.contradiction import detect_l1_contradiction

    memories = [
        _l1("m-city", "用户现在住北京", "现居地"),
    ]
    llm_result = {
        "has_conflict": True,
        "conflicting_memory_id": "m-city",
        "old_content": "用户现在住北京",
        "new_info": "用户现在住上海",
        "conflict_description": "现居地从北京变为上海",
    }

    with (
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.find_many",
            new=AsyncMock(return_value=memories),
        ),
        patch(
            "app.services.memory.interaction.contradiction.get_prompt_text",
            new=AsyncMock(return_value="{user_message}\n{existing_l1_memory}"),
        ),
        patch(
            "app.services.memory.interaction.contradiction.invoke_json",
            new=AsyncMock(return_value=llm_result),
        ),
    ):
        result = await detect_l1_contradiction(
            "我现在住上海了",
            "u1",
            workspace_id="w1",
        )

    assert result == llm_result


def test_contradiction_detection_reads_context_but_not_as_conflict_source():
    """省略句要放回上下文才知道在说什么; 但上下文只能帮助理解, 不能当矛盾的一端."""
    from app.services.prompting.defaults import MEMORY_CONTRADICTION_DETECTION_PROMPT
    p = MEMORY_CONTRADICTION_DETECTION_PROMPT

    assert "{recent_context}" in p
    assert "矛盾的任何一端都不能取自这里" in p
    assert "当下状态" in p, "当下状态/临时行程跟长期身份属性不是同一属性"


def test_contradiction_inquiry_answers_the_message_and_avoids_rhetorical_questions():
    """追问是本轮唯一回复: 必须先接住对方这句话, 也不能用反问句质疑对方."""
    from app.services.prompting.defaults import MEMORY_CONTRADICTION_INQUIRY_PROMPT
    p = MEMORY_CONTRADICTION_INQUIRY_PROMPT

    assert "{recent_context}" in p
    assert "先结合最近对话接住对方这句话本身" in p
    assert "反问" in p


def _detect_patches(memories, llm_result, captured=None):
    async def invoke(_model, prompt):
        if captured is not None:
            captured.append(prompt)
        return llm_result

    return (
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.find_many",
            new=AsyncMock(return_value=memories),
        ),
        patch(
            "app.services.memory.interaction.contradiction.get_prompt_text",
            new=AsyncMock(return_value="{user_message}\n{recent_context}\n{existing_l1_memory}"),
        ),
        patch(
            "app.services.memory.interaction.contradiction.invoke_json",
            new=AsyncMock(side_effect=invoke),
        ),
    )


@pytest.mark.asyncio
async def test_contradiction_detector_only_compares_identity_attributes():
    """当下状态 / 情景 / "其他" 类混进 L1 也不参与对照 —— 一句"我不在啊"
    不能因为 L1 里有"用户已经在工位了"就被判成矛盾."""
    from app.services.memory.interaction.contradiction import detect_l1_contradiction

    memories = [
        _l1("m-desk", "用户已经在工位了", "其他", main="生活"),
        _l1("m-ask", "用户询问是否和对方在一个地方", "其他", main="思维"),
        _l1("m-misc", "用户身份其他", "其他"),
    ]
    p1, p2, p3 = _detect_patches(memories, {"has_conflict": True, "conflicting_memory_id": "m-desk"})
    with p1 as find_many, p2, p3 as invoke:
        result = await detect_l1_contradiction("我不在啊", "u1", workspace_id="w1")

    assert result is None
    invoke.assert_not_awaited()
    assert find_many.await_args.kwargs["where"]["mainCategory"] == "身份"


@pytest.mark.asyncio
async def test_contradiction_detector_sends_context_and_record_dates():
    from app.services.memory.interaction.contradiction import detect_l1_contradiction

    captured: list[str] = []
    p1, p2, p3 = _detect_patches(
        [_l1("m-city", "用户现在住北京", "现居地")], {"has_conflict": False}, captured,
    )
    with p1, p2, p3:
        result = await detect_l1_contradiction(
            "我不在啊", "u1", workspace_id="w1", recent_context="AI: 你也在北京吗？",
        )

    assert result is None
    assert "AI: 你也在北京吗？" in captured[0]
    assert "用户现在住北京（现居地，记录于 2026-10-10）" in captured[0]


@pytest.mark.asyncio
async def test_contradiction_detector_without_context_says_so():
    from app.services.memory.interaction.contradiction import detect_l1_contradiction

    captured: list[str] = []
    p1, p2, p3 = _detect_patches(
        [_l1("m-city", "用户现在住北京", "现居地")], {"has_conflict": False}, captured,
    )
    with p1, p2, p3:
        await detect_l1_contradiction("我搬到上海了", "u1", workspace_id="w1")

    assert "(无)" in captured[0]


def test_record_date_uses_local_day_and_tolerates_missing_or_naive_stamps():
    from app.services.memory.interaction.contradiction import _record_date

    assert _record_date(SimpleNamespace(createdAt=None)) == "时间未知"
    # 库里是 UTC 的 naive 时间: 23:30 UTC 已经是本地次日
    assert _record_date(SimpleNamespace(createdAt=datetime(2026, 10, 10, 23, 30))) == "2026-10-11"
    assert _record_date(SimpleNamespace(createdAt=datetime(2026, 10, 10, 1, 0, tzinfo=UTC))) == "2026-10-10"
