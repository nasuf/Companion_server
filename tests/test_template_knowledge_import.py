"""Knowledge-supplement document parser (agent_template/knowledge_import.py).

The parser is deterministic (no LLM): section headings + 标签：内容 lines →
self-contained memory summaries prefixed with the section subject.
"""

from __future__ import annotations

import pytest

from app.services.agent_template.knowledge_import import (
    MAX_KNOWLEDGE_ITEMS,
    parse_knowledge_document,
)

XIJIA_DOC = """公司介绍
公司名称：伴生
公司定位：陪伴科技公司
公司核心理念：打造“有生命的AI”，追求有独立人格、真实情绪、日常作息与个人边界的陪伴体验。
产品介绍
产品名称：伴生App
产品上线时间：预计2026年9月
合作项目介绍
项目名称：2026年恒洁杯第二十届佛山“西甲”足球联赛
赛事时间：2026年7月10日至8月23日
赛事地点：佛山三水云秀山体育场
"""


def _summaries(text: str) -> list[str]:
    return [item.summary for item in parse_knowledge_document(text.encode("utf-8"))]


def test_parse_sections_persona_voice_summaries():
    items = parse_knowledge_document(XIJIA_DOC.encode("utf-8"))
    summaries = [i.summary for i in items]

    # Recognized sections (公司/产品/合作) render as FIRST-PERSON work
    # memories — the relationship the section heading carries must survive
    # into the stored text, or the AI can never say "我们公司合作的比赛".
    assert "我所在的公司名称：伴生" in summaries
    assert "我所在的公司「伴生」定位：陪伴科技公司" in summaries
    assert "我们公司的产品名称：伴生App" in summaries
    assert "我们公司的产品「伴生App」上线时间：预计2026年9月" in summaries
    assert (
        "我们公司合作的项目「2026年恒洁杯第二十届佛山“西甲”足球联赛」"
        "赛事时间：2026年7月10日至8月23日"
    ) in summaries

    # Section / label metadata preserved for the admin preview.
    time_item = next(i for i in items if i.label == "赛事时间")
    assert time_item.section == "合作项目介绍"
    assert time_item.content == "2026年7月10日至8月23日"


def test_relational_stem_without_name_line_has_no_subject_brackets():
    # 活动 keyword → stem, but no 名称 line → no 「subject」 duplication.
    summaries = _summaries("周边活动介绍\n优惠范围：三水超1000家商户\n")
    assert summaries == ["我们公司的活动优惠范围：三水超1000家商户"]


def test_neutral_section_falls_back_to_subject_prefix():
    # No 公司/产品/合作/活动 keyword → pre-existing neutral subject form.
    summaries = _summaries("家乡风物介绍\n特产：小锅米线\n")
    assert summaries == ["家乡风物的特产：小锅米线"]


def test_reject_five_dimension_profile():
    profile_doc = "\n".join(
        [
            "林某的五维人格记忆档案",
            "1. AI自我姓名",
            "大名：林某",
            "2. AI自我年龄",
            "22岁",
            "3. AI自我性别",
            "女",
        ]
    )
    with pytest.raises(ValueError, match="五维人格档案"):
        parse_knowledge_document(profile_doc.encode("utf-8"))


def test_empty_file_raises():
    with pytest.raises(ValueError):
        parse_knowledge_document(b"")


def test_no_items_raises():
    # Short colon-less lines are headings; a doc of only headings has 0 items.
    with pytest.raises(ValueError, match="未识别到任何知识条目"):
        parse_knowledge_document("公司介绍\n产品介绍\n".encode("utf-8"))


def test_duplicate_lines_are_deduped():
    doc = "公司介绍\n公司名称：伴生\n公司名称：伴生\n"
    assert _summaries(doc) == ["我所在的公司名称：伴生"]


def test_heading_variants_hash_enum_and_trailing_colon():
    doc = "#其他\n1. 公司介绍：\n公司名称：伴生\n"
    items = parse_knowledge_document(doc.encode("utf-8"))
    assert [i.summary for i in items] == ["我所在的公司名称：伴生"]
    assert items[0].section == "公司介绍"


def test_long_colonless_line_joins_current_section_as_content():
    long_line = "这个项目覆盖粤港澳大湾区十一个城市并且包含一百零四场比赛非常盛大"
    doc = f"合作项目介绍\n项目名称：西甲联赛\n{long_line}\n"
    summaries = _summaries(doc)
    assert f"我们公司合作的项目「西甲联赛」：{long_line}" in summaries


def test_colon_inside_sentence_is_not_a_label():
    # "Label" longer than the heading cap (24) → whole line kept as content.
    doc = (
        "公司介绍\n公司名称：伴生\n"
        "他们的口号是这样说的所以大家每一个人都一定要记住这句话：只为陪伴而生\n"
    )
    summaries = _summaries(doc)
    assert any(s.startswith("我所在的公司「伴生」：他们的口号") for s in summaries)


def test_gb18030_documents_decode():
    assert "我所在的公司「伴生」定位：陪伴科技公司" in [
        i.summary
        for i in parse_knowledge_document(XIJIA_DOC.encode("gb18030"))
    ]


def test_item_cap_enforced():
    lines = ["压力测试"] + [f"字段{i}：值{i}" for i in range(MAX_KNOWLEDGE_ITEMS + 1)]
    with pytest.raises(ValueError, match="知识条目过多"):
        parse_knowledge_document("\n".join(lines).encode("utf-8"))
