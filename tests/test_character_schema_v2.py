"""Schema v2 完整性测试: 13 大分类 + taxonomy 子类对齐 + 字段类型规范."""

from __future__ import annotations

import pytest

from app.services.character import (
    DEFAULT_TEMPLATE_SCHEMA,
    DEFAULT_TEMPLATE_DEFAULTS,
    build_generation_prompt,
    _apply_postprocess_overrides,
    _is_current_schema,
)
from app.services.prompting.defaults import CHARACTER_TEMPLATE_HEADER_PROMPT
from app.services.memory.taxonomy import TAXONOMY_MATRIX


def _flatten_schema_subs(schema: dict) -> set[str]:
    subs: set[str] = set()
    for cat in schema.get("categories", []):
        cat_sub = cat.get("_memory_sub")
        if cat_sub:
            subs.add(cat_sub)
        for field in cat.get("fields", []):
            field_sub = field.get("_memory_sub")
            if field_sub:
                subs.add(field_sub)
    return subs


def test_schema_has_13_categories():
    assert len(DEFAULT_TEMPLATE_SCHEMA["categories"]) == 13


def test_schema_includes_v2_event_categories():
    keys = {c["key"] for c in DEFAULT_TEMPLATE_SCHEMA["categories"]}
    for required in ("life_events", "emotion_events", "interpersonal", "lifestyle", "taboo"):
        assert required in keys, f"missing v2 category {required}"


def test_schema_has_no_fears_category():
    """fears 已删除, 信息移至 emotion_events.fear hint."""
    keys = {c["key"] for c in DEFAULT_TEMPLATE_SCHEMA["categories"]}
    assert "fears" not in keys


def test_schema_identity_no_ethnicity_field():
    """民族硬编码汉族, schema 不暴露给 LLM."""
    identity = next(c for c in DEFAULT_TEMPLATE_SCHEMA["categories"] if c["key"] == "identity")
    field_keys = {f["key"] for f in identity["fields"]}
    assert "ethnicity" not in field_keys


def test_schema_identity_has_zodiac_constellation_name_pet_social():
    """v2 新增字段."""
    identity = next(c for c in DEFAULT_TEMPLATE_SCHEMA["categories"] if c["key"] == "identity")
    field_keys = {f["key"] for f in identity["fields"]}
    for required in ("name", "zodiac", "constellation", "social_relations", "pet_profile"):
        assert required in field_keys, f"missing identity field {required}"


def test_schema_values_has_all_thought_subs():
    """思维 8 项全覆盖 (PDF #34 第五维)."""
    values = next(c for c in DEFAULT_TEMPLATE_SCHEMA["categories"] if c["key"] == "values")
    field_keys = {f["key"] for f in values["fields"]}
    for required in ("motto", "believes", "opposes", "worldview", "goal",
                     "interpersonal_view", "social_view", "faith"):
        assert required in field_keys, f"missing values field {required}"


def test_schema_life_events_11_fields():
    le = next(c for c in DEFAULT_TEMPLATE_SCHEMA["categories"] if c["key"] == "life_events")
    assert len(le["fields"]) == 11


def test_schema_emotion_events_15_fields():
    ee = next(c for c in DEFAULT_TEMPLATE_SCHEMA["categories"] if c["key"] == "emotion_events")
    assert len(ee["fields"]) == 15


def test_schema_subs_align_with_ai_taxonomy():
    """schema 所有 _memory_sub 必须命中 taxonomy.MATRIX[ai][1] 的 L1 子类
    (character profile 写入 memories_ai, 用 ai 侧 taxonomy)."""
    schema_subs = _flatten_schema_subs(DEFAULT_TEMPLATE_SCHEMA)
    ai_l1 = TAXONOMY_MATRIX["ai"][1]
    all_l1_subs = {sub for subs in ai_l1.values() for sub in subs}
    unmapped = schema_subs - all_l1_subs
    assert not unmapped, f"schema has subs not in ai taxonomy: {unmapped}"


def test_taxonomy_ai_l1_named_subs_covered_by_schema():
    """除 5 项豁免 + 民族(后处理硬写)外, ai taxonomy L1 命名子类应都有 schema 字段对应."""
    from app.services.memory.taxonomy import L1_COVERAGE_EXEMPT, L1_CONDITIONAL_SUBS

    schema_subs = _flatten_schema_subs(DEFAULT_TEMPLATE_SCHEMA)
    ai_l1 = TAXONOMY_MATRIX["ai"][1]
    skip = {sub for _main, sub in L1_COVERAGE_EXEMPT}  # "其他" 等豁免
    skip.add("提醒")  # 运行时累积, 不预填
    skip.add("民族")  # 后处理硬写「汉族」, schema 不暴露给 LLM
    # 条件可选子类 (依赖 profile 信号) 启发式补, 不强制 schema 字段
    skip_conditional_only = {sub for _main, sub in L1_CONDITIONAL_SUBS}
    expected = set()
    for _main, subs in ai_l1.items():
        for sub in subs:
            if sub in skip:
                continue
            expected.add(sub)
    missing = expected - schema_subs - skip_conditional_only
    assert not missing, f"taxonomy subs missing schema field: {missing}"


def test_is_current_schema_detects_correctly():
    assert _is_current_schema(DEFAULT_TEMPLATE_SCHEMA) is True
    old_schema = {"categories": [{"key": "identity", "fields": []}]}
    assert _is_current_schema(old_schema) is False
    assert _is_current_schema(None) is False


def test_is_current_schema_rejects_legacy_output_field():
    """老 schema 残留 career.output 应触发迁移."""
    legacy = {
        "categories": [
            {"key": "life_events", "fields": []},
            {"key": "emotion_events", "fields": []},
            {"key": "career", "fields": [{"key": "output", "name": "主要产出物"}]},
        ],
    }
    assert _is_current_schema(legacy) is False


def test_is_current_schema_rejects_legacy_dislikes_habits():
    """老 schema 残留 dislikes.habits 应触发迁移."""
    legacy = {
        "categories": [
            {"key": "life_events", "fields": []},
            {"key": "emotion_events", "fields": []},
            {"key": "dislikes", "fields": [{"key": "habits", "name": "习惯"}]},
        ],
    }
    assert _is_current_schema(legacy) is False


def test_schema_career_no_output_field():
    """spec 4.21 删除主要产出物."""
    career = next(c for c in DEFAULT_TEMPLATE_SCHEMA["categories"] if c["key"] == "career")
    field_keys = {f["key"] for f in career["fields"]}
    assert "output" not in field_keys


def test_schema_dislikes_no_habits_field():
    """习惯不在 spec dislikes 维度内."""
    dislikes = next(c for c in DEFAULT_TEMPLATE_SCHEMA["categories"] if c["key"] == "dislikes")
    field_keys = {f["key"] for f in dislikes["fields"]}
    assert "habits" not in field_keys


def test_default_prompt_header_contains_required_clauses():
    assert "AI人格构建专家" in CHARACTER_TEMPLATE_HEADER_PROMPT
    assert "中国区域内的地球镜像世界" in CHARACTER_TEMPLATE_HEADER_PROMPT
    assert "社会主义核心价值观" in CHARACTER_TEMPLATE_HEADER_PROMPT


def test_default_defaults_contains_all_pdf_required_rules():
    required_phrases = [
        "中国区域内的地球镜像世界",
        "社会主义核心价值观与公序良俗",
        "20-29 岁",
        "男性身高 165-190cm，女性身高 155-175cm",
        "生肖必须与出生年份对应",
        "星座必须与生日月日对应",
        "性格 7 维分数深度绑定",
        "life_events 每类 3-5 个具体片段",
        "emotion_events 每类 3-5 个具体情境",
        "fear 字段必须明确害怕的动物/物品/氛围",
        "年收入 < 10 万",
    ]
    for phrase in required_phrases:
        assert phrase in DEFAULT_TEMPLATE_DEFAULTS, f"defaults missing: {phrase}"


def test_build_generation_prompt_injects_name_when_provided():
    prompt = build_generation_prompt(
        DEFAULT_TEMPLATE_SCHEMA, DEFAULT_TEMPLATE_DEFAULTS,
        index=0, name="李小雨",
    )
    assert "【姓名】李小雨" in prompt
    assert "identity.name" in prompt


def test_build_generation_prompt_injects_personality_7dim():
    personality = {
        "liveliness": 85, "rationality": 30, "sensitivity": 75,
        "planning": 40, "spontaneity": 80, "imagination": 70, "humor": 65,
    }
    prompt = build_generation_prompt(
        DEFAULT_TEMPLATE_SCHEMA, DEFAULT_TEMPLATE_DEFAULTS,
        index=0, personality=personality,
    )
    for label in ("活泼度", "理性度", "感性度", "计划度", "随性度", "脑洞度", "幽默度"):
        assert label in prompt
    assert "85" in prompt and "30" in prompt and "65" in prompt


def test_build_generation_prompt_excludes_career_category_when_career_provided():
    career = {"title": "日落收集师", "duties": "黄昏观测", "clients": "机构"}
    prompt = build_generation_prompt(
        DEFAULT_TEMPLATE_SCHEMA, DEFAULT_TEMPLATE_DEFAULTS,
        index=0, career=career,
    )
    assert "日落收集师" in prompt
    # career 分类被剔除
    assert "职业与生存闭环（key: career）" not in prompt


def test_build_generation_prompt_renders_career_income_default():
    """career 缺 income 时 _build_career_section 应填合规默认 < 10 万."""
    career = {"title": "测试职业"}
    prompt = build_generation_prompt(
        DEFAULT_TEMPLATE_SCHEMA, DEFAULT_TEMPLATE_DEFAULTS,
        index=0, career=career,
    )
    assert "经济状况" in prompt
    assert "5-10 万" in prompt or "5-10万" in prompt


def test_build_generation_prompt_gender_constraint():
    prompt_f = build_generation_prompt(
        DEFAULT_TEMPLATE_SCHEMA, DEFAULT_TEMPLATE_DEFAULTS,
        index=0, gender="female",
    )
    assert "女性" in prompt_f
    assert "「女」" in prompt_f
    prompt_m = build_generation_prompt(
        DEFAULT_TEMPLATE_SCHEMA, DEFAULT_TEMPLATE_DEFAULTS,
        index=0, gender="male",
    )
    assert "男性" in prompt_m
    assert "「男」" in prompt_m


def test_postprocess_hardcodes_ethnicity_han():
    profile = {"identity": {"gender": "女", "name": "test"}}
    result = _apply_postprocess_overrides(profile, agent_name="test", career=None)
    assert result["identity"]["ethnicity"] == "汉族"


def test_postprocess_overrides_name_with_agent_name():
    """LLM 即使乱填 name, 后处理也会硬写为 agent_name (PDF #34 直接引用)."""
    profile = {"identity": {"name": "WrongName"}}
    result = _apply_postprocess_overrides(profile, agent_name="李小雨", career=None)
    assert result["identity"]["name"] == "李小雨"


def test_postprocess_strips_name_when_no_agent_name():
    """背景池场景 (admin 批量生成无姓名): LLM 自填的随机名应被清掉,
    名字由用户在 agent 创建页输入."""
    profile = {"identity": {"name": "林晓晴", "age": 25}}
    result = _apply_postprocess_overrides(profile, agent_name=None, career=None)
    assert "name" not in result["identity"]
    assert result["identity"]["age"] == 25  # 其他字段保留


def test_postprocess_fills_career_from_template():
    profile = {"identity": {}}
    career = {
        "title": "日落收集师", "duties": "观测", "socialValue": "调节情绪",
        "clients": "机构", "income": "年薪 6 万",
    }
    result = _apply_postprocess_overrides(profile, agent_name="x", career=career)
    assert result["career"]["title"] == "日落收集师"
    assert result["career"]["income"] == "年薪 6 万"
    assert result["career"]["social_value"] == "调节情绪"


def test_postprocess_career_income_default_when_missing():
    career = {"title": "x"}
    profile = {"identity": {}}
    result = _apply_postprocess_overrides(profile, agent_name="x", career=career)
    assert result["career"]["income"] == "年薪 5-10 万"


def test_postprocess_career_clients_split_from_string():
    """clients 既支持 list 也支持原始 DB string (含、，；分隔符)."""
    career = {"title": "x", "clients": "饲养伤残动物的主人；宠物医院、动物康复机构"}
    profile = {"identity": {}}
    result = _apply_postprocess_overrides(profile, agent_name="x", career=career)
    assert result["career"]["clients"] == [
        "饲养伤残动物的主人", "宠物医院", "动物康复机构",
    ]


def test_postprocess_career_clients_already_list_passthrough():
    career = {"title": "x", "clients": ["A", "B", " C "]}
    profile = {"identity": {}}
    result = _apply_postprocess_overrides(profile, agent_name="x", career=career)
    assert result["career"]["clients"] == ["A", "B", "C"]


# ── repair (缺字段补齐) 测试 ──

from app.services.character import (  # noqa: E402
    _detect_missing_fields,
    _is_field_empty,
    _build_repair_persona_summary,
    _build_repair_missing_fields_text,
)


def test_is_field_empty_tags():
    assert _is_field_empty([], "tags") is True
    assert _is_field_empty(None, "tags") is True
    assert _is_field_empty(["x"], "tags") is False


def test_is_field_empty_text():
    assert _is_field_empty("", "textarea") is True
    assert _is_field_empty("   ", "textarea") is True
    assert _is_field_empty(None, "text") is True
    assert _is_field_empty("hello", "text") is False


def test_is_field_empty_textarea_dict_treated_as_empty():
    """LLM 偶尔给 textarea 返对象 (如 goal: {short: ..., long: ...}),
    会渲染成 [object Object]. 视为 empty 触发 repair 重生成成合规字符串."""
    assert _is_field_empty({"short": "x", "long": "y"}, "textarea") is True
    assert _is_field_empty({"a": 1}, "text") is True


def test_is_field_empty_number():
    assert _is_field_empty(None, "number") is True
    assert _is_field_empty(0, "number") is False  # 0 is valid
    assert _is_field_empty(25, "number") is False


def test_detect_missing_fields_skips_career_and_identity_overrides():
    """career 整分类 + identity.{name, ethnicity, gender} 不参与 repair (后处理硬覆盖)."""
    schema = {
        "categories": [
            {"key": "identity", "fields": [
                {"key": "name", "type": "text"},
                {"key": "ethnicity", "type": "text"},
                {"key": "gender", "type": "select"},
                {"key": "age", "type": "number"},  # 应该被检测
            ]},
            {"key": "career", "fields": [
                {"key": "title", "type": "text"},  # 跳过整分类
            ]},
        ],
    }
    data = {"identity": {}, "career": {}}  # 全空
    missing = _detect_missing_fields(schema, data)
    field_keys = [f.get("key") for _, f in missing]
    assert field_keys == ["age"]


def test_detect_missing_fields_truncation_pattern():
    """模拟 LLM 截断: life_events 末尾 4 个字段空."""
    schema = {
        "categories": [
            {"key": "life_events", "fields": [
                {"key": "interaction", "type": "tags"},
                {"key": "education", "type": "tags"},
                {"key": "relationships", "type": "tags"},  # 缺
                {"key": "skill_learning", "type": "tags"},  # 缺
                {"key": "life", "type": "tags"},  # 缺
                {"key": "special", "type": "tags"},  # 缺
            ]},
        ],
    }
    data = {"life_events": {
        "interaction": ["事件1"], "education": ["事件2"],
        "relationships": [], "skill_learning": [], "life": [], "special": [],
    }}
    missing = _detect_missing_fields(schema, data)
    field_keys = [f.get("key") for _, f in missing]
    assert field_keys == ["relationships", "skill_learning", "life", "special"]


def test_repair_persona_summary_includes_identity_and_career():
    """注入 character.repair_missing_fields 模板的 {persona_summary} 部分。"""
    data = {
        "identity": {"name": "李小雨", "gender": "女", "age": 25, "location": "上海"},
        "career": {"title": "占卜师"},
    }
    summary = _build_repair_persona_summary(data)
    assert "李小雨" in summary
    assert "占卜师" in summary
    assert "上海" in summary


def test_repair_missing_fields_text_lists_each_field():
    """注入 character.repair_missing_fields 模板的 {missing_fields} 部分。"""
    schema = {
        "categories": [
            {"key": "life_events", "name": "生活记忆事件", "fields": []},
        ],
    }
    missing = [
        ("life_events", {"key": "relieved", "name": "释怀", "type": "tags",
                         "hint": "30-80 字"}),
    ]
    text = _build_repair_missing_fields_text(schema, missing)
    assert "释怀" in text
    assert "relieved" in text
    assert "tags" in text
    assert "30-80 字" in text
