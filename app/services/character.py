"""AI 角色背景管理服务。

模板 CRUD、画像批量生成（LLM）、默认模板 seed。
"""

from __future__ import annotations

import logging

from app.db import db
from app.services.llm.models import get_utility_model, invoke_json
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)


# ── 默认基础模板 (PRD 9大分类) ──

DEFAULT_TEMPLATE_SCHEMA = {
    "categories": [
        {
            "key": "identity",
            "name": "基础身份",
            "sort": 1,
            "fields": [
                {"key": "gender", "name": "性别", "type": "select", "options": ["男", "女"], "required": True},
                {"key": "age", "name": "年龄", "type": "number", "min": 20, "max": 29, "hint": "20-29岁之间"},
                {"key": "birthday", "name": "出生日期", "type": "date"},
                {"key": "education", "name": "教育背景", "type": "textarea"},
                {"key": "location", "name": "所在地", "type": "text", "hint": "出生地=居住地=工作地"},
                {"key": "family", "name": "家庭关系", "type": "textarea", "hint": "父母健在、性格温和、不喜欢热闹、独生"},
            ],
        },
        {
            "key": "appearance",
            "name": "外貌与形象",
            "sort": 2,
            "fields": [
                {"key": "height", "name": "身高", "type": "text", "hint": "女160-170cm，男177-185cm"},
                {"key": "weight", "name": "体型", "type": "text", "hint": "女=身高÷3.6，男=身高÷2.6"},
                {"key": "features", "name": "外貌特征", "type": "textarea"},
                {"key": "style", "name": "穿搭风格", "type": "textarea"},
                {"key": "voice", "name": "声音特点", "type": "textarea"},
            ],
        },
        {
            "key": "education_knowledge",
            "name": "教育与知识",
            "sort": 3,
            "fields": [
                {"key": "degree", "name": "学历", "type": "textarea", "hint": "没上全日制大学，偏爱「XX 职业」相关事情，跟前辈学习"},
                {"key": "strengths", "name": "知识擅长范围", "type": "tags"},
                {"key": "self_taught", "name": "自学过的特殊技能", "type": "tags"},
            ],
        },
        {
            "key": "career",
            "name": "职业与生存闭环",
            "sort": 4,
            "fields": [
                {"key": "title", "name": "职业", "type": "text", "hint": "关联《职业背景设定表》"},
                {"key": "duties", "name": "工作内容", "type": "textarea"},
                {"key": "outputs", "name": "主要产出物", "type": "tags"},
                {"key": "social_value", "name": "社会价值", "type": "textarea"},
                {"key": "clients", "name": "服务对象", "type": "tags"},
            ],
        },
        {
            "key": "likes",
            "name": "喜好",
            "sort": 5,
            "fields": [
                {"key": "colors", "name": "颜色", "type": "tags"},
                {"key": "foods", "name": "食物", "type": "tags"},
                {"key": "fruits", "name": "水果", "type": "tags"},
                {"key": "season", "name": "季节", "type": "tags"},
                {"key": "weather", "name": "天气", "type": "tags"},
                {"key": "plants", "name": "植物", "type": "tags"},
                {"key": "animals", "name": "动物", "type": "tags"},
                {"key": "music", "name": "音乐", "type": "tags"},
                {"key": "songs", "name": "歌曲", "type": "tags"},
                {"key": "sounds", "name": "声音", "type": "tags"},
                {"key": "scents", "name": "气味", "type": "tags"},
                {"key": "books", "name": "书籍类型", "type": "tags"},
                {"key": "movies", "name": "电影", "type": "tags"},
                {"key": "sports", "name": "运动", "type": "tags"},
                {"key": "quirks", "name": "小癖好", "type": "textarea"},
            ],
        },
        {
            "key": "dislikes",
            "name": "讨厌",
            "sort": 6,
            "fields": [
                {"key": "foods", "name": "食物", "type": "tags"},
                {"key": "sounds", "name": "噪音", "type": "tags"},
                {"key": "smells", "name": "气味", "type": "tags"},
                {"key": "habits", "name": "习惯", "type": "tags"},
            ],
        },
        {
            "key": "fears",
            "name": "害怕",
            "sort": 7,
            "fields": [
                {"key": "animals", "name": "动物", "type": "tags"},
                {"key": "objects", "name": "物品", "type": "tags"},
                {"key": "atmospheres", "name": "氛围", "type": "tags"},
            ],
        },
        {
            "key": "values",
            "name": "价值观与信念",
            "sort": 8,
            "fields": [
                {"key": "motto", "name": "人生信条", "type": "textarea"},
                {"key": "believes", "name": "相信什么", "type": "tags"},
                {"key": "opposes", "name": "反对什么", "type": "tags"},
                {"key": "goal", "name": "人生目标", "type": "textarea"},
            ],
        },
        {
            "key": "abilities",
            "name": "能力与边界",
            "sort": 9,
            "fields": [
                {"key": "good_at", "name": "擅长的事情", "type": "tags"},
                {"key": "never_do", "name": "绝对不会做的事情", "type": "tags"},
                {"key": "limits", "name": "能力上限", "type": "tags"},
            ],
        },
    ]
}

DEFAULT_TEMPLATE_DEFAULTS = (
    "通用规则：\n"
    "1. 人物身处中国区域内的地球镜像世界，地域、文化、生活规则与现实中国一致\n"
    "2. 年龄在20-29岁之间\n"
    "3. 女性身高160-170cm，男性177-185cm\n"
    "4. 女性体重=身高÷3.6，男性体重=身高÷2.6\n"
    "5. 出生地=居住地=工作地\n"
    "6. 没上全日制大学，偏爱某个领域，跟前辈学习\n"
    "7. 父母健在、性格温和、不喜欢热闹、独生\n"
    "8. 所有内容具体、有画面感，各项内容逻辑自洽"
)


DEFAULT_TEMPLATE_NAME = "标准角色背景模板"

async def ensure_default_template() -> None:
    """启动时确保默认背景模板存在。不创建示例背景 — 背景通过管理后台批量生成。

    对于已存在的模板，做一次性的 best-effort 字面迁移（如 18-27 → 20-29），
    让历史部署也能跟上 spec 调整；admin 已经手动改过的部分（比如自定义其他
    规则文本）保留不动，只替换具体数字字面。
    """
    from prisma import Json

    existing = await db.charactertemplate.find_first(
        where={"name": DEFAULT_TEMPLATE_NAME},
    )
    if existing:
        await _migrate_default_template_age_range(existing)
        return

    await db.charactertemplate.create(
        data={
            "name": DEFAULT_TEMPLATE_NAME,
            "description": "PRD 9大分类完整角色设定，适用于大多数AI伴侣角色",
            "schemaData": Json(DEFAULT_TEMPLATE_SCHEMA),
            "defaults": DEFAULT_TEMPLATE_DEFAULTS,
            "status": "active",
        }
    )
    logger.info("Default character template seeded")


async def _migrate_default_template_age_range(existing) -> None:
    """One-shot migration: 18-27 → 20-29 in defaults text + schema.identity.age."""
    from prisma import Json

    updates: dict = {}

    # 1) defaults 文本里的 "18-27" 字面替换。只动数字，admin 自定义文本保留。
    if existing.defaults and "18-27" in existing.defaults:
        updates["defaults"] = (
            existing.defaults
            .replace("年龄在18-27岁之间", "年龄在20-29岁之间")
            .replace("18-27", "20-29")
        )

    # 2) schemaData.identity.age min/max + hint
    schema = existing.schemaData if isinstance(existing.schemaData, dict) else None
    if schema:
        changed = False
        for cat in schema.get("categories", []):
            if cat.get("key") != "identity":
                continue
            for field in cat.get("fields", []):
                if field.get("key") != "age":
                    continue
                if field.get("min") == 18:
                    field["min"] = 20
                    changed = True
                if field.get("max") == 27:
                    field["max"] = 29
                    changed = True
                if field.get("hint") == "18-27岁之间":
                    field["hint"] = "20-29岁之间"
                    changed = True
        if changed:
            updates["schemaData"] = Json(schema)

    if updates:
        await db.charactertemplate.update(where={"id": existing.id}, data=updates)
        logger.info(
            f"Migrated default character template age range to 20-29 "
            f"(fields: {list(updates.keys())})"
        )


# ── LLM 画像生成 ──

def _build_schema_description(schema: dict) -> str:
    """将 template.schemaData 转换为 LLM 可读的文本描述。"""
    lines = []
    for cat in schema.get("categories", []):
        lines.append(f"\n{cat['name']}（key: {cat['key']}）:")
        for field in cat.get("fields", []):
            parts = [f"  - {field['name']}（key: {field['key']}，类型: {field['type']}）"]
            if field.get("hint"):
                parts.append(f"    提示: {field['hint']}")
            if field.get("options"):
                parts.append(f"    选项: {', '.join(field['options'])}")
            if field.get("required"):
                parts.append("    [必填]")
            lines.append(" ".join(parts) if len(parts) == 1 else "\n".join(parts))
    return "\n".join(lines)


DEFAULT_PROMPT_HEADER = "你是一个AI角色设计师。请根据以下模板结构，生成一个完整的虚构人物。"

# 输出要求模板。{index} 是占位符 — 在 build_generation_prompt 时被替换为实际的角色序号。
DEFAULT_PROMPT_REQUIREMENTS = (
    "要求：\n"
    "1. 所有字段必须填充，内容具体、有画面感\n"
    "2. 各项内容逻辑自洽（职业、教育、喜好、价值观之间要合理关联）\n"
    "3. tags 类型的字段返回字符串数组\n"
    "4. text/textarea 类型的字段返回字符串\n"
    "5. number 类型的字段返回数字\n"
    "6. 每次生成的角色要有独特性，不要雷同\n"
    "7. 这是第 {index} 个角色，请确保与之前的角色有明显差异\n\n"
    "返回JSON，顶层 key 为分类的 key，值为该分类所有字段 key→value 的 dict。\n"
    "例如: {{\"identity\": {{\"name\": \"...\", \"gender\": \"...\", ...}}, \"appearance\": {{...}}, ...}}"
)


def get_default_prompts() -> dict[str, str]:
    """返回系统默认的提示词开头和输出要求，供前端"重置默认"使用。"""
    return {
        "header": DEFAULT_PROMPT_HEADER,
        "requirements": DEFAULT_PROMPT_REQUIREMENTS,
    }


def _format_requirements(requirements_template: str, index: int) -> str:
    """安全替换 {index} 占位符。如果模板里没有 {index}，则原样返回。"""
    try:
        return requirements_template.format(index=index + 1)
    except (KeyError, IndexError, ValueError):
        return requirements_template


def _build_career_section(career: dict) -> str:
    """将职业背景数据格式化为 prompt 注入段落。告知 LLM 围绕此职业生成其他分类，
    但不要求 LLM 生成职业字段本身（后处理直接赋值）。"""
    return (
        "===== 该角色的职业背景（仅供参考，不需要在输出中生成 career 分类）=====\n"
        f"职业: {career.get('title', '')}\n"
        f"工作内容: {career.get('duties', '')}\n"
        f"主要产出物: {career.get('outputs', '')}\n"
        f"社会价值: {career.get('socialValue', career.get('social_value', ''))}\n"
        f"服务对象: {career.get('clients', '')}\n\n"
        "请围绕上述职业合理生成其他分类（身份、外貌、教育、喜好等），使整体逻辑自洽。\n"
        "注意：不要在 JSON 输出中包含 career 分类，该分类由系统自动填充。\n"
    )


def build_generation_prompt(
    schema: dict,
    defaults: str | None,
    index: int = 0,
    *,
    header: str | None = None,
    requirements: str | None = None,
    career: dict | None = None,
    gender: str | None = None,
) -> str:
    """组装完整的角色生成提示词。

    结构: header + [性别约束] + [职业背景设定] + 模板结构 + 生成规则 + 输出要求

    Args:
        schema: 模板的分类/字段定义
        defaults: 模板的生成规则
        index: 当前生成的角色序号
        header: 提示词开头（None = 用 DEFAULT_PROMPT_HEADER）
        requirements: 输出要求（None = 用 DEFAULT_PROMPT_REQUIREMENTS）
        career: 职业背景数据 dict（None = 不注入职业约束）
        gender: "male" / "female" / None — 指定时约束 LLM 整体描写向该性别靠拢
    """
    actual_header = header if header is not None else DEFAULT_PROMPT_HEADER
    actual_requirements = requirements if requirements is not None else DEFAULT_PROMPT_REQUIREMENTS

    # 有职业时，从 schema 中剔除 career 分类 — 不让 LLM 生成职业字段（后处理直接赋值）
    effective_schema = schema
    if career:
        effective_schema = {
            **schema,
            "categories": [
                cat for cat in schema.get("categories", [])
                if cat.get("key") != "career"
            ],
        }
    schema_desc = _build_schema_description(effective_schema)

    prompt = f"{actual_header}\n\n"
    if gender in ("male", "female"):
        gender_zh = "男性" if gender == "male" else "女性"
        prompt += (
            f"【性别约束】该角色必须是{gender_zh}。identity.gender 字段必须填 "
            f"{'“男”' if gender == 'male' else '“女”'},外貌、称呼、兴趣等描写需与该性别一致。\n\n"
        )
    if career:
        prompt += _build_career_section(career) + "\n"
    prompt += f"模板结构：{schema_desc}\n\n"
    if defaults:
        prompt += f"生成规则：\n{defaults}\n\n"
    prompt += _format_requirements(actual_requirements, index)
    return prompt


async def generate_single_profile(
    schema: dict,
    defaults: str | None,
    index: int = 0,
    *,
    header: str | None = None,
    requirements: str | None = None,
    career: dict | None = None,
    gender: str | None = None,
) -> dict:
    """用 LLM 根据模板结构生成一份完整角色背景数据。gender ∈ {"male","female",None}。"""
    prompt = build_generation_prompt(
        schema, defaults, index,
        header=header, requirements=requirements, career=career, gender=gender,
    )

    # 可选：管理员可在「提示词管理」中注册 character.generation key 来覆盖默认 prompt
    try:
        custom_prompt = await get_prompt_text("character.generation")
        if custom_prompt and "{schema}" in custom_prompt:
            schema_desc = _build_schema_description(schema)
            prompt = custom_prompt.format(schema=schema_desc, defaults=defaults or "", index=index + 1)
    except Exception:
        pass

    model = get_utility_model()
    data = await invoke_json(model, prompt)
    return data if isinstance(data, dict) else {}
