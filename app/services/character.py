"""AI 角色背景管理服务。

模板 CRUD、画像批量生成（LLM）、默认模板 seed。
"""

from __future__ import annotations

import logging

from app.db import db
from app.services.llm.models import get_chat_model, invoke_json

logger = logging.getLogger(__name__)


# Spec §1.3: agent 年龄硬区间 20-29.
AGE_MIN = 20
AGE_MAX = 29


def clamp_agent_age(age: int) -> int:
    """把 LLM 反推的年龄钳回 spec §1.3 的 20-29 区间.
    prompt hint 无法保证 LLM 严格输出, 这里是最后防线.
    """
    return max(AGE_MIN, min(AGE_MAX, age))


# ── 默认基础模板 (PRD 9大分类) ──

DEFAULT_TEMPLATE_SCHEMA = {
    "categories": [
        # ═══ 身份层 (映射 memories_ai main_category="身份") ═══
        {
            "key": "identity",
            "name": "基础身份",
            "sort": 1,
            "fields": [
                {"key": "name", "name": "姓名", "type": "text",
                 "hint": "直接引用注入的【姓名】，原样填入不要修改",
                 "_memory_sub": "姓名"},
                {"key": "gender", "name": "性别", "type": "select",
                 "options": ["男", "女"], "required": True,
                 "_memory_sub": "性别"},
                {"key": "age", "name": "年龄", "type": "number",
                 "min": 20, "max": 29, "hint": "20-29岁之间",
                 "_memory_sub": "年龄"},
                {"key": "birthday", "name": "出生日期", "type": "date",
                 "_memory_sub": "生日"},
                {"key": "zodiac", "name": "生肖", "type": "text",
                 "hint": "必须与出生年份对应（鼠/牛/虎/兔/龙/蛇/马/羊/猴/鸡/狗/猪）",
                 "_memory_sub": "生肖"},
                {"key": "constellation", "name": "星座", "type": "text",
                 "hint": "必须与生日月日对应",
                 "_memory_sub": "星座"},
                {"key": "blood_type", "name": "血型", "type": "select",
                 "options": ["O型", "A型", "B型", "AB型"],
                 "required": True,
                 "hint": "必须从 4 个选项中选一个并简述与性格的微关联",
                 "_memory_sub": "血型"},
                {"key": "location", "name": "现居地", "type": "text",
                 "hint": "出生地=居住地=工作地，格式参考居民身份证地址",
                 "_memory_sub": "现居地"},
                {"key": "birthplace", "name": "出生地", "type": "text",
                 "hint": "具体到市/区",
                 "_memory_sub": "出生地"},
                {"key": "growing_up_location", "name": "成长地", "type": "text",
                 "hint": "18岁前长期居住地",
                 "_memory_sub": "成长地"},
                {"key": "family", "name": "亲属关系", "type": "textarea",
                 "hint": "父母职业 + 关系模式 + 是否兄弟姐妹（默认独生）",
                 "_memory_sub": "亲属关系"},
                {"key": "social_relations", "name": "社会关系", "type": "textarea",
                 "hint": "朋友数量质量、同事关系、社交圈层特点",
                 "_memory_sub": "社会关系"},
                {"key": "pet_profile", "name": "宠物", "type": "textarea",
                 "hint": "养宠则写种类+名字+由来；不养则写「无」",
                 "_memory_sub": "宠物"},
                # ethnicity 删除 — 后处理硬写"汉族"对齐 spec
            ],
        },
        {
            "key": "appearance",
            "name": "外貌与形象",
            "sort": 2,
            "_memory_sub": "外貌特征",  # 整分类合成 1 条 身份/外貌特征
            "fields": [
                {"key": "height", "name": "身高", "type": "text",
                 "hint": "男 165-190cm，女 155-175cm，给出具体数值"},
                {"key": "weight", "name": "体型", "type": "text",
                 "hint": "匀称/清瘦/微胖/结实等定性描述，与身高、性别协调"},
                {"key": "features", "name": "外貌特征", "type": "textarea",
                 "hint": "脸型、眉眼、鼻唇特点及整体气质"},
                {"key": "style", "name": "穿搭风格", "type": "textarea",
                 "hint": "日常通勤、居家、社交场合着装偏好"},
                {"key": "voice", "name": "声音特点", "type": "textarea",
                 "hint": "音色、语速、语调习惯"},
            ],
        },
        {
            "key": "education_knowledge",
            "name": "教育与知识",
            "sort": 3,
            "_memory_sub": "教育背景",
            "fields": [
                {"key": "degree", "name": "学历", "type": "textarea",
                 "hint": "没上全日制大学，偏爱本职业相关事情，跟前辈学习"},
                {"key": "strengths", "name": "知识擅长范围", "type": "tags",
                 "hint": "3 个左右专业或知识领域"},
                {"key": "self_taught", "name": "自学特殊技能", "type": "tags",
                 "hint": "1-2 项与职业或兴趣相关的技能"},
            ],
        },
        {
            "key": "career",
            "name": "职业与生存闭环",
            "sort": 4,
            "_memory_sub": "职业/与经济",
            "fields": [
                {"key": "title", "name": "职业", "type": "text"},
                {"key": "duties", "name": "工作内容", "type": "textarea"},
                {"key": "social_value", "name": "社会价值", "type": "textarea"},
                {"key": "clients", "name": "服务对象", "type": "tags"},
                {"key": "income", "name": "经济状况", "type": "text",
                 "hint": "年薪区间，必须 < 10 万"},
            ],
        },

        # ═══ 偏好层 (映射 memories_ai main_category="偏好") ═══
        {
            "key": "likes",
            "name": "喜好",
            "sort": 5,
            "fields": [
                # 饮食喜好
                {"key": "foods", "name": "食物", "type": "tags",
                 "_memory_sub": "饮食喜好"},
                {"key": "fruits", "name": "水果", "type": "tags",
                 "_memory_sub": "饮食喜好"},
                # 审美爱好 13 子项
                {"key": "colors", "name": "颜色", "type": "tags",
                 "_memory_sub": "审美爱好"},
                {"key": "season", "name": "季节", "type": "tags",
                 "_memory_sub": "审美爱好"},
                {"key": "weather", "name": "天气", "type": "tags",
                 "_memory_sub": "审美爱好"},
                {"key": "plants", "name": "植物", "type": "tags",
                 "_memory_sub": "审美爱好"},
                {"key": "animals", "name": "动物", "type": "tags",
                 "_memory_sub": "审美爱好"},
                {"key": "music", "name": "音乐", "type": "tags",
                 "_memory_sub": "审美爱好"},
                {"key": "songs", "name": "歌曲", "type": "tags",
                 "_memory_sub": "审美爱好"},
                {"key": "sounds", "name": "声音", "type": "tags",
                 "_memory_sub": "审美爱好"},
                {"key": "scents", "name": "气味", "type": "tags",
                 "_memory_sub": "审美爱好"},
                {"key": "books", "name": "书籍类型", "type": "tags",
                 "_memory_sub": "审美爱好"},
                {"key": "movies", "name": "电影", "type": "tags",
                 "_memory_sub": "审美爱好"},
                {"key": "sports", "name": "运动", "type": "tags",
                 "_memory_sub": "审美爱好"},
                {"key": "quirks", "name": "小癖好", "type": "textarea",
                 "_memory_sub": "审美爱好"},
            ],
        },
        {
            "key": "dislikes",
            "name": "讨厌",
            "sort": 6,
            "fields": [
                # 饮食厌恶
                {"key": "foods", "name": "食物", "type": "tags",
                 "hint": "含不吃的原因（过敏/口味/心理）",
                 "_memory_sub": "饮食厌恶"},
                # 审美厌恶
                {"key": "sounds", "name": "噪音", "type": "tags",
                 "_memory_sub": "审美厌恶"},
                {"key": "smells", "name": "气味", "type": "tags",
                 "_memory_sub": "审美厌恶"},
            ],
        },
        {
            "key": "interpersonal",
            "name": "人际偏好",
            "sort": 7,
            "fields": [
                {"key": "liked_traits", "name": "人际喜好", "type": "tags",
                 "hint": "欣赏的人际特质",
                 "_memory_sub": "人际喜好"},
                {"key": "disliked_traits", "name": "人际厌恶", "type": "tags",
                 "hint": "反感的行为或性格",
                 "_memory_sub": "人际厌恶"},
            ],
        },
        {
            "key": "lifestyle",
            "name": "生活习惯",
            "sort": 8,
            "_memory_sub": "生活习惯",  # 整分类合成 1 条 偏好/生活习惯
            "fields": [
                {"key": "routine", "name": "作息规律", "type": "textarea"},
                {"key": "hygiene", "name": "卫生习惯", "type": "textarea"},
                {"key": "leisure", "name": "休闲方式", "type": "textarea"},
            ],
        },
        {
            "key": "taboo",
            "name": "禁忌雷区",
            "sort": 9,
            "fields": [
                {"key": "items", "name": "绝对禁忌", "type": "tags",
                 "hint": "不可触碰的底线，3-5 项",
                 "_memory_sub": "禁忌/雷区"},
            ],
        },

        # ═══ 生活层 (映射 memories_ai main_category="生活") ═══
        # 每字段是 tags 数组, 3-5 个具体场景, 每场景 → 1 条独立 memories_ai
        {
            "key": "life_events",
            "name": "生活记忆事件",
            "sort": 10,
            "hint": "每字段返回 3-5 个具体场景字符串数组，每段 50-100 字",
            "fields": [
                {"key": "interaction", "name": "交互事件", "type": "tags",
                 "hint": "与他人产生深度连接或冲突的典型社交事件",
                 "_memory_sub": "交互"},
                {"key": "education", "name": "重要教育事件", "type": "tags",
                 "hint": "求学阶段影响深远的事",
                 "_memory_sub": "教育"},
                {"key": "work", "name": "重要工作事件", "type": "tags",
                 "hint": "职业生涯关键节点",
                 "_memory_sub": "工作"},
                {"key": "travel", "name": "重要旅行事件", "type": "tags",
                 "hint": "印象深刻的旅行经历及途中小故事",
                 "_memory_sub": "旅行"},
                {"key": "living", "name": "重要居住事件", "type": "tags",
                 "hint": "搬家/租房/买房/装修过程的特别记忆",
                 "_memory_sub": "居住"},
                {"key": "health", "name": "重要健康事件", "type": "tags",
                 "hint": "生病/受伤/体检异常及带来的改变",
                 "_memory_sub": "健康"},
                {"key": "pet", "name": "重要宠物事件", "type": "tags",
                 "hint": "不养宠物则写与流浪动物或他人宠物的特殊互动",
                 "_memory_sub": "宠物"},
                {"key": "relationships", "name": "重要人际事件", "type": "tags",
                 "hint": "朋友/恋人/同事关系建立或破裂的转折点",
                 "_memory_sub": "人际"},
                {"key": "skill_learning", "name": "重要技能学习事件", "type": "tags",
                 "hint": "学会某项关键技能的过程和契机",
                 "_memory_sub": "技能"},
                {"key": "life", "name": "重要生活事件", "type": "tags",
                 "hint": "生活中普通但具有象征意义的场景",
                 "_memory_sub": "生活"},
                {"key": "special", "name": "重要其他特殊事件", "type": "tags",
                 "hint": "未归入上述分类但极具个人特色的偶然事件",
                 "_memory_sub": "其他特殊事件"},
            ],
        },

        # ═══ 情绪层 (映射 memories_ai main_category="情绪") ═══
        {
            "key": "emotion_events",
            "name": "情绪记忆事件",
            "sort": 11,
            "hint": "每字段返回 3-5 个具体情境字符串数组，每段 30-80 字",
            "fields": [
                {"key": "happy", "name": "重要高兴的记忆", "type": "tags",
                 "_memory_sub": "高兴"},
                {"key": "sad", "name": "重要悲伤的记忆", "type": "tags",
                 "_memory_sub": "悲伤"},
                {"key": "angry", "name": "重要愤怒的记忆", "type": "tags",
                 "_memory_sub": "愤怒"},
                {"key": "fear", "name": "重要恐惧的记忆", "type": "tags",
                 "hint": "必须明确体现害怕的【动物】【物品】或【氛围】",
                 "_memory_sub": "恐惧"},
                {"key": "disgust", "name": "重要厌恶的记忆", "type": "tags",
                 "_memory_sub": "厌恶"},
                {"key": "anxiety", "name": "重要焦虑的记忆", "type": "tags",
                 "_memory_sub": "焦虑"},
                {"key": "disappointment", "name": "重要失望的记忆", "type": "tags",
                 "_memory_sub": "失望"},
                {"key": "pride", "name": "重要自豪的记忆", "type": "tags",
                 "_memory_sub": "自豪"},
                {"key": "moved", "name": "重要感动的记忆", "type": "tags",
                 "_memory_sub": "感动"},
                {"key": "embarrassed", "name": "重要尴尬的记忆", "type": "tags",
                 "_memory_sub": "尴尬"},
                {"key": "regret", "name": "重要遗憾的记忆", "type": "tags",
                 "_memory_sub": "遗憾"},
                {"key": "lonely", "name": "重要孤独的记忆", "type": "tags",
                 "_memory_sub": "孤独"},
                {"key": "surprised", "name": "重要惊讶的记忆", "type": "tags",
                 "_memory_sub": "惊讶"},
                {"key": "grateful", "name": "重要感激的记忆", "type": "tags",
                 "_memory_sub": "感激"},
                {"key": "relieved", "name": "重要释怀的记忆", "type": "tags",
                 "_memory_sub": "释怀"},
            ],
        },

        # ═══ 思维层 (映射 memories_ai main_category="思维") ═══
        {
            "key": "values",
            "name": "价值观与信念",
            "sort": 12,
            "fields": [
                {"key": "motto", "name": "人生观", "type": "textarea",
                 "hint": "对生命意义、生活方式的基本看法",
                 "_memory_sub": "人生观"},
                {"key": "believes", "name": "相信什么", "type": "tags",
                 "_memory_sub": "价值观"},
                {"key": "opposes", "name": "反对什么", "type": "tags",
                 "_memory_sub": "价值观"},
                {"key": "worldview", "name": "世界观", "type": "textarea",
                 "hint": "对世界运行规则的理解和态度（宏观视角）",
                 "_memory_sub": "世界观"},
                {"key": "goal", "name": "理想与目标", "type": "textarea",
                 "hint": "短期 1-3 年 + 长期 5-10 年",
                 "_memory_sub": "理想与目标"},
                {"key": "interpersonal_view", "name": "人际关系观", "type": "textarea",
                 "hint": "对亲情、友情、爱情的理解及处理原则",
                 "_memory_sub": "人际关系观"},
                {"key": "social_view", "name": "社会观点", "type": "textarea",
                 "hint": "对当下社会热点或普遍现象的 1-2 个具体看法",
                 "_memory_sub": "社会观点"},
                {"key": "faith", "name": "信仰/精神寄托", "type": "textarea",
                 "hint": "必填；即使无宗教信仰也要写精神寄托（艺术/自然/家人/事业等）",
                 "_memory_sub": "信仰/寄托"},
            ],
        },
        {
            "key": "abilities",
            "name": "能力与边界",
            "sort": 13,
            "_memory_sub": "自我认知",  # 整分类合成 1 条 思维/自我认知
            "fields": [
                {"key": "good_at", "name": "擅长的事情", "type": "tags",
                 "hint": "3-5 项具体强项"},
                {"key": "never_do", "name": "绝对不会做的事情", "type": "tags",
                 "hint": "道德或原则底线上的禁止行为"},
                {"key": "limits", "name": "能力上限", "type": "tags",
                 "hint": "坦诚不擅长或感到吃力的领域"},
            ],
        },
    ]
}

DEFAULT_TEMPLATE_DEFAULTS = (
    "通用规则：\n"
    "1. 人物身处中国区域内的地球镜像世界，地域、文化、生活规则与现实中国一致\n"
    "2. 所有内容必须符合社会主义核心价值观与公序良俗\n"
    "3. 年龄在 20-29 岁之间\n"
    "4. 男性身高 165-190cm，女性身高 155-175cm，给出具体数值\n"
    "5. 体型描述与身高、性别协调（匀称/清瘦/微胖/结实等定性词）\n"
    "6. 出生地=居住地=工作地\n"
    "7. 没上全日制大学，偏爱某个领域，跟前辈学习\n"
    "8. 父母健在、性格温和、不喜欢热闹、独生\n"
    "9. 生肖必须与出生年份对应；星座必须与生日月日对应\n"
    "10. 血型选定后简述与性格的微关联\n"
    "11. 内容与输入的性格 7 维分数深度绑定，描写要明显体现该性格分布\n"
    "12. 时间线/年龄/生肖/星座/经历需全部逻辑自洽，不得互相矛盾\n"
    "13. life_events 每类 3-5 个具体片段（不少于 3、不多于 5），每段独立完整场景，"
    "时间线按当前年龄反推（教育事件多在 18-22 岁，工作事件在 22-当前岁之间）\n"
    "14. emotion_events 每类 3-5 个具体情境；fear 字段必须明确害怕的动物/物品/氛围\n"
    "15. 经济状况按注入的职业 income 字段，年收入 < 10 万\n"
    "16. 严禁留空或使用「暂无」「无」等占位符（除 pet_profile 不养时写「无」外）"
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
        await _migrate_default_template(existing)
        return

    await db.charactertemplate.create(
        data={
            "name": DEFAULT_TEMPLATE_NAME,
            "description": "PRD 13大分类完整角色设定 (含 26 项过去事件), 适用于大多数AI伴侣角色",
            "schemaData": Json(DEFAULT_TEMPLATE_SCHEMA),
            "defaults": DEFAULT_TEMPLATE_DEFAULTS,
            "status": "active",
        }
    )
    logger.info("Default character template seeded (v2)")


def _is_current_schema(schema: dict | None) -> bool:
    """当前 schema 标志 (v2.1):
    - 包含 life_events / emotion_events 分类
    - career 不再有 output 字段 (4.21 spec 删除)
    - dislikes 不再有 habits 字段 (不在 spec)
    """
    if not isinstance(schema, dict):
        return False
    cats = schema.get("categories", [])
    keys = {cat.get("key") for cat in cats}
    if not ("life_events" in keys and "emotion_events" in keys):
        return False
    for cat in cats:
        if cat.get("key") == "career":
            if any(f.get("key") == "output" for f in cat.get("fields", [])):
                return False
        if cat.get("key") == "dislikes":
            if any(f.get("key") == "habits" for f in cat.get("fields", [])):
                return False
    return True


async def _migrate_default_template(existing) -> None:
    """One-shot 迁移: 旧 schema → 最新版本.

    检测当前版本标志缺失时全量覆盖默认模板的 schemaData + defaults + description.
    用户改过的非默认模板 (status != active 或 name != DEFAULT_TEMPLATE_NAME) 不动.
    """
    from prisma import Json

    schema = existing.schemaData if isinstance(existing.schemaData, dict) else None
    if _is_current_schema(schema):
        return  # 已是最新

    await db.charactertemplate.update(
        where={"id": existing.id},
        data={
            "schemaData": Json(DEFAULT_TEMPLATE_SCHEMA),
            "defaults": DEFAULT_TEMPLATE_DEFAULTS,
            "description": "PRD 13大分类完整角色设定 (含 26 项过去事件), 适用于大多数AI伴侣角色",
        },
    )
    logger.info(
        f"Migrated default character template to v2 schema "
        f"({len(DEFAULT_TEMPLATE_SCHEMA['categories'])} categories, "
        f"{sum(len(c['fields']) for c in DEFAULT_TEMPLATE_SCHEMA['categories'])} fields)"
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


# 默认 header / requirements 真源已迁到 prompting registry
# (character.template_header / character.template_requirements):
# - admin「提示词管理」UI 可编辑 → 写 Redis + DB
# - admin「Agent管理 / 背景模板」UI 也读这两个 key, 「重置默认」按钮拿 registry 值
# - per-template promptHeader / promptRequirements 仍可覆盖
# 这里保留极简静态 fallback 应对 registry 启动期不可达的极端情况。
_FALLBACK_PROMPT_HEADER = "你是一个AI人格构建专家。请根据基础信息构建虚拟人格档案。"
_FALLBACK_PROMPT_REQUIREMENTS = "返回JSON，顶层 key 为分类 key，值为字段 key→value dict。"


async def get_default_prompts() -> dict[str, str]:
    """返回系统默认的提示词开头和输出要求 (从 registry 读取).

    供 admin「Agent管理 / 背景模板」的「重置默认」按钮 + per-template fallback 共用.
    """
    from app.services.prompting.store import get_prompt_text

    try:
        header = await get_prompt_text("character.template_header")
    except Exception as e:
        logger.warning(f"character.template_header registry read failed: {e}")
        header = _FALLBACK_PROMPT_HEADER
    try:
        requirements = await get_prompt_text("character.template_requirements")
    except Exception as e:
        logger.warning(f"character.template_requirements registry read failed: {e}")
        requirements = _FALLBACK_PROMPT_REQUIREMENTS
    return {"header": header, "requirements": requirements}


def _format_requirements(requirements_template: str, index: int) -> str:
    """安全替换 {index} 占位符。如果模板里没有 {index}，则原样返回。"""
    try:
        return requirements_template.format(index=index + 1)
    except (KeyError, IndexError, ValueError):
        return requirements_template


def _build_career_section(career: dict) -> str:
    """将职业背景数据格式化为 prompt 注入段落。告知 LLM 围绕此职业生成其他分类，
    但不要求 LLM 生成职业字段本身（后处理直接赋值）。

    income 缺省时用合规默认值: spec PDF 要求年收入 < 10 万。
    """
    income = career.get("income") or "年薪 5-10 万"
    social_value = career.get("socialValue", career.get("social_value", ""))
    return (
        "===== 该角色的职业背景（仅供参考，不需要在输出中生成 career 分类）=====\n"
        f"职业: {career.get('title', '')}\n"
        f"工作内容: {career.get('duties', '')}\n"
        f"社会价值: {social_value}\n"
        f"服务对象: {career.get('clients', '')}\n"
        f"经济状况: {income}\n\n"
        "请围绕上述职业合理生成其他分类（身份、外貌、教育、喜好等），使整体逻辑自洽。\n"
        "注意：不要在 JSON 输出中包含 career 分类，该分类由系统自动填充。\n"
    )


_PERSONALITY_DIM_LABELS: dict[str, str] = {
    "liveliness": "活泼度",
    "rationality": "理性度",
    "sensitivity": "感性度",
    "planning": "计划度",
    "spontaneity": "随性度",
    "imagination": "脑洞度",
    "humor": "幽默度",
}


def _build_name_section(name: str) -> str:
    """注入用户填的姓名 (PDF #34 输入变量第一项: 直接引用)."""
    return (
        f"【姓名】{name}\n"
        f"请把 identity.name 字段直接填为「{name}」，原样不要修改。"
        "可在 appearance.features / appearance.voice 等字段中适当呼应这个名字的气质。\n"
    )


def _build_personality_section(personality: dict) -> str:
    """注入性格 7 维分数 (PDF #34 生成要求: 与输入性格深度绑定)."""
    lines = ["【性格 7 维分数】（0-100，用于深度绑定描写风格、喜好、价值观）"]
    for key, label in _PERSONALITY_DIM_LABELS.items():
        value = personality.get(key)
        if value is None:
            value = personality.get(key.rstrip("ness").rstrip("ity"))  # 兼容简写命名
        if value is None:
            value = 50
        lines.append(f"- {label}（{key}）：{value}")
    lines.append(
        "请让 likes / dislikes / values / abilities / appearance.style / "
        "life_events / emotion_events 等字段的描写明显体现这套性格分布。"
    )
    return "\n".join(lines) + "\n"


def build_generation_prompt(
    schema: dict,
    defaults: str | None,
    index: int = 0,
    *,
    header: str | None = None,
    requirements: str | None = None,
    career: dict | None = None,
    gender: str | None = None,
    name: str | None = None,
    personality: dict | None = None,
) -> str:
    """组装完整的角色生成提示词 (对齐 PDF #34 + taxonomy v2).

    结构: header + 性别约束 + [姓名] + [性格 7 维] + [职业背景] + 模板结构
          + 生成规则 + 输出要求

    Args:
        schema: 模板的分类/字段定义
        defaults: 模板的生成规则
        index: 当前生成的角色序号
        header: 提示词开头. 调用方应已通过 get_default_prompts() 从 registry
                解析过；None 时用极简静态 fallback 应对启动期 registry 不可达。
        requirements: 输出要求, 同 header.
        career: 职业背景数据 dict（None = 不注入职业约束）
        gender: "male" / "female" / None
        name: 角色姓名 (PDF #34 输入第一项: "AI 自我姓名: 直接引用")
        personality: 性格 7 维 dict (liveliness/rationality/sensitivity/...)
    """
    actual_header = header if header is not None else _FALLBACK_PROMPT_HEADER
    actual_requirements = requirements if requirements is not None else _FALLBACK_PROMPT_REQUIREMENTS

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
            f"{'「男」' if gender == 'male' else '「女」'}，"
            "外貌、称呼、兴趣等描写需与该性别一致。\n\n"
        )
    if name:
        prompt += _build_name_section(name) + "\n"
    if personality:
        prompt += _build_personality_section(personality) + "\n"
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
    name: str | None = None,
    personality: dict | None = None,
    agent_name: str | None = None,
) -> dict:
    """用 LLM 根据模板结构生成一份完整角色背景数据。

    name vs agent_name: name 注入 prompt 让 LLM 看到; agent_name 后处理硬写
    identity.name (PDF #34 第一维 #1 "直接引用"). 通常两者相同, 但兼容
    旧调用方仅传 agent_name 不传 name 的情况.
    """
    effective_name = name or agent_name
    # header / requirements 解析顺序:
    #   per-template 覆盖 (caller 传入非空)  >  registry default (character.template_*)
    #   >  _FALLBACK_PROMPT_*  (registry 不可达极端兜底)
    # 整段 prompt 由 schema/career/name/personality 多段运行时拼装, 拼装结构
    # 不进 registry; 但顶部规则 / 底部要求两段进 registry, admin 双 UI 共享一份。
    # 用 truthy 判断而非 `is None`: 兼容 admin 在背景模板页保存了空字符串覆盖
    # 的情况——空覆盖应回退 registry default, 而不是直接送空给 LLM。
    if not header or not requirements:
        resolved = await get_default_prompts()
        header = header or resolved["header"]
        requirements = requirements or resolved["requirements"]
    prompt = build_generation_prompt(
        schema, defaults, index,
        header=header, requirements=requirements,
        career=career, gender=gender,
        name=effective_name, personality=personality,
    )

    # Schema v2: prompt ~6K + 输出 JSON ~3-4K (含 26 项过去事件 × 3-5 场景),
    # 远超 utility_fast 8s timeout. 用 chat 大模型 + background profile (120s).
    model = get_chat_model()
    data = await invoke_json(model, prompt, profile="background")
    if not isinstance(data, dict):
        return {}

    # 缺字段 repair: schema v2 满输出可能擦边 max_tokens, 偶发末尾字段截断.
    # 检测后只对缺失字段发一次 follow-up 调用 (输出 1-2K, 不会再截断).
    missing = _detect_missing_fields(schema, data)
    if missing:
        logger.warning(f"character profile missing {len(missing)} fields, running repair")
        repaired = await _repair_missing_fields(schema, data, missing, model)
        for cat_key, fields in repaired.items():
            data.setdefault(cat_key, {}).update(fields)

    return _apply_postprocess_overrides(
        data, agent_name=effective_name, career=career,
    )


# 这些字段由 _apply_postprocess_overrides 硬覆盖, 不参与 repair 检测.
_REPAIR_SKIP_CATEGORIES = frozenset({"career"})
_REPAIR_SKIP_IDENTITY_FIELDS = frozenset({"name", "ethnicity", "gender", "blood_type"})

_BLOOD_TYPE_OPTIONS = ("O型", "A型", "B型", "AB型")


def _is_field_empty(value: object, field_type: str) -> bool:
    """判定 LLM 输出字段是否为空 (需要 repair)."""
    if value is None:
        return True
    if field_type in ("tags",):
        return not isinstance(value, list) or len(value) == 0
    if field_type in ("text", "textarea", "select", "date"):
        return not isinstance(value, str) or not value.strip()
    if field_type == "number":
        return not isinstance(value, (int, float))
    return value in (None, "", [], {})


def _detect_missing_fields(
    schema: dict, data: dict,
) -> list[tuple[str, dict]]:
    """返回 [(category_key, field_dict), ...], 跳过 career 与 identity 硬覆盖字段."""
    missing: list[tuple[str, dict]] = []
    for cat in schema.get("categories", []):
        cat_key = cat.get("key")
        if not cat_key or cat_key in _REPAIR_SKIP_CATEGORIES:
            continue
        cat_data = data.get(cat_key)
        cat_data = cat_data if isinstance(cat_data, dict) else {}
        for field in cat.get("fields", []):
            field_key = field.get("key")
            if not field_key:
                continue
            if cat_key == "identity" and field_key in _REPAIR_SKIP_IDENTITY_FIELDS:
                continue
            if _is_field_empty(cat_data.get(field_key), field.get("type", "text")):
                missing.append((cat_key, field))
    return missing


def _build_repair_persona_summary(data: dict) -> str:
    """已生成 profile 的浓缩单行摘要, 注入 character.repair_missing_fields prompt."""
    identity = data.get("identity", {}) if isinstance(data.get("identity"), dict) else {}
    career = data.get("career", {}) if isinstance(data.get("career"), dict) else {}
    return (
        f"姓名: {identity.get('name', '')}; "
        f"性别: {identity.get('gender', '')}; "
        f"年龄: {identity.get('age', '')}; "
        f"职业: {career.get('title', '')}; "
        f"现居地: {identity.get('location', '')}"
    )


def _build_repair_missing_fields_text(
    schema: dict, missing: list[tuple[str, dict]],
) -> str:
    """缺字段清单浓缩文本, 注入 character.repair_missing_fields prompt."""
    by_cat: dict[str, list[dict]] = {}
    for cat_key, field in missing:
        by_cat.setdefault(cat_key, []).append(field)

    cat_name_map = {cat["key"]: cat.get("name", cat["key"]) for cat in schema.get("categories", [])}
    cat_hint_map = {cat["key"]: cat.get("hint", "") for cat in schema.get("categories", [])}

    lines: list[str] = []
    for cat_key, fields in by_cat.items():
        cat_label = cat_name_map.get(cat_key, cat_key)
        cat_hint = cat_hint_map.get(cat_key, "")
        lines.append(f"[{cat_label}]（key: {cat_key}）" + (f" — {cat_hint}" if cat_hint else ""))
        for f in fields:
            line = f"  - {f.get('name')}（key: {f.get('key')}, 类型: {f.get('type')}）"
            # select 字段必须把 options 透传给 LLM, 否则 LLM 不知道有效值是什么
            # → 选填 / 跳过 (例如 blood_type 经常空着). 直接列出可选枚举强制选一个。
            opts = f.get("options")
            if isinstance(opts, list) and opts:
                line += f"; 必须从以下选项中严格选一个: {', '.join(map(str, opts))}"
            if f.get("hint"):
                line += f"; 提示: {f['hint']}"
            lines.append(line)
    return "\n".join(lines)


async def _repair_missing_fields(
    schema: dict, data: dict, missing: list[tuple[str, dict]], model,
) -> dict[str, dict]:
    """调 LLM 补缺. 失败返回空 dict (不抛, 让上层接受不完整结果)."""
    from app.services.prompting.store import get_prompt_text

    template = await get_prompt_text("character.repair_missing_fields")
    prompt = template.format(
        persona_summary=_build_repair_persona_summary(data),
        missing_fields=_build_repair_missing_fields_text(schema, missing),
    )
    try:
        result = await invoke_json(model, prompt, profile="background")
    except Exception as e:
        logger.warning(f"character repair LLM call failed: {e}")
        return {}
    if not isinstance(result, dict):
        return {}
    # 过滤: 只保留我们要求的 (cat_key, field_key) 组合, 防 LLM 多嘴
    wanted: dict[str, set[str]] = {}
    for cat_key, field in missing:
        fk = field.get("key")
        if fk:
            wanted.setdefault(cat_key, set()).add(fk)
    cleaned: dict[str, dict] = {}
    for cat_key, fields in result.items():
        if cat_key not in wanted or not isinstance(fields, dict):
            continue
        sub: dict = {}
        for fk, fv in fields.items():
            if fk in wanted[cat_key]:
                sub[fk] = fv
        if sub:
            cleaned[cat_key] = sub
    return cleaned


def _split_clients(clients) -> list[str]:
    """职业服务对象: 既兼容 list (新数据) 又兼容字符串 (DB 原始字段, 含 、，；\\n 分隔符)."""
    if isinstance(clients, list):
        return [str(s).strip() for s in clients if str(s).strip()]
    if not clients:
        return []
    val = str(clients)
    for sep in ("、", "，", "；", "\n"):
        val = val.replace(sep, ",")
    return [s.strip() for s in val.split(",") if s.strip()]


def _apply_postprocess_overrides(
    profile: dict,
    *,
    agent_name: str | None,
    career: dict | None,
) -> dict:
    """LLM 输出后强制覆盖几个字段, 保证 spec 一致性:
    - identity.ethnicity 硬写「汉族」(spec PDF #34 第一维 #8: "汉族")
    - identity.blood_type LLM 偶尔跳过 (即使 required + repair 也漏); spec
      PDF #34 第一维 #7 原文是"随机生成 A/B/O/AB 型之一", 本就是后端兜底字
      段, 这里若 LLM 给的不是 4 选项之一就随机补 (类比 gender / ethnicity
      在 admin 层面的硬覆盖)
    - identity.name 直接引用注入的姓名 (PDF #34 第一维 #1 "直接引用");
      未注入 (admin 背景池批量生成无姓名) 时清空, 防 LLM 自填随机名进 DB
    - career 分类用预设池数据回填 (LLM 不生成 career, 见 build_generation_prompt)
    """
    import random as _r

    identity = profile.setdefault("identity", {})
    identity["ethnicity"] = "汉族"

    blood = identity.get("blood_type")
    if blood not in _BLOOD_TYPE_OPTIONS:
        identity["blood_type"] = _r.choice(_BLOOD_TYPE_OPTIONS)

    if agent_name:
        identity["name"] = agent_name
    else:
        # 背景池场景: 名字由用户在 agent 创建页输入, 这里清掉 LLM 自填的随机名
        identity.pop("name", None)
    if career:
        profile["career"] = {
            "title": career.get("title"),
            "duties": career.get("duties"),
            "social_value": career.get("socialValue") or career.get("social_value") or "",
            "clients": _split_clients(career.get("clients")),
            "income": career.get("income") or "年薪 5-10 万",
        }
    return profile
