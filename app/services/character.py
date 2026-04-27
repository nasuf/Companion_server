"""AI 角色背景辅助工具.

Plan B 后, 本模块仅保留单步生成流程 (character_generation.py) 仍依赖的工具:
- clamp_agent_age           : age 钳到 spec §1.3 区间 [20, 29]
- _detect_missing_fields    : LLM 输出 → 缺字段列表
- _repair_missing_fields    : 调 LLM 补缺
- _apply_postprocess_overrides : ethnicity 汉族 / blood_type 4 选 1 / name / career 硬覆盖

旧 admin 「背景模板」「背景管理」批量预生成路径 (DEFAULT_TEMPLATE_SCHEMA /
ensure_default_template / build_generation_prompt / generate_single_profile /
get_default_prompts ...) 已随 Plan B 删除。
"""

from __future__ import annotations

import logging

from app.services.llm.models import invoke_json
from app.services.memory.demographics import sample_blood_type

logger = logging.getLogger(__name__)


# Spec §1.3: agent 年龄硬区间 20-29.
AGE_MIN = 20
AGE_MAX = 29


def clamp_agent_age(age: int) -> int:
    """把 LLM 反推的年龄钳回 spec §1.3 的 20-29 区间.
    prompt hint 无法保证 LLM 严格输出, 这里是最后防线.
    """
    return max(AGE_MIN, min(AGE_MAX, age))


# ── LLM 输出校验 + repair ──

# 这些字段由 _apply_postprocess_overrides 硬覆盖, 不参与 repair 检测.
_REPAIR_SKIP_CATEGORIES = frozenset({"career"})
_REPAIR_SKIP_IDENTITY_FIELDS = frozenset({"name", "ethnicity", "gender", "blood_type"})

_BLOOD_TYPE_OPTIONS = ("O型", "A型", "B型", "AB型")


def _is_field_empty(value: object, field_type: str, min_items: int = 1) -> bool:
    """判定 LLM 输出字段是否为空 / 不满足最小条数 (需要 repair).

    For tag/list fields, also flag when len(value) < min_items so the repair
    pass is triggered to top up the list (spec §1.4 demands minimum coverage
    counts per L1 sub-category — emotion 2-3, life events 3-5 etc).
    """
    if value is None:
        return True
    if field_type in ("tags",):
        if not isinstance(value, list):
            return True
        return len(value) < max(1, min_items)
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
            min_items = int(field.get("min_items", 1) or 1)
            if _is_field_empty(
                cat_data.get(field_key), field.get("type", "text"), min_items=min_items,
            ):
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
            opts = f.get("options")
            if isinstance(opts, list) and opts:
                line += f"; 必须从以下选项中严格选一个: {', '.join(map(str, opts))}"
            # 列表字段透传最小条数: _is_field_empty 用 min_items 触发 repair,
            # 但不告诉 LLM 它就会再返回 1 条 → 死循环陷阱. 显式说至少几条.
            min_items = f.get("min_items")
            if isinstance(min_items, int) and min_items > 1:
                line += f"; 至少 {min_items} 条"
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
    - identity.ethnicity 硬写「汉族」(spec PDF #34 第一维 #8)
    - identity.blood_type 必须 4 选 1 (spec PDF #34 第一维 #7 后端兜底字段);
      用 demographics.sample_blood_type 拿到 O/A/B/AB 加权分布 (34/31/27/8)
      而不是均匀 random.choice — 与 sample_ethnicity 的策略对齐
    - identity.name 直接引用注入的姓名 (PDF #34 第一维 #1)
    - career 分类用预设池数据回填 (Plan B prompt 不让 LLM 生 career)
    """
    identity = profile.setdefault("identity", {})
    identity["ethnicity"] = "汉族"

    blood = identity.get("blood_type")
    if blood not in _BLOOD_TYPE_OPTIONS:
        identity["blood_type"] = sample_blood_type(agent_name)

    if agent_name:
        identity["name"] = agent_name
    else:
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
