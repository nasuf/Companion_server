from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UserTag:
    label: str
    score: int
    source: str


_CATEGORY_LABELS = {
    "姓名",
    "性别",
    "年龄",
    "职业",
    "职业与经济",
    "职业/与经济",
    "经济",
    "禁忌",
    "雷区",
    "禁忌/雷区",
    "人生观",
    "价值观",
    "信仰",
    "寄托",
    "信仰/寄托",
    "理想",
    "目标",
    "理想与目标",
    "身份",
    "情绪",
    "偏好边界",
    "生活",
    "思维",
    "其他",
    "日常",
}

_SENSITIVE_PATTERNS = (
    "轻生",
    "告别",
    "辱骂",
    "愤怒",
    "吵架",
    "冷战",
    "分手",
    "前女友",
    "嫌他穷",
    "台湾",
)

_RULES: tuple[tuple[str, int, tuple[str, ...]], ...] = (
    ("咖啡成瘾", 110, ("咖啡", "咖啡馆", "咖啡师")),
    ("爱喝茶", 90, ("喝茶", "茶")),
    ("甜品控", 90, ("甜品", "蛋糕", "奶茶", "糖水")),
    ("菠萝爱好者", 84, ("菠萝",)),
    ("音乐爱好者", 108, ("歌曲", "音乐", "一起听", "爵士乐", "歌手")),
    ("独立音乐", 94, ("独立音乐", "ambient", "sion")),
    ("文艺片爱好者", 96, ("电影", "穆赫兰道", "文艺片")),
    ("艺术展爱好者", 96, ("艺术展", "展览", "美术馆")),
    ("爱逛书店", 102, ("书店", "绝版书", "买书")),
    ("阅读收藏癖", 88, ("绝版书", "封面图片", "秦二世必须死")),
    ("喜欢户外", 96, ("爬山", "户外", "旅游", "旅行")),
    ("爱探店", 80, ("西餐店", "餐厅", "咖啡馆")),
    ("个人开发者", 112, ("个人开发者", "开发者")),
    ("法律人", 88, ("律师", "法律")),
    ("独立游戏创作", 112, ("视觉小说", "Renpy", "游戏", "首周销量")),
    ("AI工具玩家", 98, ("Codex", "Claude", "Opus", "token", "Skill")),
    ("创作者心态", 82, ("打赏", "销量", "回本", "创作")),
    ("创业中", 86, ("创业",)),
    ("深夜话痨", 74, ("深夜聊天", "半夜聊天", "凌晨聊天")),
    ("慢热型", 70, ("慢热",)),
)


def derive_user_tags(rows: list[Any], *, limit: int = 9) -> list[str]:
    candidates: dict[str, UserTag] = {}
    for row in rows:
        main = _field(row, "main_category", "mainCategory")
        sub = _field(row, "sub_category", "subCategory")
        text = _memory_text(row)
        if not _is_tag_source_allowed(text, str(main or ""), str(sub or "")):
            continue
        for tag in _tags_for_memory(text, str(main or ""), str(sub or "")):
            existing = candidates.get(tag.label)
            if existing is None or tag.score > existing.score:
                candidates[tag.label] = tag
    ordered = sorted(candidates.values(), key=lambda item: (-item.score, item.label))
    return [item.label for item in ordered[:limit]]


def _tags_for_memory(text: str, main_category: str, sub_category: str) -> list[UserTag]:
    tags: list[UserTag] = []
    normalized = text.lower()
    for label, score, keywords in _RULES:
        if any(keyword.lower() in normalized for keyword in keywords):
            tags.append(UserTag(label=label, score=score + _category_bonus(main_category), source=text))
    return tags


def _is_tag_source_allowed(text: str, main_category: str, sub_category: str) -> bool:
    if not text or _is_category_label(text):
        return False
    if sub_category in {"提醒", "姓名", "年龄", "性别", "生日", "重要日期"}:
        return False
    if any(pattern in text for pattern in _SENSITIVE_PATTERNS):
        return False
    return main_category in {"偏好", "生活", "身份", "思维"} or sub_category in {
        "审美爱好",
        "饮食喜好",
        "生活习惯",
        "旅行",
        "工作",
        "职业/与经济",
        "职业与经济",
    }


def _category_bonus(main_category: str) -> int:
    return {
        "偏好": 8,
        "生活": 4,
        "身份": 2,
        "思维": 1,
    }.get(main_category, 0)


def _memory_text(row: Any) -> str:
    summary = str(_field(row, "summary") or "").strip()
    content = str(_field(row, "content") or "").strip()
    if summary and not _is_category_label(summary):
        return summary
    return content


def _is_category_label(value: str) -> bool:
    normalized = value.strip("：:，,、 的").replace("／", "/")
    return normalized in _CATEGORY_LABELS


def _field(row: Any, snake: str, camel: str | None = None) -> Any:
    if isinstance(row, dict):
        if snake in row:
            return row[snake]
        if camel and camel in row:
            return row[camel]
        return None
    if hasattr(row, snake):
        return getattr(row, snake)
    if camel and hasattr(row, camel):
        return getattr(row, camel)
    return None
