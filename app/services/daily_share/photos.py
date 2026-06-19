from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from typing import Any

from app.db import db
from app.models.daily_share import DailySharePhoto, DailySharePhotoGroup


_GROUP_DEFS = [
    {
        "id": "evening-light",
        "title": "傍晚光线",
        "subtitle": "适合一句很短的晚安。",
        "signals": (
            ("傍晚", 4),
            ("黄昏", 4),
            ("夕阳", 4),
            ("晚霞", 4),
            ("落日", 4),
            ("日落", 4),
            ("晚安", 3),
            ("夜景", 3),
            ("夜晚", 2),
            ("天空", 1),
            ("光线", 1),
            ("阳光", 1),
        ),
    },
    {
        "id": "desk-fragments",
        "title": "桌面碎片",
        "subtitle": "咖啡、书页、拍立得和没收好的耳机。",
        "signals": (
            ("桌面", 4),
            ("书桌", 4),
            ("办公桌", 4),
            ("咖啡", 3),
            ("杯", 2),
            ("书页", 3),
            ("书", 2),
            ("纸", 2),
            ("笔记", 3),
            ("电脑", 3),
            ("键盘", 3),
            ("耳机", 3),
            ("室内", 1),
        ),
    },
    {
        "id": "on-the-road",
        "title": "路上看到",
        "subtitle": "可以整理成一张“今天经过这里”的卡。",
        "signals": (
            ("户外", 4),
            ("风景", 4),
            ("自然", 3),
            ("旅途", 4),
            ("旅行", 4),
            ("路上", 4),
            ("道路", 3),
            ("街道", 3),
            ("街", 2),
            ("公路", 3),
            ("山", 3),
            ("海", 3),
            ("湖", 3),
            ("瀑布", 4),
            ("花海", 4),
            ("花田", 4),
            ("大片花", 4),
            ("花丛", 3),
            ("草地", 3),
            ("草丛", 2),
            ("城市", 2),
            ("建筑", 2),
            ("车辆", 2),
            ("车", 1),
        ),
    },
    {
        "id": "little-things",
        "title": "小物件",
        "subtitle": "不用解释也能知道今天怎么过的。",
        "signals": (
            ("小物", 4),
            ("物件", 4),
            ("玩具", 4),
            ("摆件", 4),
            ("特写", 2),
            ("盆栽", 3),
            ("花瓶", 3),
            ("单朵", 3),
            ("鲜花", 2),
            ("食物", 3),
            ("餐", 2),
            ("猫", 3),
            ("狗", 3),
            ("手", 2),
        ),
    },
]

_FALLBACK_GROUP = {
    "id": "recent-photos",
    "title": "最近照片",
    "subtitle": "先把能分享的画面收在这里。",
    "signals": (),
}


async def list_user_photo_groups(
    user_id: str,
    *,
    limit: int | None = None,
) -> list[DailySharePhotoGroup]:
    query = """
    SELECT
      a.id,
      a.message_id,
      a.conversation_id,
      a.name,
      a.mime,
      a.size,
      a.width,
      a.height,
      a.url,
      a.vision_summary,
      COALESCE(m.created_at, a.created_at) AS created_at
    FROM chat_message_attachments a
    JOIN messages m ON m.id = a.message_id
    JOIN conversations c ON c.id = a.conversation_id
    WHERE a.user_id = $1
      AND c.user_id = $1
      AND c.is_deleted = FALSE
      AND m.role = 'user'
      AND a.kind = 'image'
      AND a.message_id IS NOT NULL
    ORDER BY COALESCE(m.created_at, a.created_at) DESC, a.created_at DESC
    """
    if limit is None:
        rows = await db.query_raw(query, user_id)
    else:
        rows = await db.query_raw(f"{query}\nLIMIT $2", user_id, max(1, min(limit, 1000)))
    photos = [_photo_from_row(row) for row in rows or []]
    return _group_photos(photos)


def _group_photos(photos: list[DailySharePhoto]) -> list[DailySharePhotoGroup]:
    buckets: OrderedDict[str, dict[str, Any]] = OrderedDict(
        (item["id"], {**item, "photos": []}) for item in [*_GROUP_DEFS, _FALLBACK_GROUP]
    )
    for photo in photos:
        group_id = _classify_photo(photo)
        buckets[group_id]["photos"].append(photo)

    groups = []
    for bucket in buckets.values():
        group_photos = bucket["photos"]
        if not group_photos:
            continue
        groups.append(
            DailySharePhotoGroup(
                id=bucket["id"],
                title=bucket["title"],
                subtitle=bucket["subtitle"],
                count=len(group_photos),
                photos=group_photos,
            )
        )
    return groups


def _classify_photo(photo: DailySharePhoto) -> str:
    text = _normalise_classification_text(photo)
    scored = [
        (_score_group(text, group["signals"]), group["id"]) for group in _GROUP_DEFS
    ]
    best_score, best_id = max(scored, key=lambda item: item[0])
    if best_score > 0:
        return best_id
    return _FALLBACK_GROUP["id"]


def _normalise_classification_text(photo: DailySharePhoto) -> str:
    return f"{photo.name or ''} {photo.vision_summary or ''}".lower()


def _score_group(text: str, signals: tuple[tuple[str, int], ...]) -> int:
    score = 0
    for signal, weight in signals:
        if signal.lower() in text:
            score += weight
    return score


def _photo_from_row(row: Any) -> DailySharePhoto:
    created_at = _value(row, "created_at", "createdAt")
    if isinstance(created_at, datetime):
        created = created_at.isoformat()
    elif created_at is None:
        created = None
    else:
        created = str(created_at)
    return DailySharePhoto(
        id=str(_value(row, "id") or ""),
        message_id=str(_value(row, "message_id", "messageId") or ""),
        conversation_id=str(_value(row, "conversation_id", "conversationId") or ""),
        name=_value(row, "name"),
        mime=str(_value(row, "mime") or "image/jpeg"),
        size=int(_value(row, "size") or 0),
        width=_int_or_none(_value(row, "width")),
        height=_int_or_none(_value(row, "height")),
        url=str(_value(row, "url") or ""),
        vision_summary=_value(row, "vision_summary", "visionSummary"),
        created_at=created,
    )


def _value(row: Any, snake: str, camel: str | None = None) -> Any:
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


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
