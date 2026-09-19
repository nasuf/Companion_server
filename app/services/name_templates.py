"""Agent name library: seed from the PM name workbook + random-by-gender pick.

The workbook (姓名库2.xlsx) is classified by gender (male / female). Duplicate
(gender, name) rows were dropped at import time so the unique constraint holds.
Admin can still add / edit / delete rows after seed; seeding only fills missing
keys and never overwrites operator edits.
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.db import db

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "name_templates.json"
_SEED_CHUNK = 200


def load_default_names() -> list[dict[str, str]]:
    """Load the bundled name workbook dump (gender/name/nickname)."""
    raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("name_templates.json must be a list")
    return [row for row in raw if isinstance(row, dict)]


def normalize_name_gender(value: str | None) -> str | None:
    text = (value or "").strip().lower()
    if not text:
        return None
    if text in {"female", "male"}:
        return text
    if "女" in text:
        return "female"
    if "男" in text:
        return "male"
    return None


def name_row_to_dict(row) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "nickname": row.nickname or "",
        "gender": row.gender,
        "status": row.status,
        "sort_order": row.sortOrder,
        "created_at": str(row.createdAt),
        "updated_at": str(row.updatedAt),
    }


async def pick_random_names(
    gender: str,
    count: int,
    *,
    exclude: set[str] | None = None,
) -> list[dict]:
    """Sample ``count`` active names of ``gender``, skipping ``exclude``.

    Raises ValueError when the remaining pool is smaller than ``count``.
    """
    normalized = normalize_name_gender(gender)
    if normalized not in {"male", "female"}:
        raise ValueError("gender 只能是 male/female/男/女")
    if count < 1:
        raise ValueError("count must be >= 1")

    where: dict = {"gender": normalized, "status": "active"}
    skipped = {name.strip() for name in (exclude or set()) if name and name.strip()}
    if skipped:
        where["name"] = {"not_in": list(skipped)}

    rows = await db.nametemplate.find_many(where=where)
    if len(rows) < count:
        excluded_note = "，已排除现有模板名" if skipped else ""
        raise ValueError(
            f"该性别姓名库可用名字不足 {count} 个（当前 {len(rows)} 个{excluded_note}）"
        )
    chosen = random.sample(rows, count)
    return [name_row_to_dict(row) for row in chosen]


async def ensure_default_names() -> None:
    """Insert bundled names that are not already in the table.

    Unique on (gender, name) so re-seeding is idempotent. Existing rows keep
    whatever nickname / sort_order an admin may have edited.
    """
    defaults = load_default_names()
    existing = await db.nametemplate.find_many()
    existing_keys = {(row.gender, row.name) for row in existing}

    per_gender_max: dict[str, int] = {}
    for row in existing:
        per_gender_max[row.gender] = max(per_gender_max.get(row.gender, 0), row.sortOrder)

    missing: list[dict] = []
    now = datetime.now(UTC)
    for item in defaults:
        gender = normalize_name_gender(item.get("gender"))
        name = (item.get("name") or "").strip()
        if gender not in {"male", "female"} or not name:
            continue
        if (gender, name) in existing_keys:
            continue
        per_gender_max[gender] = per_gender_max.get(gender, 0) + 1
        missing.append(
            {
                "id": str(uuid.uuid4()),
                "name": name,
                "nickname": (item.get("nickname") or "").strip(),
                "gender": gender,
                "sortOrder": per_gender_max[gender],
                "status": "active",
                "updatedAt": now,
            }
        )
        existing_keys.add((gender, name))

    for start in range(0, len(missing), _SEED_CHUNK):
        chunk = missing[start : start + _SEED_CHUNK]
        await db.nametemplate.create_many(data=chunk)
    if missing:
        logger.info("Seeded %d missing name templates", len(missing))
