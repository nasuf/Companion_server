from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.achievements.definitions import ACHIEVEMENTS
from app.services.achievements import repository
from app.services.achievements.repository import unlock_achievement
from app.services.achievements.rule_registry import (
    ACHIEVEMENT_RULES,
    DISABLED_ACHIEVEMENT_IDS,
    validate_rule_registry,
)


_WORKBOOK_CATALOG_SHA256 = (
    "06c815df2dac9a88956d086e571e43b794ab0f38aee51b6ef46551f93aca6284"
)
_BEHAVIOR_TESTED_ACTIVE_IDS = {
    1, 2, 5, 6, 7, 8, 9, 18, 19, 20, 21, 23, 25, 26, 27, 28, 29, 30,
    31, 33, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
}
def test_catalog_matches_verified_workbook_snapshot():
    payload = [definition.to_dict() for definition in ACHIEVEMENTS]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert len(payload) == 97
    assert [definition.id for definition in ACHIEVEMENTS] == list(range(1, 98))
    assert hashlib.sha256(encoded).hexdigest() == _WORKBOOK_CATALOG_SHA256


def test_rule_registry_covers_all_active_and_disabled_achievements():
    validate_rule_registry()

    active = {rule.id for rule in ACHIEVEMENT_RULES.values() if rule.enabled}
    disabled = {rule.id for rule in ACHIEVEMENT_RULES.values() if not rule.enabled}

    assert len(active) == 82
    assert len(disabled) == 15
    assert disabled == set(DISABLED_ACHIEVEMENT_IDS)
    assert active | disabled == set(range(1, 98))
    assert all(ACHIEVEMENT_RULES[item].evaluator != "disabled" for item in active)
    assert all(ACHIEVEMENT_RULES[item].evaluator == "disabled" for item in disabled)
    assert active == _BEHAVIOR_TESTED_ACTIVE_IDS


def test_testcase_document_covers_every_achievement_once():
    document = (
        Path(__file__).parents[1] / "docs" / "achievement_testcase_matrix.md"
    ).read_text()
    matches = [
        (int(match.group(1)), match.group(2))
        for match in re.finditer(r"(?m)^(\d+)\.\s+\*\*(.+)$", document)
    ]
    counts = Counter(achievement_id for achievement_id, _ in matches)
    rows = dict(matches)

    assert len(matches) == 97
    assert set(rows) == set(range(1, 98))
    assert set(counts.values()) == {1}
    for achievement_id, text in rows.items():
        if achievement_id in DISABLED_ACHIEVEMENT_IDS:
            assert "停用" in text
            assert "不得" in text
        else:
            assert "启用" in text
            assert "正例" in text
            assert "反例" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("achievement_id", sorted(DISABLED_ACHIEVEMENT_IDS))
async def test_disabled_achievement_can_never_be_unlocked(achievement_id: int):
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock()

    with patch("app.services.achievements.repository.db", fake_db):
        unlocked = await unlock_achievement(
            user_id="u1",
            agent_id="a1",
            workspace_id="w1",
            conversation_id="c1",
            achievement_id=achievement_id,
        )

    assert unlocked is False
    fake_db.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_marks_disabled_rules_and_excludes_historical_disabled_unlocks():
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[
        {"achievement_id": 1, "unlocked_at": "2026-07-11T00:00:00Z"},
        {"achievement_id": 3, "unlocked_at": "2026-07-11T00:00:00Z"},
    ])
    with (
        patch.object(repository, "db", fake_db),
        patch.object(repository, "_cache_unlocked_achievements", AsyncMock()),
    ):
        result = await repository.list_achievements("u1", "a1")

    item_by_id = {
        item["achievement_id"]: item
        for item in result["items"]
    }
    assert result["total"] == 97
    assert result["active_total"] == 82
    assert result["disabled_total"] == 15
    assert result["unlocked"] == 1
    assert item_by_id[1]["enabled"] is True
    assert item_by_id[1]["unlocked"] is True
    assert item_by_id[3]["enabled"] is False
    assert item_by_id[3]["unlocked"] is False
