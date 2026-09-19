"""Name library seed, gender normalize, and random pick."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.name_templates import (
    ensure_default_names,
    load_default_names,
    normalize_name_gender,
    pick_random_names,
)


def test_bundled_names_cover_both_genders_without_key_dups():
    rows = load_default_names()
    assert len(rows) >= 1300
    keys = [(row["gender"], row["name"]) for row in rows]
    assert len(keys) == len(set(keys))
    genders = {row["gender"] for row in rows}
    assert genders == {"male", "female"}
    assert all(row["name"].strip() for row in rows)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("male", "male"),
        ("female", "female"),
        ("男", "male"),
        ("女", "female"),
        ("MALE", "male"),
        ("", None),
        ("other", None),
    ],
)
def test_normalize_name_gender(raw, expected):
    assert normalize_name_gender(raw) == expected


@pytest.mark.asyncio
async def test_pick_random_names_filters_gender_and_excludes():
    rows = [
        SimpleNamespace(
            id="1", name="陈砚", nickname="阿砚", gender="male",
            status="active", sortOrder=1, createdAt="t", updatedAt="t",
        ),
        SimpleNamespace(
            id="2", name="吴时远", nickname="阿远", gender="male",
            status="active", sortOrder=2, createdAt="t", updatedAt="t",
        ),
        SimpleNamespace(
            id="3", name="赵叙", nickname="阿叙", gender="male",
            status="active", sortOrder=3, createdAt="t", updatedAt="t",
        ),
    ]
    with patch("app.services.name_templates.db") as mock_db:
        mock_db.nametemplate.find_many = AsyncMock(return_value=rows[1:])
        picked = await pick_random_names("男", 2, exclude={"陈砚"})

    assert {row["name"] for row in picked} == {"吴时远", "赵叙"}
    where = mock_db.nametemplate.find_many.await_args.kwargs["where"]
    assert where["gender"] == "male"
    assert where["status"] == "active"
    assert where["name"]["not_in"] == ["陈砚"]


@pytest.mark.asyncio
async def test_pick_random_names_raises_when_pool_too_small():
    with patch("app.services.name_templates.db") as mock_db:
        mock_db.nametemplate.find_many = AsyncMock(return_value=[])
        with pytest.raises(ValueError, match="可用名字不足"):
            await pick_random_names("female", 3)


@pytest.mark.asyncio
async def test_ensure_default_names_inserts_only_missing_keys():
    defaults = load_default_names()
    first = defaults[0]
    existing_row = SimpleNamespace(
        gender=first["gender"],
        name=first["name"],
        sortOrder=7,
    )
    created: list[list] = []

    async def _create_many(*, data):
        created.append(data)
        return len(data)

    with patch("app.services.name_templates.db") as mock_db:
        mock_db.nametemplate.find_many = AsyncMock(return_value=[existing_row])
        mock_db.nametemplate.create_many = AsyncMock(side_effect=_create_many)
        await ensure_default_names()

    inserted = [row for chunk in created for row in chunk]
    assert len(inserted) == len(defaults) - 1
    assert all("id" in row and row["updatedAt"] for row in inserted)
    assert all(
        (row["gender"], row["name"]) != (first["gender"], first["name"])
        for row in inserted
    )
