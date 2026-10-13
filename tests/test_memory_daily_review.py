"""Trusted profile category policy is independent of daily generated summaries."""

import pytest
from app.services.memory.taxonomy import resolve_taxonomy


@pytest.mark.parametrize("level", [2, 3])
@pytest.mark.parametrize(
    "category,sub", [("身份", "职业"), ("偏好", "兴趣"), ("思维", "价值观")]
)
@pytest.mark.parametrize("provenance", ["profile_seed", "knowledge_seed"])
def test_trusted_seed_taxonomy_survives_cooling(level, category, sub, provenance):
    result = resolve_taxonomy(
        main_category=category,
        sub_category=sub,
        source="ai",
        level=level,
        provenance=provenance,
    )
    assert result.allowed and result.main_category == category
    denied = resolve_taxonomy(
        main_category=category,
        sub_category=sub,
        source="ai",
        level=level,
        provenance="ai_authored",
    )
    assert not denied.allowed
