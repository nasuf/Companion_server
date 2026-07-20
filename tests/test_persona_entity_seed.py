"""Phase 2-6: persona entity seeding at provisioning + clone entity copy."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.life_story import _extract_persona_entities, _seed_persona_entities


class TestPersonaEntityExtraction:
    def test_pet_name(self):
        ents = _extract_persona_entities("一只田园猫，橘色，名叫“大橘”。", "宠物")
        assert ents == [{"name": "大橘", "type": "pet", "role": "pet"}]

    def test_family_names_with_roles(self):
        ents = _extract_persona_entities(
            "父亲林国安是街道办事处的基层工作人员，母亲陈秀芬年轻时是茶厂质检员。",
            "亲属关系",
        )
        names = {e["name"]: e["role"] for e in ents}
        assert names == {"林国安": "father", "陈秀芬": "mother"}

    def test_friend_name(self):
        ents = _extract_persona_entities(
            "最要好的闺蜜是高中同学张雅婷，性格活泼外向。", "社会关系",
        )
        assert ents == [{"name": "张雅婷", "type": "person", "role": "friend"}]

    def test_non_relation_subcategory_returns_empty(self):
        assert _extract_persona_entities("我喜欢雾霾蓝", "审美爱好") == []

    def test_no_false_positives_on_plain_text(self):
        assert _extract_persona_entities("家庭氛围总体和睦，父母关系稳定。", "亲属关系") == []


@pytest.mark.asyncio
async def test_seed_persona_entities_links_only_relation_rows():
    rows = [
        {"id": "m1", "summary": "一只田园猫，名叫大橘。", "subCategory": "宠物"},
        {"id": "m2", "summary": "我喜欢雾霾蓝", "subCategory": "审美爱好"},
    ]
    record = AsyncMock(return_value=1)
    with patch(
        "app.services.memory.storage.entity_repo.record_entities_for_memory", record,
    ):
        linked = await _seed_persona_entities(rows, "u1", "ws1")

    assert linked == 1
    record.assert_awaited_once()
    kwargs = record.await_args.kwargs
    assert kwargs["memory_id"] == "m1"
    assert kwargs["memory_source"] == "ai"
    assert kwargs["entities"][0]["name"] == "大橘"


@pytest.mark.asyncio
async def test_clone_copies_entity_links_with_mapped_ids():
    from app.services.agent_template import clone as clone_mod

    rows = [
        {"memory_id": "tpl-m1", "canonical_name": "大橘", "entity_type": "pet",
         "role": "pet", "aliases": []},
        {"memory_id": "tpl-m9", "canonical_name": "张雅婷", "entity_type": "person",
         "role": "friend", "aliases": []},
    ]
    record = AsyncMock(return_value=1)
    with patch.object(clone_mod.db, "query_raw", AsyncMock(return_value=rows)), \
         patch(
             "app.services.memory.storage.entity_repo.record_entities_for_memory",
             record,
         ):
        linked = await clone_mod._clone_memory_entities(
            template_workspace_id="tpl-ws",
            id_pairs=[("tpl-m1", "new-m1"), ("tpl-m2", "new-m2")],
            user_id="u1",
            new_workspace_id="new-ws",
        )

    # Only tpl-m1 is in id_pairs; tpl-m9's edge has no cloned memory.
    assert linked == 1
    kwargs = record.await_args.kwargs
    assert kwargs["memory_id"] == "new-m1"
    assert kwargs["workspace_id"] == "new-ws"
    assert kwargs["entities"][0]["name"] == "大橘"
