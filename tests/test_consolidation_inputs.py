"""Compression input bounds and public metadata semantics."""

from unittest.mock import AsyncMock
import pytest
from app.services.memory.evidence_read import consolidation_details
from app.services.memory.lifecycle import (
    consolidation as cons,
    consolidation_write as write,
)
from app.services.prompting.store import PromptDisabledError


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"side": []},
        {"side": "ai", "parents": True},
        {"side": "ai", "parents": 4},
        {"side": "ai", "parents": 301},
        {"side": "ai", "parents": 5},
        {"side": "ai", "parents": 5, "sampled_at": None},
        {"side": "ai", "parents": 5, "sampled_at": "2026-10-14"},
        {"side": "ai", "parents": 5, "sampled_at": "not-a-time"},
    ],
)
def test_invalid_metadata_is_uncollected_not_zero(value):
    assert consolidation_details(value) is None


def test_public_projection_strips_private_fields():
    result = consolidation_details(
        {
            "side": "user",
            "parents": 300,
            "sampled_at": "2026-10-14T00:00:00Z",
            "prompt": "PRIVATE",
            "parents_text": "PRIVATE",
        }
    )
    assert result == {
        "side": "user",
        "parents": 300,
        "sampled_at": "2026-10-14T00:00:00Z",
    }


@pytest.mark.parametrize(
    "ids",
    [
        [],
        ["x"] * 5,
        ["a", "b", "c", "d", None],
        ["a", "b", "c", "d", ""],
        ["a", "b", "c", "d", "x" * 201],
        list(map(str, range(301))),
    ],
)
async def test_parent_identity_limit_rejects_before_query(ids):
    db = AsyncMock()
    with pytest.raises(ValueError, match="invalid_parents"):
        await write.pin_parents(
            db, source="ai", user_id="u", workspace_id="w", ids=ids, cutoff="now"
        )
    db.query_raw.assert_not_awaited()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": "invalid", "workspace_id": "w"},
        {"source": "ai", "workspace_id": None},
    ],
)
async def test_bad_scope_rejects_before_db(kwargs):
    db = AsyncMock()
    with pytest.raises(ValueError):
        await write.prepare_cluster(db, user_id="u", cluster=[], **kwargs)
    db.tx.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, 11, True, 1.5])
async def test_cluster_limit_is_bounded(limit):
    with pytest.raises(ValueError, match="cluster limit"):
        await cons.compress_l3_clusters_for_workspace(
            user_id="u", workspace_id="w", max_clusters=limit
        )


async def test_disabled_prompt_does_not_generate_or_commit(monkeypatch):
    monkeypatch.setattr(
        cons,
        "prepare_cluster",
        AsyncMock(return_value={"parents": [{"content": "original"}]}),
    )
    monkeypatch.setattr(
        cons,
        "get_prompt_text",
        AsyncMock(side_effect=PromptDisabledError("memory.consolidation")),
    )
    model = AsyncMock()
    monkeypatch.setattr(cons, "invoke_json", model)
    commit = AsyncMock()
    monkeypatch.setattr(cons, "commit_cluster", commit)
    assert (
        await cons._compress_cluster(
            source="ai", user_id="u", workspace_id="w", cluster=[]
        )
        is None
    )
    model.assert_not_awaited()
    commit.assert_not_awaited()


async def test_candidate_query_failure_is_not_reported_as_empty_success(monkeypatch):
    monkeypatch.setattr(
        cons, "_load_candidates", AsyncMock(side_effect=RuntimeError("unavailable"))
    )
    audit = AsyncMock()
    monkeypatch.setattr(cons, "_record_run", audit)
    result = await cons.compress_l3_clusters_for_workspace(
        user_id="u", workspace_id="w"
    )
    assert result["failed"] == 2 and result["digests"] == 0
    audit.assert_awaited_once()


async def test_oversized_parent_text_rejects_before_model():
    rows = [
        {
            "id": str(i),
            "content": "x" * 110000,
            "main_category": "生活",
            "sub_category": "其他",
        }
        for i in range(5)
    ]
    db = AsyncMock()
    db.query_raw.return_value = rows
    with pytest.raises(ValueError, match="input_too_large"):
        await write.pin_parents(
            db,
            source="ai",
            user_id="u",
            workspace_id="w",
            ids=[r["id"] for r in rows],
            cutoff="now",
        )


@pytest.mark.parametrize(
    "side,main,error",
    [("user", "生活", "side_mismatch"), ("ai", "身份", "invalid_taxonomy")],
)
async def test_invalid_commit_scope_or_category_has_no_effects(side, main, error):
    db = AsyncMock()
    inputs = {
        "side": "ai",
        "parents": [{"id": "x", "main_category": main, "sub_category": "姓名"}],
    }
    with pytest.raises(ValueError, match=error):
        await write.commit_cluster(
            db,
            source=side,
            user_id="u",
            workspace_id="w",
            inputs=inputs,
            digest="irrelevant",
            embedding=[],
            prompt="",
            model_name=None,
            output={},
        )
    db.tx.assert_not_called()
