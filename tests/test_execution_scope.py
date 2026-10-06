"""Scope contracts and fail-closed database boundaries (no live connections)."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.services.runtime import execution_scope as scope


@pytest.fixture
def bound():
    return scope.ExecutionScope(
        actor_user_id="operator", owner_user_id="owner", agent_id="agent",
        workspace_id="workspace", conversation_id="conversation",
        **{key + "_generation": str(uuid4()) for key in
           ("actor", "owner", "agent", "workspace", "conversation")},
    )


def test_immutable_storage_round_trip(bound):
    assert scope.ExecutionScope.from_record(bound.to_record()) == bound
    with pytest.raises(FrozenInstanceError):
        bound.owner_user_id = "other"
    assert "role" not in bound.to_record()


@pytest.mark.parametrize("mutation", [
    {"schema_version": True}, {"schema_version": 2}, {"role": "admin"},
    {"workspace_id": None}, {"conversation_id": ""}, {"actor_user_id": 42},
    {"actor_generation": "unversioned"}, {"workspace_generation": None},
    {"owner_user_id": "x" * 257},
])
def test_untrusted_or_unknown_record_format_is_rejected(bound, mutation):
    with pytest.raises((ValueError, TypeError)):
        scope.ExecutionScope.from_record({**bound.to_record(), **mutation})


def test_missing_field_and_non_mapping_rejected(bound):
    data = bound.to_record()
    del data["owner_generation"]
    for record in (data, None, [], "scope"):
        with pytest.raises(ValueError):
            scope.ExecutionScope.from_record(record)


@pytest.mark.parametrize("rows", [[], [{}, {}]])
async def test_inaccessible_scope_does_not_fall_back(rows):
    database = SimpleNamespace(query_raw=AsyncMock(return_value=rows))
    with pytest.raises(scope.ExecutionScopeUnavailable) as error:
        await scope.bind_conversation_scope(actor_user_id="actor", conversation_id="conv", database=database)
    assert str(error.value) == "Execution resources are unavailable"


async def test_database_outage_is_not_converted_to_an_authorized_scope():
    database = SimpleNamespace(query_raw=AsyncMock(side_effect=ConnectionError("synthetic")))
    with pytest.raises(ConnectionError):
        await scope.bind_conversation_scope(actor_user_id="actor", conversation_id="conv", database=database)


@pytest.mark.parametrize("field", ["actor_generation", "owner_generation", "agent_generation",
                                  "workspace_generation", "conversation_generation", "workspace_id"])
async def test_resume_rejects_any_changed_identity(bound, field):
    fresh = replace(bound, **{field: str(uuid4())})
    record = fresh.to_record()
    del record["schema_version"]
    database = SimpleNamespace(query_raw=AsyncMock(return_value=[record]))
    with pytest.raises(scope.ExecutionScopeExpired):
        await scope.revalidate_scope(bound, database=database)


async def test_valid_resume_returns_no_new_scope(bound):
    record = bound.to_record()
    del record["schema_version"]
    database = SimpleNamespace(query_raw=AsyncMock(return_value=[record]))
    assert await scope.revalidate_scope(bound, database=database) is None
