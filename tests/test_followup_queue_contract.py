"""Family selection must not turn admin inputs into SQL identifiers."""
from unittest.mock import AsyncMock

import pytest

from app.services.runtime.followup_queue import (
    FollowupFamily, MESSAGE, TURN, dispatch_followups, inspect_followups,
    prune_followups, retry_followup,
)


@pytest.mark.parametrize("operation", ["dispatch", "inspect", "prune", "retry"])
async def test_unknown_family_is_rejected_before_database_access(operation):
    database = AsyncMock()
    invalid = FollowupFamily("messages; DROP TABLE users", "message_id", "unknown", "unknown")
    with pytest.raises(ValueError, match="invalid_followup_family"):
        if operation == "dispatch":
            await dispatch_followups(database, invalid, enqueue=AsyncMock(), inspect=AsyncMock())
        elif operation == "inspect":
            await inspect_followups(database, invalid)
        elif operation == "prune":
            await prune_followups(database, invalid)
        else:
            await retry_followup(database, invalid, "id", inspect=AsyncMock())
    assert not database.mock_calls


async def test_turn_projection_is_bounded_and_keeps_root_message_identity():
    database = AsyncMock()
    database.query_raw.return_value = [{"turn_id": f"turn-{i}", "message_id": "root"} for i in range(51)]
    result = await inspect_followups(database, TURN, limit=500, status="failed")
    assert result["limit"] == 50 and result["has_more"]
    assert len(result["items"]) == 50
    sql, limit, status = database.query_raw.call_args.args
    assert "turn_id,message_id,status" in sql and "turn_followup_events" in sql
    assert limit == 51 and status == "failed"
    assert result["denominator"] == "recent_sample_only"


async def test_message_projection_keeps_its_existing_contract():
    database = AsyncMock()
    database.query_raw.return_value = []
    result = await inspect_followups(database, MESSAGE)
    sql, limit = database.query_raw.call_args.args
    assert "SELECT message_id,status" in sql and "turn_id" not in sql
    assert limit == 21 and result["items"] == []
    assert result["retention_days"] == 30 and result["status_filter"] == "all"
