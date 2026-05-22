from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_record_reply_operational_metrics_persists_visible_and_crisis(monkeypatch):
    from app.services.operations import metrics

    fake_db = SimpleNamespace(
        conversation=SimpleNamespace(
            find_unique=AsyncMock(return_value=SimpleNamespace(
                id="conv-1",
                agentId="agent-1",
                userId="u1",
                workspaceId="ws1",
            )),
        ),
        query_raw=AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(metrics, "db", fake_db)

    await metrics.record_reply_operational_metrics(
        message_id="msg-1",
        conversation_id="conv-1",
        metadata={
            "trace_id": "trace-1",
            "memory_retrieval_analysis": {
                "method": "lexical_overlap_v1",
                "selected_count": 2,
                "likely_used_count": 1,
                "likely_unused_count": 1,
                "quality_metrics": {
                    "visible_use_rate": 0.5,
                    "warning_count": 1,
                    "has_prompt_dilution": True,
                    "has_final_gate_drop": False,
                },
                "warnings": [{"code": "no_visible_memory_use"}],
                "items": [{"id": "mem1"}, {"id": "mem2"}],
            },
            "response_diagnostics": {
                "crisis_guard_status": "direct_crisis",
                "crisis_semantic_checked": False,
                "crisis_semantic_detected": False,
            },
        },
    )

    assert fake_db.query_raw.await_count == 2
    visible_args = fake_db.query_raw.await_args_list[0].args
    crisis_args = fake_db.query_raw.await_args_list[1].args
    assert "memory_visible_use_events" in visible_args[0]
    assert visible_args[9] == 2
    assert visible_args[10] == 1
    assert "crisis_events" in crisis_args[0]
    assert crisis_args[8] == "direct_crisis"
