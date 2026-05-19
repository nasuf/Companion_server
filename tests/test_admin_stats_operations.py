from unittest.mock import AsyncMock, MagicMock, patch


def _admin_override():
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    return app, require_admin_jwt


def test_admin_operations_stats_aggregates_db_and_redis(api_client):
    from app.services.proactive import triggers as proactive_triggers
    from app.services.runtime import job_queue as runtime_job_queue

    app, require_admin_jwt = _admin_override()

    class FakeRedis:
        async def zcard(self, key):
            return {
                proactive_triggers._DLQ_KEY: 2,
                runtime_job_queue._DELAYED_KEY: 3,
                runtime_job_queue._RUNNING_KEY: 4,
            }.get(key, 0)

        async def llen(self, key):
            return {
                runtime_job_queue._DLQ_KEY: 5,
                runtime_job_queue._READY_KEY: 6,
            }.get(key, 0)

    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(
        side_effect=[
            [
                {"operation": "insert", "count": 7},
                {"operation": "access", "count": 11},
                {"operation": "evidence_linked", "count": 4},
                {"operation": "contradiction_archived", "count": 1},
                {"operation": "user_bulk_delete", "count": 2},
            ],
            [{
                "request_count": 9,
                "call_count": 14,
                "input_tokens": 1000,
                "output_tokens": 250,
                "cost_cny": 0.1234567,
                "latency_ms_total": 2800,
                "latency_count": 4,
                "failure_count": 2,
                "fallback_count": 1,
                "circuit_open_count": 1,
            }],
            [{"scope": "chat", "request_count": 8}],
            [
                {"event_type": "message_sent", "count": 3},
                {"event_type": "send_skipped", "count": 1},
                {"event_type": "window_deferred", "count": 2},
            ],
            [{"trigger_type": "memory_proactive", "count": 3}],
            [{"status": "waiting_user", "count": 2}],
            [{
                "total_count": 10,
                "active_count": 8,
                "overdue_active_count": 1,
                "due_next_24h_count": 4,
            }],
            [{"fired_count": 6}],
            [
                {"status": "open", "count": 2},
                {"status": "resolved", "count": 1},
            ],
            [
                {"error_type": "memory_mixup", "reason": "记忆编造", "count": 2},
                {"error_type": "memory_mixup", "reason": "记忆幻觉", "count": 1},
                {"error_type": "tone_robot", "reason": "像机器人", "count": 1},
            ],
            [{
                "trace_id": "trace-1",
                "message_id": "msg-1",
                "conversation_id": "conv-1",
                "agent_id": "agent-1",
                "user_id": "user-1",
                "root_message": "我今天状态很差",
                "total_duration_ms": 32000,
                "llm_step_count": 9,
                "share_status": "shared",
                "open_bug_count": 1,
                "created_at": "2026-05-19T10:00:00",
            }],
        ],
    )

    try:
        with (
            patch("app.api.admin.stats.db", fake_db),
            patch("app.api.admin.stats.get_redis", new_callable=AsyncMock, return_value=FakeRedis()),
        ):
            response = api_client.get(
                "/admin-api/stats/operations",
                params={"days": 7, "agent_id": "agent-1", "user_id": "user-1"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["memory"]["stored_count"] == 7
        assert data["memory"]["retrieval_access_count"] == 11
        assert data["memory"]["evidence_link_count"] == 4
        assert data["memory"]["correction_count"] == 1
        assert data["memory"]["deletion_count"] == 2
        assert data["llm"]["cost_cny"] == 0.123457
        assert data["llm"]["avg_latency_ms"] == 700
        assert data["llm"]["failure_count"] == 2
        assert data["llm"]["fallback_count"] == 1
        assert data["llm"]["fallback_rate"] == 0.25
        assert data["llm"]["circuit_open_count"] == 1
        assert data["proactive"]["sent_count"] == 3
        assert data["proactive"]["skipped_count"] == 3
        assert data["proactive"]["waiting_user_count"] == 2
        assert data["reminders"]["dlq_count"] == 2
        assert data["runtime_jobs"]["ready_count"] == 6
        assert data["runtime_jobs"]["delayed_count"] == 3
        assert data["runtime_jobs"]["running_count"] == 4
        assert data["runtime_jobs"]["dlq_count"] == 5
        assert data["bug_reports"]["created_count"] == 3
        assert data["bug_reports"]["by_error_type"]["memory_mixup"] == 3
        assert data["bug_reports"]["by_eval_category"]["memory_safety"] == 3
        assert data["bug_reports"]["by_eval_category"]["tone"] == 1
        assert data["high_risk_traces"]["window_hours"] == 24
        assert data["high_risk_traces"]["count"] == 1
        trace = data["high_risk_traces"]["items"][0]
        assert trace["trace_id"] == "trace-1"
        assert trace["risk_score"] == 8
        assert trace["risk_reasons"] == [
            "slow_trace_30s",
            "many_llm_steps_8",
            "open_bug_report",
        ]
        assert data["data_quality"]["redis_available"] is True
        assert data["data_quality"]["llm_latency_available"] is True
        assert data["data_quality"]["llm_fallback_available"] is True

        reminder_call = fake_db.query_raw.await_args_list[6]
        reminder_sql = reminder_call.args[0]
        assert "t.ai_agent_id = $1" in reminder_sql
        assert "t.user_id = $2" in reminder_sql
        assert "t.trigger_time < $3::timestamp" in reminder_sql
        assert reminder_call.args[1:3] == ("agent-1", "user-1")

        high_risk_call = fake_db.query_raw.await_args_list[10]
        high_risk_sql = high_risk_call.args[0]
        assert "mt.created_at >= $1::timestamp" in high_risk_sql
        assert "c.agent_id = $2" in high_risk_sql
        assert "c.user_id = $3" in high_risk_sql
        assert high_risk_call.args[2:4] == ("agent-1", "user-1")
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_admin_operations_stats_survives_redis_failure(api_client):
    app, require_admin_jwt = _admin_override()
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(
        side_effect=[
            [],
            [{
                "request_count": 0,
                "call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_cny": 0.0,
                "latency_ms_total": 0,
                "latency_count": 0,
                "failure_count": 0,
                "fallback_count": 0,
                "circuit_open_count": 0,
            }],
            [],
            [],
            [],
            [],
            [{"total_count": 0, "active_count": 0, "overdue_active_count": 0, "due_next_24h_count": 0}],
            [{"fired_count": 0}],
            [],
            [],
            [],
        ],
    )

    try:
        with (
            patch("app.api.admin.stats.db", fake_db),
            patch("app.api.admin.stats.get_redis", new_callable=AsyncMock, side_effect=RuntimeError("redis down")),
        ):
            response = api_client.get("/admin-api/stats/operations")

        assert response.status_code == 200
        data = response.json()
        assert data["data_quality"]["redis_available"] is False
        assert data["reminders"]["dlq_count"] == 0
        assert data["runtime_jobs"]["dlq_count"] == 0
        assert data["high_risk_traces"]["items"] == []
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)
