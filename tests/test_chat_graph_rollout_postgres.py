"""G03 SQL on disposable PostgreSQL; never use application DB URLs."""

from datetime import datetime, timedelta, timezone
import json
import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from prisma import Prisma
import pytest

from app.services.ops.chat_graph_rollout import Observation, collect

NOW = datetime(2026, 10, 4, 3, tzinfo=timezone.utc)
RUNTIME = dict(executor="langgraph", allowlist="c-1", graph_version="chat-g01-v1",
               durable_execution_ready=False, trace_backend="local")
REQUEST = Observation(("c-1",), "langgraph", "chat-g01-v1",
                      NOW - timedelta(hours=26), NOW - timedelta(hours=1))


@pytest.fixture
async def postgres():
    url = os.getenv("G03_TEST_DATABASE_URL")
    if not url:
        pytest.skip("G03_TEST_DATABASE_URL must target disposable local PostgreSQL")
    parsed = urlsplit(url)
    assert parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    params = dict(parse_qsl(parsed.query))
    params["connection_limit"] = "1"  # Temporary tables stay on one connection.
    database = Prisma(datasource={"url": urlunsplit(parsed._replace(query=urlencode(params)))},
                      http={"trust_env": False})
    await database.connect()
    try:
        # Connection-local tables mirror the existing Prisma projection types.
        for sql in (
            """CREATE TEMP TABLE trace_runs (
                run_id TEXT PRIMARY KEY, trace_id TEXT, parent_id TEXT, name TEXT,
                run_type TEXT, status TEXT, started_at TIMESTAMP(3), ended_at TIMESTAMP(3),
                inputs_json JSONB, extra_json JSONB, created_at TIMESTAMP(3))""",
            """CREATE TEMP TABLE llm_usage (
                trace_id TEXT, conversation_id TEXT, scope TEXT, cost_cny DOUBLE PRECISION,
                call_count INTEGER, failure_count INTEGER, fallback_count INTEGER)""",
            """CREATE TEMP TABLE messages (
                conversation_id TEXT, role TEXT, content TEXT, metadata JSONB,
                created_at TIMESTAMP(3))""",
        ):
            await database.execute_raw(sql)
        for i in range(20):
            trace, stamp = f"trace-{i}", (REQUEST.since + timedelta(minutes=i)).isoformat()
            meta = {"metadata": dict(executor="langgraph", graph_version="chat-g01-v1",
                                    checkpoint_enabled=False, usage_expected=True)}
            await database.execute_raw(
                """INSERT INTO trace_runs VALUES ($1,$1,NULL,'chat_request','chain','success',
                ($2::timestamptz AT TIME ZONE 'UTC'),
                ($2::timestamptz AT TIME ZONE 'UTC')+interval '1 second',
                $3::jsonb,$4::jsonb,($2::timestamptz AT TIME ZONE 'UTC'))""",
                trace, stamp, json.dumps({"conversation_id": "c-1", "message": "PRIVATE"}),
                json.dumps(meta),
            )
            for name in ("main_chat", "finish_turn"):
                await database.execute_raw(
                    """INSERT INTO trace_runs VALUES ($1,$2,$2,$3,'chain','success',
                    ($4::timestamptz AT TIME ZONE 'UTC'),
                    ($4::timestamptz AT TIME ZONE 'UTC')+interval '1 second',
                    '{}'::jsonb,$5::jsonb,($4::timestamptz AT TIME ZONE 'UTC'))""",
                    f"{trace}-{name}", trace, name, stamp, json.dumps(meta),
                )
            await database.execute_raw(
                "INSERT INTO llm_usage VALUES ($1,'c-1','chat',.01,4,0,0)", trace,
            )
            await database.execute_raw(
                """INSERT INTO messages VALUES ('c-1','assistant','PRIVATE_REPLY',
                   $1::jsonb,($2::timestamptz AT TIME ZONE 'UTC'))""",
                json.dumps({"trace_id": trace}), stamp,
            )
        yield database
    finally:
        await database.disconnect()


async def test_real_sql_scopes_and_extracts_typed_metadata(postgres):
    await postgres.execute_raw(
        "INSERT INTO llm_usage VALUES ('trace-0','unauthorized','chat',999,999,9,9)",
    )
    result = await collect(postgres, REQUEST, RUNTIME, now=NOW)
    assert result["telemetry_ready"]
    assert result["metrics"]["traced_turns"] == 20
    assert result["metrics"]["mean_chat_cost_cny"] == pytest.approx(.01)
    assert result["metrics"]["p95_root_duration_ms"] == 1000
    assert "PRIVATE" not in json.dumps(result)
    assert result["release_ready"] is False


async def test_real_sql_zero_model_and_missing_billing_are_distinct(postgres):
    await postgres.execute_raw("DELETE FROM llm_usage WHERE trace_id='trace-0'")
    assert not (await collect(postgres, REQUEST, RUNTIME, now=NOW))["telemetry_ready"]
    await postgres.execute_raw(
        """UPDATE trace_runs SET extra_json=jsonb_set(extra_json,
           '{metadata,usage_expected}','false'::jsonb) WHERE run_id='trace-0'""",
    )
    result = await collect(postgres, REQUEST, RUNTIME, now=NOW)
    assert result["telemetry_ready"]
    assert result["metrics"]["zero_model_turns"] == 1
    assert result["metrics"]["mean_chat_cost_cny"] == pytest.approx(.0095)


async def test_real_sql_node_version_drift_is_visible(postgres):
    await postgres.execute_raw(
        """UPDATE trace_runs SET extra_json=jsonb_set(
           extra_json,'{metadata,graph_version}','"chat-future"'::jsonb)
           WHERE run_id='trace-0-finish_turn'""",
    )
    result = await collect(postgres, REQUEST, RUNTIME, now=NOW)
    assert not result["telemetry_ready"]
    assert "node_version" in {v["code"] for v in result["issues"]}


async def test_queries_use_read_only_repeatable_read(postgres):
    class CheckedContext:
        def __init__(self, **kwargs):
            self.context = postgres.tx(**kwargs)

        async def __aenter__(self):
            tx = await self.context.__aenter__()

            class CheckedTransaction:
                async def execute_raw(self, *args):
                    return await tx.execute_raw(*args)

                async def query_raw(self, query, *params):
                    state = await tx.query_raw(
                        """SELECT current_setting('transaction_read_only') AS read_only,
                        current_setting('transaction_isolation') AS isolation,
                        current_setting('statement_timeout') AS timeout""",
                    )
                    assert state == [{"read_only": "on", "isolation": "repeatable read",
                                      "timeout": "5s"}]
                    return await tx.query_raw(query, *params)

            return CheckedTransaction()

        async def __aexit__(self, *args):
            return await self.context.__aexit__(*args)

    class CheckedDatabase:
        tx = CheckedContext

    assert (await collect(CheckedDatabase(), REQUEST, RUNTIME, now=NOW))["telemetry_ready"]
