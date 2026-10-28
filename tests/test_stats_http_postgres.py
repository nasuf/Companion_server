"""Real JWT -> current SQL administrator -> dashboard routes -> isolated PG.

Exercises contract failures as well as success; no auth or statistics overrides.
"""

from datetime import timedelta
from unittest.mock import AsyncMock
import httpx
import pytest
from fastapi import FastAPI
from app.api.admin import stats
from app.services.auth import create_jwt
from app.services.effect import daily_facts, definitions, dashboard
from tests.test_beta_daily_facts_postgres import (
    pg,
    facts_refresh_redis,
    _seed,
    _now,
    DAY,
)


async def test_admin_stats_http_contracts_and_scope_boundaries(
    pg, facts_refresh_redis, monkeypatch
):
    ids = await _seed(pg)
    now = _now(DAY + timedelta(days=5))
    await daily_facts.compute_day(DAY, now=now)
    monkeypatch.setattr(stats, "db", pg.db)
    monkeypatch.setattr(stats, "get_redis", AsyncMock(return_value=facts_refresh_redis))
    monkeypatch.setattr(
        "app.redis_client.get_redis", AsyncMock(return_value=facts_refresh_redis)
    )
    monkeypatch.setattr(
        definitions, "local_today", lambda now=None: DAY + timedelta(days=5)
    )
    monkeypatch.setattr(stats, "local_today", definitions.local_today)
    # Healing is separately exercised on real SQL+Redis; don't leave a future
    # clock's refresh job alive past this request fixture's transaction lifetime.
    monkeypatch.setattr(
        "app.services.runtime.tasks.fire_background", lambda coro: coro.close()
    )
    app = FastAPI()
    app.include_router(stats.router)
    args = {"days": 1, "end": str(DAY), "cohort": "all", "q": ids.external}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://isolated"
    ) as client:
        endpoints = [
            "users",
            "retention",
            "review-items",
            "reply-timing",
            "costs",
            "token-reconciliation",
            "proactive-funnel",
            "release-markers",
            "weekly-review",
            "detail-metrics/context",
            "system-health-summary",
            "message-distribution",
        ]
        for path in endpoints:
            for user, role, expected in [
                (None, None, 401),
                (ids.external, "user", 403),
            ]:
                headers = (
                    {}
                    if user is None
                    else {"Authorization": "Bearer " + create_jwt(user, role)}
                )
                response = await client.get(
                    "/admin-api/stats/" + path, params=args, headers=headers
                )
                assert response.status_code == expected, (path, response.status_code)
        client.headers["Authorization"] = "Bearer " + create_jwt(ids.admin, "admin")
        results = {}
        for path in endpoints:
            response = await client.get("/admin-api/stats/" + path, params=args)
            assert response.status_code == 200, (path, response.text[:200])
            results[path] = response.json()
            assert "合成" not in response.text  # fixture message content never leaks
        assert results["users"]["total"] == 1
        assert results["reply-timing"]["coverage"]["turns"] >= 1
        assert results["message-distribution"]["coverage"]["state"] == "ok"
        assert (
            results["weekly-review"]["current"]["summary"]["chat_active_users"]["state"]
            == "not_computed"
        )
        # Role comes from current SQL state, not a forged/stale JWT role.
        client.headers["Authorization"] = "Bearer " + create_jwt(ids.external, "admin")
        assert (
            await client.get("/admin-api/stats/users", params=args)
        ).status_code == 403
        client.headers["Authorization"] = "Bearer " + create_jwt(ids.admin, "admin")
        for path in endpoints:
            if path == "system-health-summary":
                continue
            response = await client.get(
                "/admin-api/stats/" + path, params={**args, "end": "invalid"}
            )
            assert response.status_code == 422, path
        for path, extra in [
            ("review-items", {"kind": "arbitrary"}),
            ("retention", {"platform": "x" * 65}),
            ("detail-metrics/hourly", {"agent": pg.created.agents[0]}),
            ("token-reconciliation", {"cohort": "external"}),
            ("daily-metrics", {"metrics": "token_exact_match", "cohort": "external"}),
            ("daily-metrics", {"dimension": "reply_path", "metrics": "user_messages"}),
        ]:
            assert (
                await client.get("/admin-api/stats/" + path, params={**args, **extra})
            ).status_code == 422, path
        response = await client.get(
            "/admin-api/stats/daily-metrics",
            params={**args, "summary": True, "compare": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["chat_active_users"]["value"] == 3
        assert (
            data["previous"]["summary"]["chat_active_users"]["state"] == "not_computed"
        )
        journey = await client.get(
            f"/admin-api/stats/users/{ids.external}/journey", params=args
        )
        assert (
            journey.status_code == 200
            and journey.json()["items"][0]["user_messages"] == 6
        )
        assert (
            await client.get(
                f"/admin-api/stats/users/{ids.external}/journey?end=invalid"
            )
        ).status_code == 422
        target = await client.get(
            f"/admin-api/stats/users/{ids.external}/chat-target",
            params={"date": str(DAY)},
        )
        assert (
            target.status_code == 200
            and target.json()["conversation_id"] == pg.created.conversations[0]
        )
        assert (
            await client.get(
                f"/admin-api/stats/users/{ids.external}/chat-target?date=invalid"
            )
        ).status_code == 422
        assert (
            await client.get(
                "/admin-api/stats/users/absent/chat-target", params={"date": str(DAY)}
            )
        ).status_code == 404
        markdown = await client.get(
            "/admin-api/stats/weekly-review", params={**args, "format": "markdown"}
        )
        assert (
            markdown.status_code == 200
            and "# 统计周报" in markdown.text
            and markdown.headers["cache-control"] == "no-store"
        )
        assert (
            await client.post("/admin-api/stats/weekly-review/queue", params=args)
        ).status_code == 422  # incomplete week
        monkeypatch.setattr(
            dashboard,
            "reply_timing",
            AsyncMock(side_effect=RuntimeError("private DB error")),
        )
        failed = await client.get("/admin-api/stats/reply-timing", params=args)
        assert failed.status_code == 503 and "private" not in failed.text
