from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


def _admin_override():
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    return app, require_admin_jwt


def test_monitoring_aggregates_all_dimensions(api_client):
    """/monitoring 聚合注册/在线/分时段/句子区间/费用榜单 + 概览."""
    app, require_admin_jwt = _admin_override()

    # asyncio.gather 顺序: reg / daily_active / hourly / buckets / cost / totals / activity
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(
        side_effect=[
            [  # registrations daily
                {"bucket": "2026-07-18", "count": 3},
                {"bucket": "2026-07-19", "count": 5},
            ],
            [  # daily active
                {"bucket": "2026-07-18", "active_users": 2, "user_messages": 40},
                {"bucket": "2026-07-19", "active_users": 4, "user_messages": 60},
            ],
            [  # hourly active (only a few hours present)
                {"hour": 9, "users": 3, "user_messages": 30},
                {"hour": 21, "users": 5, "user_messages": 70},
            ],
            [  # message buckets
                {
                    "b1": 10, "b2": 4, "b3": 2, "b4": 1, "b5": 0, "b6": 1,
                    "active_users": 18,
                }
            ],
            [  # cost top users
                {"user_id": "u1", "username": "alice", "cost_cny": 1.2345678,
                 "request_count": 40, "total_tokens": 5000},
                {"user_id": "u2", "username": "bob", "cost_cny": 0.5,
                 "request_count": 20, "total_tokens": 2000},
            ],
            [  # totals
                {"total_users": 100, "total_conversations": 250, "total_agents": 120}
            ],
            [  # activity dau/wau/mau/5min
                {"dau": 12, "wau": 40, "mau": 80, "active_5min": 3}
            ],
        ],
    )

    try:
        with (
            patch("app.api.admin.stats.db", fake_db),
            patch(
                "app.api.admin.stats.count_online_users",
                new_callable=AsyncMock,
                return_value=(7, True),
            ),
        ):
            response = api_client.get("/admin-api/stats/monitoring", params={"days": 7})

        assert response.status_code == 200
        data = response.json()

        # online
        assert data["online_now"]["count"] == 7
        assert data["online_now"]["active_5min"] == 3
        assert data["online_now"]["redis_available"] is True

        # overview
        assert data["overview"]["total_users"] == 100
        assert data["overview"]["total_conversations"] == 250
        assert data["overview"]["dau"] == 12
        assert data["overview"]["wau"] == 40
        assert data["overview"]["mau"] == 80
        assert data["overview"]["new_users_window"] == 8  # 3 + 5
        assert data["overview"]["active_users_window"] == 18
        assert data["overview"]["user_messages_window"] == 100  # 40 + 60

        # registrations
        assert data["registrations"]["total"] == 8
        assert len(data["registrations"]["daily"]) == 2

        # hourly padded to 24
        assert len(data["hourly_active"]) == 24
        assert data["hourly_active"][9]["users"] == 3
        assert data["hourly_active"][21]["users"] == 5
        assert data["hourly_active"][0]["users"] == 0

        # message buckets: leading "0 条" bucket + 6 range buckets
        buckets = data["message_buckets"]["buckets"]
        assert len(buckets) == 7
        assert buckets[0]["min"] == 0
        assert buckets[0]["users"] == 82  # 100 total - 18 active
        assert buckets[1]["users"] == 10
        assert buckets[-1]["users"] == 1

        # cost top users
        assert data["cost_top_users"][0]["username"] == "alice"
        assert data["cost_top_users"][0]["cost_cny"] == 1.234568
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_monitoring_survives_redis_failure(api_client):
    """Redis 挂掉时在线人数降级为 0 + redis_available=False, 其余照常."""
    app, require_admin_jwt = _admin_override()
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(
        side_effect=[
            [], [], [], [{"b1": 0, "b2": 0, "b3": 0, "b4": 0, "b5": 0, "b6": 0, "active_users": 0}],
            [], [{"total_users": 5, "total_conversations": 0, "total_agents": 3}],
            [{"dau": 0, "wau": 0, "mau": 0, "active_5min": 0}],
        ],
    )

    try:
        with (
            patch("app.api.admin.stats.db", fake_db),
            patch(
                "app.api.admin.stats.count_online_users",
                new_callable=AsyncMock,
                return_value=(0, False),
            ),
        ):
            response = api_client.get("/admin-api/stats/monitoring")

        assert response.status_code == 200
        data = response.json()
        assert data["online_now"]["count"] == 0
        assert data["online_now"]["redis_available"] is False
        # zero bucket = all registered users (no active users)
        assert data["message_buckets"]["buckets"][0]["users"] == 5
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_monitoring_all_time_omits_window_filter(api_client):
    """days=0 (全部) 时不注入时间过滤参数 (WHERE 用 1=1)."""
    app, require_admin_jwt = _admin_override()
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(
        side_effect=[
            [], [], [], [{"b1": 0, "b2": 0, "b3": 0, "b4": 0, "b5": 0, "b6": 0, "active_users": 0}],
            [], [{"total_users": 0, "total_conversations": 0, "total_agents": 0}],
            [{"dau": 0, "wau": 0, "mau": 0, "active_5min": 0}],
        ],
    )

    try:
        with (
            patch("app.api.admin.stats.db", fake_db),
            patch(
                "app.api.admin.stats.count_online_users",
                new_callable=AsyncMock,
                return_value=(0, True),
            ),
        ):
            response = api_client.get("/admin-api/stats/monitoring", params={"days": 0})

        assert response.status_code == 200
        # registrations query is the first gather call; with all-time there are no
        # positional params bound.
        reg_call = fake_db.query_raw.await_args_list[0]
        assert "1=1" in reg_call.args[0]
        assert len(reg_call.args) == 1  # sql only, no window param
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)
