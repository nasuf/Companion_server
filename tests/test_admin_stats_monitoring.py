from unittest.mock import AsyncMock, MagicMock, patch


def _admin_override():
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    return app, require_admin_jwt


class _FakeRedis:
    """In-memory stand-in so monitoring cache read/write is deterministic."""

    def __init__(self, store=None):
        self.store = store if store is not None else {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


def _monitoring_side_effect():
    # asyncio.gather order: reg / daily_active / hourly / buckets / cost / totals / activity
    return [
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
            {"b1": 10, "b2": 4, "b3": 2, "b4": 1, "b5": 0, "b6": 1, "active_users": 18}
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
    ]


def test_monitoring_aggregates_all_dimensions(api_client):
    """/monitoring 聚合注册/在线/分时段/句子区间/费用榜单 + 概览."""
    app, require_admin_jwt = _admin_override()

    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=_monitoring_side_effect())

    try:
        with (
            patch("app.api.admin.stats.db", fake_db),
            patch("app.api.admin.stats.get_redis", new_callable=AsyncMock, return_value=_FakeRedis()),
            patch(
                "app.api.admin.stats.count_online_users",
                new_callable=AsyncMock,
                return_value=(7, True),
            ),
        ):
            response = api_client.get("/admin-api/stats/monitoring", params={"days": 7})

        assert response.status_code == 200
        data = response.json()

        assert data["cached"] is False
        assert data["online_now"]["count"] == 7
        assert data["online_now"]["active_5min"] == 3

        assert data["overview"]["total_users"] == 100
        assert data["overview"]["dau"] == 12
        assert data["overview"]["wau"] == 40
        assert data["overview"]["mau"] == 80
        assert data["overview"]["new_users_window"] == 8
        assert data["overview"]["active_users_window"] == 18
        assert data["overview"]["user_messages_window"] == 100

        assert data["registrations"]["total"] == 8
        assert len(data["hourly_active"]) == 24
        assert data["hourly_active"][9]["users"] == 3
        assert data["hourly_active"][0]["users"] == 0

        buckets = data["message_buckets"]["buckets"]
        assert len(buckets) == 7
        assert buckets[0]["users"] == 82  # 100 total - 18 active
        assert buckets[1]["users"] == 10
        assert buckets[-1]["users"] == 1

        assert data["cost_top_users"][0]["username"] == "alice"
        assert data["cost_top_users"][0]["cost_cny"] == 1.234568
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_monitoring_returns_cached_payload_on_second_call(api_client):
    """第二次调用命中服务端缓存: 不再跑 DB 聚合, cached=True."""
    app, require_admin_jwt = _admin_override()

    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=_monitoring_side_effect())
    shared_redis = _FakeRedis()

    try:
        with (
            patch("app.api.admin.stats.db", fake_db),
            patch("app.api.admin.stats.get_redis", new_callable=AsyncMock, return_value=shared_redis),
            patch(
                "app.api.admin.stats.count_online_users",
                new_callable=AsyncMock,
                return_value=(7, True),
            ),
        ):
            first = api_client.get("/admin-api/stats/monitoring", params={"days": 7})
            second = api_client.get("/admin-api/stats/monitoring", params={"days": 7})

        assert first.status_code == 200 and second.status_code == 200
        assert first.json()["cached"] is False
        assert second.json()["cached"] is True
        # DB aggregation ran exactly once (7 queries), second call served from cache.
        assert fake_db.query_raw.await_count == 7
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_monitoring_refresh_bypasses_cache(api_client):
    """refresh=true 强制重算, 即使缓存存在."""
    app, require_admin_jwt = _admin_override()

    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=_monitoring_side_effect() + _monitoring_side_effect())
    shared_redis = _FakeRedis()

    try:
        with (
            patch("app.api.admin.stats.db", fake_db),
            patch("app.api.admin.stats.get_redis", new_callable=AsyncMock, return_value=shared_redis),
            patch(
                "app.api.admin.stats.count_online_users",
                new_callable=AsyncMock,
                return_value=(7, True),
            ),
        ):
            api_client.get("/admin-api/stats/monitoring", params={"days": 7})
            forced = api_client.get(
                "/admin-api/stats/monitoring", params={"days": 7, "refresh": "true"}
            )

        assert forced.status_code == 200
        assert forced.json()["cached"] is False
        assert fake_db.query_raw.await_count == 14  # recomputed both times
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_monitoring_survives_redis_failure(api_client):
    """Redis 挂掉时缓存降级为直接计算, 在线人数 redis_available=False."""
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
            patch("app.api.admin.stats.get_redis", new_callable=AsyncMock, side_effect=RuntimeError("redis down")),
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
            patch("app.api.admin.stats.get_redis", new_callable=AsyncMock, return_value=_FakeRedis()),
            patch(
                "app.api.admin.stats.count_online_users",
                new_callable=AsyncMock,
                return_value=(0, True),
            ),
        ):
            response = api_client.get("/admin-api/stats/monitoring", params={"days": 0})

        assert response.status_code == 200
        reg_call = fake_db.query_raw.await_args_list[0]
        assert "1=1" in reg_call.args[0]
        assert len(reg_call.args) == 1
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_online_users_endpoint_paginates_with_login_methods(api_client):
    """/online/users 分页返回在线用户 + 登录方式 + 标识 (微信/手机/邮箱密码)."""
    from types import SimpleNamespace

    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}

    # 3 个在线用户, page_size=2 → 第一页 2 条, total_pages=2.
    online_ids = ["u-wechat", "u-phone", "u-pw"]

    users = [
        SimpleNamespace(id="u-wechat", username="wx_abc", email=None, hashedPassword=None),
        SimpleNamespace(id="u-phone", username="ph_abc", email=None, hashedPassword=None),
    ]
    identities = [
        SimpleNamespace(
            userId="u-wechat", provider="wechat", openid="op-1",
            providerAccountId="op-1", rawProfile={"nickname": "小明"},
        ),
        SimpleNamespace(
            userId="u-phone", provider="phone",
            providerAccountId="13812345678", openid=None, rawProfile={"phone": "13812345678"},
        ),
    ]

    fake_db = MagicMock()
    fake_db.user = MagicMock()
    fake_db.user.find_many = AsyncMock(return_value=users)
    fake_db.authidentity = MagicMock()
    fake_db.authidentity.find_many = AsyncMock(return_value=identities)

    try:
        with (
            patch("app.api.admin.stats.db", fake_db),
            patch(
                "app.api.admin.stats.list_online_user_ids",
                new_callable=AsyncMock,
                return_value=(online_ids, True),
            ),
        ):
            resp = api_client.get(
                "/admin-api/stats/online/users", params={"page": 1, "page_size": 2}
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert data["total_pages"] == 2
        assert data["page"] == 1
        assert len(data["items"]) == 2

        wx = data["items"][0]
        assert wx["username"] == "wx_abc"
        assert wx["methods"][0]["type"] == "wechat"
        assert wx["methods"][0]["identifier"] == "小明"

        ph = data["items"][1]
        assert ph["methods"][0]["type"] == "phone"
        assert ph["methods"][0]["identifier"] == "138****5678"  # masked
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_online_users_password_method_from_hashed_password(api_client):
    """有 hashedPassword 的账号显示 邮箱/密码 登录方式, 标识用 email."""
    from types import SimpleNamespace

    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}

    users = [SimpleNamespace(id="u-pw", username="alice", email="a@b.com", hashedPassword="x")]

    fake_db = MagicMock()
    fake_db.user = MagicMock()
    fake_db.user.find_many = AsyncMock(return_value=users)
    fake_db.authidentity = MagicMock()
    fake_db.authidentity.find_many = AsyncMock(return_value=[])

    try:
        with (
            patch("app.api.admin.stats.db", fake_db),
            patch(
                "app.api.admin.stats.list_online_user_ids",
                new_callable=AsyncMock,
                return_value=(["u-pw"], True),
            ),
        ):
            resp = api_client.get("/admin-api/stats/online/users")

        assert resp.status_code == 200
        item = resp.json()["items"][0]
        assert item["methods"][0]["type"] == "password"
        assert item["methods"][0]["identifier"] == "a@b.com"
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_online_endpoint_is_redis_only(api_client):
    """/online 只读 Redis presence, 不碰 DB."""
    app, require_admin_jwt = _admin_override()
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=AssertionError("online must not query DB"))

    try:
        with (
            patch("app.api.admin.stats.db", fake_db),
            patch(
                "app.api.admin.stats.count_online_users",
                new_callable=AsyncMock,
                return_value=(11, True),
            ),
        ):
            response = api_client.get("/admin-api/stats/online")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 11
        assert data["redis_available"] is True
        assert data["threshold_seconds"] == 90
        fake_db.query_raw.assert_not_awaited()
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)
