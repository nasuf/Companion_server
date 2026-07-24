from unittest.mock import AsyncMock, MagicMock, patch


def _admin_override():
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    return app, require_admin_jwt


class _FakeRedis:
    async def info(self, section=None):
        return {
            "used_memory": 2_624_344,
            "maxmemory": 536_870_912,
            "keyspace_hits": 900,
            "keyspace_misses": 100,
            "connected_clients": 9,
            "evicted_keys": 0,
            "uptime_in_seconds": 4941544,
        }

    async def dbsize(self):
        return 677


def _db_side_effect():
    # asyncio.gather order in _collect_db_metrics: size / tables / conns / setting
    return [
        [{"size_bytes": 253_755_392, "size_pretty": "242 MB"}],
        [
            {"name": "memory_embeddings", "total_bytes": 122_683_392,
             "total_pretty": "117 MB", "rows": 7435},
            {"name": "game_events", "total_bytes": 26_214_400,
             "total_pretty": "25 MB", "rows": 12449},
        ],
        [{"active": 7}],
        [{"max_connections": 50}],
    ]


def _fake_host_metrics():
    return {
        "system": {
            "cpu_count": 4,
            "cpu_percent": 12.5,
            "load_avg": [0.1, 0.2, 0.15],
            "load_percent": 2.5,
            "uptime_seconds": 4991520.0,
        },
        "memory": {
            "total": 8_053_063_680,
            "used": 3_221_225_472,
            "available": 4_831_838_208,
            "percent": 40.0,
            "swap_total": 8_589_934_592,
            "swap_used": 615_514_112,
            "swap_percent": 7.2,
        },
        "disks": [
            {"mount": "/", "label": "系统盘", "total": 53_687_091_200,
             "used": 21_474_836_480, "free": 30_064_771_072, "percent": 43.0},
            {"mount": "/data/chat_media", "label": "数据盘", "total": 52_613_349_376,
             "used": 8_804_527_104, "free": 41_875_931_136, "percent": 18.0},
        ],
        "network": {
            "bytes_recv": 1_000_000, "bytes_sent": 2_000_000,
            "packets_recv": 5000, "packets_sent": 6000,
            "recv_rate_bps": 1234, "sent_rate_bps": 5678,
        },
    }


def test_resources_aggregates_host_db_redis(api_client):
    """/resources 聚合主机 CPU/内存/磁盘/网络 + Postgres + Redis 健康."""
    app, require_admin_jwt = _admin_override()
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=_db_side_effect())
    try:
        with (
            patch("app.api.admin.stats.db", fake_db),
            patch(
                "app.api.admin.stats.get_redis",
                new_callable=AsyncMock,
                return_value=_FakeRedis(),
            ),
            patch(
                "app.api.admin.stats.collect_host_metrics",
                new_callable=AsyncMock,
                return_value=_fake_host_metrics(),
            ),
        ):
            response = api_client.get("/admin-api/stats/resources")
        assert response.status_code == 200
        body = response.json()

        assert body["system"]["cpu_count"] == 4
        assert body["memory"]["percent"] == 40.0
        assert len(body["disks"]) == 2
        assert body["network"]["recv_rate_bps"] == 1234

        db_block = body["database"]
        assert db_block["available"] is True
        assert db_block["size_pretty"] == "242 MB"
        assert db_block["largest_tables"][0]["name"] == "memory_embeddings"
        assert db_block["connections"]["active"] == 7
        assert db_block["connections"]["max"] == 50
        assert db_block["connections"]["percent"] == 14.0

        redis_block = body["redis"]
        assert redis_block["available"] is True
        assert redis_block["used_memory"] == 2_624_344
        assert redis_block["keys"] == 677
        assert redis_block["hit_rate"] == 90.0
        assert "as_of" in body
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_resources_degrades_when_db_and_redis_unavailable(api_client):
    """任一维度失败降级为 available=false, 不 500."""
    app, require_admin_jwt = _admin_override()
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=RuntimeError("db down"))
    try:
        with (
            patch("app.api.admin.stats.db", fake_db),
            patch(
                "app.api.admin.stats.get_redis",
                new_callable=AsyncMock,
                side_effect=RuntimeError("redis down"),
            ),
            patch(
                "app.api.admin.stats.collect_host_metrics",
                new_callable=AsyncMock,
                return_value=_fake_host_metrics(),
            ),
        ):
            response = api_client.get("/admin-api/stats/resources")
        assert response.status_code == 200
        body = response.json()
        assert body["database"]["available"] is False
        assert body["redis"]["available"] is False
        # Host metrics still present.
        assert body["system"]["cpu_count"] == 4
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)
