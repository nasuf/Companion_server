"""UTC 时间戳 → 本地小时/日期的 SQL 换算。

真实事故 (2026-10-17 发现): 主动消息的节律系数和成就"聊天天数"都对
`timestamp without time zone` 的 UTC 列写了 `AT TIME ZONE 'Asia/Shanghai'`,
结果小时被算早 8 小时 —— 节律系数一直在拿错开 8 小时的时段做参考, 全程不报错。
"""

from __future__ import annotations

import ast
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pytest
from prisma import Prisma

from app.services.proactive import history
from app.services.schedule_domain.sql_time import local_date_sql, local_hour_sql

UTC = timezone.utc
APP_DIR = Path(__file__).resolve().parents[1] / "app"
# 作用在带时区的值上才是对的: CURRENT_TIMESTAMP / now() / ::timestamptz, 或者先用
# AT TIME ZONE 'UTC' 声明了这是 UTC 的两步换算
_TZ_AWARE_OPERAND = re.compile(
    r"(CURRENT_TIMESTAMP|now\(\)|timestamptz|AT TIME ZONE 'UTC')\)?\s*$", re.IGNORECASE,
)


def _code_only(source: str) -> str:
    """去掉 docstring 与注释 (注释里正是在讲"别这么写")."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_no_local_zone_conversion_on_naive_utc_columns():
    offenders = []
    for path in APP_DIR.rglob("*.py"):
        code = _code_only(path.read_text(encoding="utf-8"))
        for match in re.finditer(r"AT TIME ZONE '(?!UTC')[^']+'", code, re.IGNORECASE):
            operand = code[max(0, match.start() - 40):match.start()].rstrip()
            if not _TZ_AWARE_OPERAND.search(operand):
                offenders.append(f"{path.relative_to(APP_DIR)}: …{operand[-30:]} {match.group(0)}")
    assert not offenders, (
        "库里的时间戳是不带时区的 UTC, 对它写 AT TIME ZONE '<本地时区>' 会把时间往回减 —— "
        "用 app.services.schedule_domain.sql_time:\n" + "\n".join(offenders)
    )


def test_helpers_add_the_offset():
    assert local_hour_sql("created_at") == "EXTRACT(HOUR FROM created_at + INTERVAL '8 hours')::int"
    assert local_date_sql("m.created_at") == "(m.created_at + INTERVAL '8 hours')::date"


# ── 真实 PostgreSQL ────────────────────────────────────────────────────

@pytest.fixture
async def pg():
    url = os.environ.get("PROACTIVE_E2E_DATABASE_URL", "")
    if not url:
        pytest.skip("Local-time SQL requires isolated PostgreSQL")
    parsed = urlsplit(url)
    assert parsed.hostname in {"localhost", "127.0.0.1"}
    assert parsed.path == "/companion_proactive_e2e"
    params = dict(parse_qsl(parsed.query))
    params["connection_limit"] = "1"  # 临时表只在这条连接上可见
    database = Prisma(datasource={"url": urlunsplit(parsed._replace(query=urlencode(params)))},
                      http={"trust_env": False})
    await database.connect()
    try:
        # 同名临时表遮住正式表 (pg_temp 优先于 public), 不碰任何真实数据也免去外键
        await database.execute_raw(
            """CREATE TEMP TABLE proactive_event_logs (
                agent_id TEXT, user_id TEXT, workspace_id TEXT, event_type TEXT,
                payload JSONB, created_at TIMESTAMP(3))"""
        )
        yield database
    finally:
        await database.disconnect()


def _naive(ts: datetime) -> str:
    return ts.astimezone(UTC).replace(tzinfo=None).isoformat()


async def test_helpers_match_shanghai_wall_clock(pg):
    # UTC 2026-10-14 14:00 = 北京时间 22:00; UTC 17:00 = 北京时间次日 01:00
    rows = await pg.query_raw(
        f"""SELECT {local_hour_sql("t")} AS h, {local_date_sql("t")}::text AS d,
                   {local_hour_sql("u")} AS h2, {local_date_sql("u")}::text AS d2
            FROM (SELECT TIMESTAMP '2026-10-14 14:00:00' AS t,
                         TIMESTAMP '2026-10-14 17:00:00' AS u) x"""
    )
    assert rows[0] == {"h": 22, "d": "2026-10-14", "h2": 1, "d2": "2026-10-15"}


async def test_rhythm_counts_the_same_local_hour(pg, monkeypatch):
    """北京时间 22 点用户总会回 → 22 点放大命中率; 早上 6 点 (旧 bug 会对上的时段) 不受影响."""
    monkeypatch.setattr(history, "db", pg)
    replied_at = datetime(2026, 10, 14, 14, 5, tzinfo=UTC)  # 北京时间 22:05
    for day in range(3):
        for event in ("message_sent", "user_replied"):
            await pg.execute_raw(
                "INSERT INTO proactive_event_logs VALUES ($1,$2,$3,$4,'{}'::jsonb,$5::timestamp)",
                "a1", "u1", "ws-1", event, _naive(replied_at - timedelta(days=day)),
            )

    at_22 = await history.get_proactive_rhythm_adjustment(
        "a1", "u1", workspace_id="ws-1", now=datetime(2026, 10, 15, 14, 30, tzinfo=UTC),
    )
    assert at_22["local_hour"] == 22
    assert at_22["components"]["replied_same_hour"] == 3
    assert at_22["multiplier"] == 1.25

    at_06 = await history.get_proactive_rhythm_adjustment(
        "a1", "u1", workspace_id="ws-1", now=datetime(2026, 10, 15, 22, 30, tzinfo=UTC),
    )
    assert at_06["local_hour"] == 6
    assert at_06["components"]["replied_same_hour"] == 0
    assert at_06["multiplier"] == 1.0
