"""SQL 里把 UTC 时间戳换算成本地 (UTC+8) 小时 / 日期的唯一写法。

库里的时间戳是 `timestamp without time zone` 存 UTC。对这种列写
`AT TIME ZONE 'Asia/Shanghai'` 的语义是"把这个值当作上海时间, 转成 UTC", 方向跟
直觉相反 —— 会**减** 8 小时: 晚上 21 点被算成早上 5 点, 而整条链路不报任何错。
这里统一用加固定偏移, 中国大陆不实行夏令时, 固定 +8 是精确的。
"""

from __future__ import annotations

LOCAL_UTC_OFFSET_HOURS = 8


def local_hour_sql(column: str) -> str:
    """`column` 对应的本地小时 (0-23)。"""
    return f"EXTRACT(HOUR FROM {column} + INTERVAL '{LOCAL_UTC_OFFSET_HOURS} hours')::int"


def local_date_sql(column: str) -> str:
    """`column` 对应的本地日期。"""
    return f"({column} + INTERVAL '{LOCAL_UTC_OFFSET_HOURS} hours')::date"
