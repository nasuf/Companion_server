from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

# 各电商/物流 provider 共用的解析工具，避免 _parse_dt 在多个 provider 文件里重复。

_FALLBACK_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S")


def parse_provider_dt(value: Any) -> datetime | None:
    """把 provider 返回的时间字段解析成带时区的 datetime（无时区按 UTC 补齐）。

    依次尝试：已是 datetime → ISO8601（含末尾 Z）→ 常见非标准格式。全部失败返回 None。
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not (isinstance(value, str) and value.strip()):
        return None
    iso_text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        pass
    for fmt in _FALLBACK_FORMATS:
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.replace(tzinfo=UTC)
        except ValueError:
            continue
    return None
