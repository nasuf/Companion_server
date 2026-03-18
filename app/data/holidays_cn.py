"""中国节假日数据（2025-2027）。

包含法定假日、传统节日、国际常见节日。
按 ISO 日期字符串索引，支持快速查询。
"""

from __future__ import annotations

# type: legal(法定) / traditional(传统) / international(国际)
_RAW: list[dict] = [
    # ===== 2025 =====
    {"name": "元旦", "date": "2025-01-01", "type": "legal"},
    {"name": "除夕", "date": "2025-01-28", "type": "traditional"},
    {"name": "春节", "date": "2025-01-29", "type": "legal"},
    {"name": "春节假期", "date": "2025-01-30", "type": "legal"},
    {"name": "春节假期", "date": "2025-01-31", "type": "legal"},
    {"name": "情人节", "date": "2025-02-14", "type": "international"},
    {"name": "元宵节", "date": "2025-02-12", "type": "traditional"},
    {"name": "妇女节", "date": "2025-03-08", "type": "international"},
    {"name": "清明节", "date": "2025-04-04", "type": "legal"},
    {"name": "劳动节", "date": "2025-05-01", "type": "legal"},
    {"name": "母亲节", "date": "2025-05-11", "type": "international"},
    {"name": "端午节", "date": "2025-05-31", "type": "legal"},
    {"name": "儿童节", "date": "2025-06-01", "type": "international"},
    {"name": "父亲节", "date": "2025-06-15", "type": "international"},
    {"name": "七夕节", "date": "2025-08-29", "type": "traditional"},
    {"name": "中秋节", "date": "2025-10-06", "type": "legal"},
    {"name": "国庆节", "date": "2025-10-01", "type": "legal"},
    {"name": "国庆假期", "date": "2025-10-02", "type": "legal"},
    {"name": "国庆假期", "date": "2025-10-03", "type": "legal"},
    {"name": "重阳节", "date": "2025-10-29", "type": "traditional"},
    {"name": "万圣节", "date": "2025-10-31", "type": "international"},
    {"name": "感恩节", "date": "2025-11-27", "type": "international"},
    {"name": "冬至", "date": "2025-12-22", "type": "traditional"},
    {"name": "平安夜", "date": "2025-12-24", "type": "international"},
    {"name": "圣诞节", "date": "2025-12-25", "type": "international"},
    # ===== 2026 =====
    {"name": "元旦", "date": "2026-01-01", "type": "legal"},
    {"name": "除夕", "date": "2026-02-16", "type": "traditional"},
    {"name": "春节", "date": "2026-02-17", "type": "legal"},
    {"name": "春节假期", "date": "2026-02-18", "type": "legal"},
    {"name": "春节假期", "date": "2026-02-19", "type": "legal"},
    {"name": "情人节", "date": "2026-02-14", "type": "international"},
    {"name": "元宵节", "date": "2026-03-03", "type": "traditional"},
    {"name": "妇女节", "date": "2026-03-08", "type": "international"},
    {"name": "清明节", "date": "2026-04-05", "type": "legal"},
    {"name": "劳动节", "date": "2026-05-01", "type": "legal"},
    {"name": "母亲节", "date": "2026-05-10", "type": "international"},
    {"name": "端午节", "date": "2026-06-19", "type": "legal"},
    {"name": "儿童节", "date": "2026-06-01", "type": "international"},
    {"name": "父亲节", "date": "2026-06-21", "type": "international"},
    {"name": "七夕节", "date": "2026-08-19", "type": "traditional"},
    {"name": "中秋节", "date": "2026-09-25", "type": "legal"},
    {"name": "国庆节", "date": "2026-10-01", "type": "legal"},
    {"name": "国庆假期", "date": "2026-10-02", "type": "legal"},
    {"name": "国庆假期", "date": "2026-10-03", "type": "legal"},
    {"name": "重阳节", "date": "2026-10-18", "type": "traditional"},
    {"name": "万圣节", "date": "2026-10-31", "type": "international"},
    {"name": "感恩节", "date": "2026-11-26", "type": "international"},
    {"name": "冬至", "date": "2026-12-22", "type": "traditional"},
    {"name": "平安夜", "date": "2026-12-24", "type": "international"},
    {"name": "圣诞节", "date": "2026-12-25", "type": "international"},
    # ===== 2027 =====
    {"name": "元旦", "date": "2027-01-01", "type": "legal"},
    {"name": "除夕", "date": "2027-02-05", "type": "traditional"},
    {"name": "春节", "date": "2027-02-06", "type": "legal"},
    {"name": "春节假期", "date": "2027-02-07", "type": "legal"},
    {"name": "春节假期", "date": "2027-02-08", "type": "legal"},
    {"name": "情人节", "date": "2027-02-14", "type": "international"},
    {"name": "元宵节", "date": "2027-02-20", "type": "traditional"},
    {"name": "妇女节", "date": "2027-03-08", "type": "international"},
    {"name": "清明节", "date": "2027-04-05", "type": "legal"},
    {"name": "劳动节", "date": "2027-05-01", "type": "legal"},
    {"name": "母亲节", "date": "2027-05-09", "type": "international"},
    {"name": "端午节", "date": "2027-06-09", "type": "legal"},
    {"name": "儿童节", "date": "2027-06-01", "type": "international"},
    {"name": "父亲节", "date": "2027-06-20", "type": "international"},
    {"name": "七夕节", "date": "2027-08-08", "type": "traditional"},
    {"name": "中秋节", "date": "2027-09-15", "type": "legal"},
    {"name": "国庆节", "date": "2027-10-01", "type": "legal"},
    {"name": "国庆假期", "date": "2027-10-02", "type": "legal"},
    {"name": "国庆假期", "date": "2027-10-03", "type": "legal"},
    {"name": "重阳节", "date": "2027-10-08", "type": "traditional"},
    {"name": "万圣节", "date": "2027-10-31", "type": "international"},
    {"name": "感恩节", "date": "2027-11-25", "type": "international"},
    {"name": "冬至", "date": "2027-12-22", "type": "traditional"},
    {"name": "平安夜", "date": "2027-12-24", "type": "international"},
    {"name": "圣诞节", "date": "2027-12-25", "type": "international"},
]

# 调休上班日（以2026年为例）
WORKDAY_SWAPS: set[str] = {
    "2026-02-15",  # 春节调休
    "2026-10-10",  # 国庆调休
}

# 按日期索引，O(1)查询
HOLIDAYS: dict[str, dict] = {
    entry["date"]: {"name": entry["name"], "type": entry["type"]}
    for entry in _RAW
}

# 节日名称 → 日期列表映射（供时间解析器使用）
HOLIDAY_NAME_DATES: dict[str, list[str]] = {}
for _entry in _RAW:
    HOLIDAY_NAME_DATES.setdefault(_entry["name"], []).append(_entry["date"])
