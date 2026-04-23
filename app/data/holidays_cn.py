"""[LEGACY] 中国节假日硬编码数据 — 一次性 seed 源.

自 `holidays` 表 DB 化后 (migration 20260423120000_holidays + admin
"日历管理" 页), 运行时不再直接 import 这里的数据. 本文件唯一用途:
`scripts/seed_holidays_from_hardcoded.py` 一次性把历史数据导入 DB,
之后 admin 通过 UI 维护 + weekly refresh cron 从外部源同步.

⚠️ 请**不要**在新代码里引用 `HOLIDAYS` / `HOLIDAY_NAME_DATES` /
`WORKDAY_SWAPS` — 改用 `app.services.schedule_domain.holiday_cache`:
  - 查节日: `holiday_cache.get_by_date(d)` (sync, 启动时预加载)
  - 查调休: `holiday_cache.is_workday_swap(d)`
  - 按名字查日期: `holiday_cache.list_dates_for_name(name)`

数据源 (历史):
  - 2025 年：国务院办公厅 2024-11-12 通知
  - 2026 年：国务院办公厅 2025-12 通知
  - 2027 年：推演数据 (官方通知发布后由 admin 通过 UI 刷新)
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
    {"name": "春节假期", "date": "2025-02-01", "type": "legal"},
    {"name": "春节假期", "date": "2025-02-02", "type": "legal"},
    {"name": "春节假期", "date": "2025-02-03", "type": "legal"},
    {"name": "春节假期", "date": "2025-02-04", "type": "legal"},
    {"name": "情人节", "date": "2025-02-14", "type": "international"},
    {"name": "元宵节", "date": "2025-02-12", "type": "traditional"},
    {"name": "妇女节", "date": "2025-03-08", "type": "international"},
    {"name": "清明节", "date": "2025-04-04", "type": "legal"},
    {"name": "清明假期", "date": "2025-04-05", "type": "legal"},
    {"name": "清明假期", "date": "2025-04-06", "type": "legal"},
    {"name": "劳动节", "date": "2025-05-01", "type": "legal"},
    {"name": "劳动假期", "date": "2025-05-02", "type": "legal"},
    {"name": "劳动假期", "date": "2025-05-03", "type": "legal"},
    {"name": "劳动假期", "date": "2025-05-04", "type": "legal"},
    {"name": "劳动假期", "date": "2025-05-05", "type": "legal"},
    {"name": "母亲节", "date": "2025-05-11", "type": "international"},
    {"name": "端午节", "date": "2025-05-31", "type": "legal"},
    {"name": "端午假期", "date": "2025-06-01", "type": "legal"},
    {"name": "端午假期", "date": "2025-06-02", "type": "legal"},
    {"name": "儿童节", "date": "2025-06-01", "type": "international"},
    {"name": "父亲节", "date": "2025-06-15", "type": "international"},
    {"name": "七夕节", "date": "2025-08-29", "type": "traditional"},
    {"name": "国庆节", "date": "2025-10-01", "type": "legal"},
    {"name": "国庆假期", "date": "2025-10-02", "type": "legal"},
    {"name": "国庆假期", "date": "2025-10-03", "type": "legal"},
    {"name": "国庆假期", "date": "2025-10-04", "type": "legal"},
    {"name": "国庆假期", "date": "2025-10-05", "type": "legal"},
    {"name": "中秋节", "date": "2025-10-06", "type": "legal"},
    {"name": "国庆假期", "date": "2025-10-07", "type": "legal"},
    {"name": "国庆假期", "date": "2025-10-08", "type": "legal"},
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
    {"name": "春节假期", "date": "2026-02-20", "type": "legal"},
    {"name": "春节假期", "date": "2026-02-21", "type": "legal"},
    {"name": "春节假期", "date": "2026-02-22", "type": "legal"},
    {"name": "春节假期", "date": "2026-02-23", "type": "legal"},
    {"name": "情人节", "date": "2026-02-14", "type": "international"},
    {"name": "元宵节", "date": "2026-03-03", "type": "traditional"},
    {"name": "妇女节", "date": "2026-03-08", "type": "international"},
    {"name": "清明节", "date": "2026-04-05", "type": "legal"},
    {"name": "清明假期", "date": "2026-04-06", "type": "legal"},
    {"name": "清明假期", "date": "2026-04-07", "type": "legal"},
    {"name": "劳动节", "date": "2026-05-01", "type": "legal"},
    {"name": "劳动假期", "date": "2026-05-02", "type": "legal"},
    {"name": "劳动假期", "date": "2026-05-03", "type": "legal"},
    {"name": "劳动假期", "date": "2026-05-04", "type": "legal"},
    {"name": "劳动假期", "date": "2026-05-05", "type": "legal"},
    {"name": "母亲节", "date": "2026-05-10", "type": "international"},
    {"name": "端午节", "date": "2026-06-19", "type": "legal"},
    {"name": "端午假期", "date": "2026-06-20", "type": "legal"},
    {"name": "端午假期", "date": "2026-06-21", "type": "legal"},
    {"name": "儿童节", "date": "2026-06-01", "type": "international"},
    {"name": "父亲节", "date": "2026-06-21", "type": "international"},
    {"name": "七夕节", "date": "2026-08-19", "type": "traditional"},
    {"name": "中秋节", "date": "2026-09-25", "type": "legal"},
    {"name": "国庆节", "date": "2026-10-01", "type": "legal"},
    {"name": "国庆假期", "date": "2026-10-02", "type": "legal"},
    {"name": "国庆假期", "date": "2026-10-03", "type": "legal"},
    {"name": "国庆假期", "date": "2026-10-04", "type": "legal"},
    {"name": "国庆假期", "date": "2026-10-05", "type": "legal"},
    {"name": "国庆假期", "date": "2026-10-06", "type": "legal"},
    {"name": "国庆假期", "date": "2026-10-07", "type": "legal"},
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
    {"name": "春节假期", "date": "2027-02-09", "type": "legal"},
    {"name": "春节假期", "date": "2027-02-10", "type": "legal"},
    {"name": "春节假期", "date": "2027-02-11", "type": "legal"},
    {"name": "春节假期", "date": "2027-02-12", "type": "legal"},
    {"name": "情人节", "date": "2027-02-14", "type": "international"},
    {"name": "元宵节", "date": "2027-02-20", "type": "traditional"},
    {"name": "妇女节", "date": "2027-03-08", "type": "international"},
    {"name": "清明节", "date": "2027-04-05", "type": "legal"},
    {"name": "清明假期", "date": "2027-04-06", "type": "legal"},
    {"name": "清明假期", "date": "2027-04-07", "type": "legal"},
    {"name": "劳动节", "date": "2027-05-01", "type": "legal"},
    {"name": "劳动假期", "date": "2027-05-02", "type": "legal"},
    {"name": "劳动假期", "date": "2027-05-03", "type": "legal"},
    {"name": "劳动假期", "date": "2027-05-04", "type": "legal"},
    {"name": "劳动假期", "date": "2027-05-05", "type": "legal"},
    {"name": "母亲节", "date": "2027-05-09", "type": "international"},
    {"name": "端午节", "date": "2027-06-09", "type": "legal"},
    {"name": "端午假期", "date": "2027-06-10", "type": "legal"},
    {"name": "端午假期", "date": "2027-06-11", "type": "legal"},
    {"name": "儿童节", "date": "2027-06-01", "type": "international"},
    {"name": "父亲节", "date": "2027-06-20", "type": "international"},
    {"name": "七夕节", "date": "2027-08-08", "type": "traditional"},
    {"name": "中秋节", "date": "2027-09-15", "type": "legal"},
    {"name": "国庆节", "date": "2027-10-01", "type": "legal"},
    {"name": "国庆假期", "date": "2027-10-02", "type": "legal"},
    {"name": "国庆假期", "date": "2027-10-03", "type": "legal"},
    {"name": "国庆假期", "date": "2027-10-04", "type": "legal"},
    {"name": "国庆假期", "date": "2027-10-05", "type": "legal"},
    {"name": "国庆假期", "date": "2027-10-06", "type": "legal"},
    {"name": "国庆假期", "date": "2027-10-07", "type": "legal"},
    {"name": "重阳节", "date": "2027-10-08", "type": "traditional"},
    {"name": "万圣节", "date": "2027-10-31", "type": "international"},
    {"name": "感恩节", "date": "2027-11-25", "type": "international"},
    {"name": "冬至", "date": "2027-12-22", "type": "traditional"},
    {"name": "平安夜", "date": "2027-12-24", "type": "international"},
    {"name": "圣诞节", "date": "2027-12-25", "type": "international"},
]

# 调休上班日（原本为周末但需要补班）。
# 2025 年：国务院办公厅 2024-11-12 通知。
# 2026 年：国务院办公厅 2025 年底通知（含春节、国庆两次补班）。
# 2027 年：暂留空，待官方正式通知后再补。
WORKDAY_SWAPS: set[str] = {
    # 2025
    "2025-01-26",  # 春节调休（周日上班）
    "2025-02-08",  # 春节调休（周六上班）
    "2025-04-27",  # 劳动节调休（周日上班）
    "2025-09-28",  # 国庆中秋调休（周日上班）
    "2025-10-11",  # 国庆中秋调休（周六上班）
    # 2026
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
