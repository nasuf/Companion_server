-- 节假日数据 DB 化 (spec audit 2026-04-23):
-- 把 app/data/holidays_cn.py 硬编码数据搬到 DB, 让 admin 能通过 UI
-- 触发外部源 (chinesecalendar + nager.date) 查询后挑选、保存。
-- 运行时 is_holiday() 改查 DB + Redis 缓存, lunardate 兜底保留。
--
-- 唯一约束: (date, country_code, name) — 允许同一天同国家存在不同名称的节日
-- (例如 2026-10-06 同时是 "中秋节" 和 "国庆假期"), 只禁止完全重复记录。

CREATE TABLE IF NOT EXISTS "holidays" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "date" DATE NOT NULL,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "country_code" TEXT NOT NULL DEFAULT 'CN',
    "is_workday_swap" BOOLEAN NOT NULL DEFAULT FALSE,
    "source" TEXT NOT NULL,
    "metadata" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX "holidays_date_country_code_name_key"
    ON "holidays"("date", "country_code", "name");

CREATE INDEX "holidays_date_idx" ON "holidays"("date");
CREATE INDEX "holidays_country_code_idx" ON "holidays"("country_code");
CREATE INDEX "holidays_source_idx" ON "holidays"("source");
