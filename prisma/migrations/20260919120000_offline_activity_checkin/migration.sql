-- 线下活动「打卡闭环」重构 P0：到达/预言/归档/经纬度列 + 拍摄条件集合 + 思绪碎片 + 媒体分流列
-- 对齐方案文档《活动推荐模块重构方案.md》§3。

-- 1) 活动主表新增列 --------------------------------------------------------
ALTER TABLE offline_activity_recommendations
    ADD COLUMN IF NOT EXISTS place_lat DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS place_lng DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS place_key TEXT,
    ADD COLUMN IF NOT EXISTS reached BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS arrival_confirmed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS arrival_lat DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS arrival_lng DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS conditions_ready_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS prophecy_text TEXT,
    ADD COLUMN IF NOT EXISTS prophecy_drawn_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS auto_archived BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS auto_archive_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS travel_note TEXT;

-- 归档 cron 扫描：status=accepted AND reached AND auto_archive_at <= now
CREATE INDEX IF NOT EXISTS offline_activity_autoarchive_idx
    ON offline_activity_recommendations (status, auto_archive_at);

-- 2) 媒体表新增列（到达后分流 + 识图去重） -----------------------------------
ALTER TABLE offline_activity_media
    ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'material',
    ADD COLUMN IF NOT EXISTS recognized BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS content_fingerprint TEXT,
    ADD COLUMN IF NOT EXISTS transcript TEXT,
    ADD COLUMN IF NOT EXISTS source_message_id TEXT;

-- 3) 拍摄条件集合（到达后由大模型生成 3–5 条，永不下发前端明文） ----------------
CREATE TABLE IF NOT EXISTS offline_shooting_conditions (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    recommendation_id TEXT NOT NULL REFERENCES offline_activity_recommendations(id) ON DELETE CASCADE,
    short_name TEXT NOT NULL,
    criteria TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0,
    triggered BOOLEAN NOT NULL DEFAULT FALSE,
    triggered_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS offline_shooting_conditions_recommendation_idx
    ON offline_shooting_conditions (recommendation_id);

-- 4) 思绪碎片 ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS offline_thought_fragments (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    recommendation_id TEXT NOT NULL REFERENCES offline_activity_recommendations(id) ON DELETE CASCADE,
    tier TEXT NOT NULL,
    text TEXT NOT NULL,
    lead_in TEXT,
    condition_id TEXT,
    snapshot_media_id TEXT,
    source_message_id TEXT,
    content_fingerprint TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS offline_thought_fragment_recommendation_idx
    ON offline_thought_fragments (recommendation_id, created_at DESC);
