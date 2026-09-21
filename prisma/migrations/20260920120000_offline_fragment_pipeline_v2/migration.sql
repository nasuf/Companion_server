-- 思绪碎片管线 v2（对齐 PM 9.19 提示词文档）：拍摄「物品」大类 + 分档预生成回忆池
-- + 未命中方向暗示计数。触发概率/等级/单场上限仍在代码里按 spec §4.9/§4.10 执行。

-- 1) 拍摄条件表升级为「拍摄物品」：加 category；criteria 放开默认空（物品无判定要点）。
ALTER TABLE offline_shooting_conditions
    ADD COLUMN IF NOT EXISTS category TEXT;
ALTER TABLE offline_shooting_conditions
    ALTER COLUMN criteria SET DEFAULT '';

-- 2) 活动主表：未命中 / 已发暗示计数。
ALTER TABLE offline_activity_recommendations
    ADD COLUMN IF NOT EXISTS miss_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS hint_count INTEGER NOT NULL DEFAULT 0;

-- 3) 分档预生成回忆池：每个拍摄物品 × 档位(rare/epic/legendary) 一段 AI 预置回忆。
CREATE TABLE IF NOT EXISTS offline_prewritten_fragments (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    recommendation_id TEXT NOT NULL REFERENCES offline_activity_recommendations(id) ON DELETE CASCADE,
    condition_id TEXT NOT NULL REFERENCES offline_shooting_conditions(id) ON DELETE CASCADE,
    tier TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS offline_prewritten_fragment_recommendation_idx
    ON offline_prewritten_fragments (recommendation_id);
CREATE INDEX IF NOT EXISTS offline_prewritten_fragment_condition_tier_idx
    ON offline_prewritten_fragments (condition_id, tier);
