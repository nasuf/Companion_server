-- Rename the game level ladder (PM copy 2026-08-02) and split its wording into
-- three fields so clients never have to parse a display string.
--
--   stage_name    皮革手套      the glove itself
--   stage_caption 初学起步      the descriptive line shown underneath it
--   tier_name     白            the colour ranking this step inside the stage
--
-- Thresholds are unchanged: five steps per stage, cumulative totals 0 → 13450.
-- Rows are matched on sort_order, so the seed in 20260722193000_game_points
-- still creates them for a fresh database and this migration renames them.

ALTER TABLE game_level_tiers
    ADD COLUMN IF NOT EXISTS stage_caption TEXT NOT NULL DEFAULT '';

UPDATE game_level_tiers SET stage_name = '皮革手套', stage_caption = '初学起步', tier_name = '白' WHERE sort_order = 1;
UPDATE game_level_tiers SET stage_name = '皮革手套', stage_caption = '初学起步', tier_name = '绿' WHERE sort_order = 2;
UPDATE game_level_tiers SET stage_name = '皮革手套', stage_caption = '初学起步', tier_name = '黄' WHERE sort_order = 3;
UPDATE game_level_tiers SET stage_name = '皮革手套', stage_caption = '初学起步', tier_name = '蓝' WHERE sort_order = 4;
UPDATE game_level_tiers SET stage_name = '皮革手套', stage_caption = '初学起步', tier_name = '黑' WHERE sort_order = 5;

UPDATE game_level_tiers SET stage_name = '尼龙手套', stage_caption = '进阶提升', tier_name = '白' WHERE sort_order = 6;
UPDATE game_level_tiers SET stage_name = '尼龙手套', stage_caption = '进阶提升', tier_name = '绿' WHERE sort_order = 7;
UPDATE game_level_tiers SET stage_name = '尼龙手套', stage_caption = '进阶提升', tier_name = '黄' WHERE sort_order = 8;
UPDATE game_level_tiers SET stage_name = '尼龙手套', stage_caption = '进阶提升', tier_name = '蓝' WHERE sort_order = 9;
UPDATE game_level_tiers SET stage_name = '尼龙手套', stage_caption = '进阶提升', tier_name = '黑' WHERE sort_order = 10;

UPDATE game_level_tiers SET stage_name = '战术手套', stage_caption = '精英段位', tier_name = '白' WHERE sort_order = 11;
UPDATE game_level_tiers SET stage_name = '战术手套', stage_caption = '精英段位', tier_name = '绿' WHERE sort_order = 12;
UPDATE game_level_tiers SET stage_name = '战术手套', stage_caption = '精英段位', tier_name = '黄' WHERE sort_order = 13;
UPDATE game_level_tiers SET stage_name = '战术手套', stage_caption = '精英段位', tier_name = '蓝' WHERE sort_order = 14;
UPDATE game_level_tiers SET stage_name = '战术手套', stage_caption = '精英段位', tier_name = '黑' WHERE sort_order = 15;

UPDATE game_level_tiers SET stage_name = '巨岩手套', stage_caption = '顶尖高手', tier_name = '白' WHERE sort_order = 16;
UPDATE game_level_tiers SET stage_name = '巨岩手套', stage_caption = '顶尖高手', tier_name = '绿' WHERE sort_order = 17;
UPDATE game_level_tiers SET stage_name = '巨岩手套', stage_caption = '顶尖高手', tier_name = '黄' WHERE sort_order = 18;
UPDATE game_level_tiers SET stage_name = '巨岩手套', stage_caption = '顶尖高手', tier_name = '蓝' WHERE sort_order = 19;
UPDATE game_level_tiers SET stage_name = '巨岩手套', stage_caption = '顶尖高手', tier_name = '黑' WHERE sort_order = 20;

UPDATE game_level_tiers SET stage_name = '玄铁手套', stage_caption = '领域大师', tier_name = '白' WHERE sort_order = 21;
UPDATE game_level_tiers SET stage_name = '玄铁手套', stage_caption = '领域大师', tier_name = '绿' WHERE sort_order = 22;
UPDATE game_level_tiers SET stage_name = '玄铁手套', stage_caption = '领域大师', tier_name = '黄' WHERE sort_order = 23;
UPDATE game_level_tiers SET stage_name = '玄铁手套', stage_caption = '领域大师', tier_name = '蓝' WHERE sort_order = 24;
UPDATE game_level_tiers SET stage_name = '玄铁手套', stage_caption = '领域大师', tier_name = '彩' WHERE sort_order = 25;
