-- Per spec《产品手册·背景信息》§1.2: MBTI 全面替换 7 维 / Big Five 作为
-- AI 性格的 canonical 表达。drop 4 个旧字段，新增 currentMbti 用于
-- trait_adjustment 调整后的状态。
--
-- 用户已确认不需要回填历史数据（数据将被清空），所以直接 DROP COLUMN。

ALTER TABLE "ai_agents"
    DROP COLUMN IF EXISTS "personality",
    DROP COLUMN IF EXISTS "seven_dim_traits",
    DROP COLUMN IF EXISTS "current_traits",
    DROP COLUMN IF EXISTS "traits_history";

ALTER TABLE "ai_agents"
    ADD COLUMN "current_mbti" JSONB;
