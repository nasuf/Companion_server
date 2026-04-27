-- character_profiles.status 简化: archived 状态合并到 draft.
-- 业务侧只保留 draft / published 二态, 历史"已归档"语义等同于"未发布",
-- 直接迁移到 draft 不丢信息 (未发布的 profile 不会被 life_story.select_
-- character_profile 召回 — 它只取 status = 'published')。
UPDATE "character_profiles" SET "status" = 'draft' WHERE "status" = 'archived';
