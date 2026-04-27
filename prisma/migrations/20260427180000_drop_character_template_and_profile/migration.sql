-- Plan B: agent 创建时单步 LLM 生 background, admin 不再预生背景池.
-- DROP character_templates / character_profiles 两表 + ai_agents.character_profile_id 列.

-- 1. 先删 ai_agents.character_profile_id (有 FK 引用 character_profiles 时会阻止 DROP TABLE)
ALTER TABLE "ai_agents" DROP COLUMN IF EXISTS "character_profile_id";

-- 2. character_profiles 引用 character_templates / career_templates, 先 DROP profiles
DROP TABLE IF EXISTS "character_profiles";
DROP TABLE IF EXISTS "character_templates";

-- 3. 清理 prompt_templates 表中不再被 registry 引用的 4 个旧 key 行 (避免孤儿数据).
--    新 key character.generation 由 ensure_prompt_templates 启动时 seed.
DELETE FROM "prompt_templates" WHERE "key" IN (
    'character.template_header',
    'character.template_requirements',
    'life_story.main_gap_fill'
);
