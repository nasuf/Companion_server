-- Agent 昵称: 建号时与姓名一起从姓名库快照。存量 agent 按 (name, gender) 精确匹配
-- 姓名库回填; 匹配不到的 (手填名 / 文档导入名, 如「小伴」) 保持 NULL, 展示回落到 name。
-- (gender, name) 在 name_templates 上唯一, 所以每个 agent 至多匹配一行。
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
ALTER TABLE ai_agents ADD COLUMN IF NOT EXISTS nickname TEXT;

UPDATE ai_agents a
   SET nickname = btrim(nt.nickname)
  FROM name_templates nt
 WHERE a.nickname IS NULL
   AND nt.name = a.name
   AND nt.gender = a.gender
   AND btrim(nt.nickname) <> '';
COMMIT;
