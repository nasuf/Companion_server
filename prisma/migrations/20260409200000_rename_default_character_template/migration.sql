-- Rename the default character template to its canonical name "标准角色背景模板".
-- Idempotent: handles both legacy names ("标准角色模板 v1" and "标准角色背景模板 v1").

UPDATE character_templates
SET name = '标准角色背景模板',
    updated_at = CURRENT_TIMESTAMP
WHERE name IN ('标准角色模板 v1', '标准角色背景模板 v1');
