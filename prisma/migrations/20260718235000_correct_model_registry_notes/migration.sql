-- Replace obsolete seed notes while preserving anything customized in Admin.
UPDATE "model_registry"
SET "notes" = '角色扮演模型；默认复用 DASHSCOPE_BASE_URL，若 workspace 提供独立兼容地址，可通过 DASHSCOPE_CHARACTER_BASE_URL 覆盖。',
    "updated_at" = NOW()
WHERE "provider" = 'dashscope'
  AND "identifier" = 'qwen-plus-character'
  AND "notes" = '角色扮演模型；需配置百炼 workspace 专属兼容接口地址。';

UPDATE "model_registry"
SET "notes" = '低延迟角色扮演模型；默认复用 DASHSCOPE_BASE_URL，若 workspace 提供独立兼容地址，可通过 DASHSCOPE_CHARACTER_BASE_URL 覆盖。',
    "updated_at" = NOW()
WHERE "provider" = 'dashscope'
  AND "identifier" = 'qwen-flash-character'
  AND "notes" = '低延迟角色扮演模型；需配置百炼 workspace 专属兼容接口地址。';

UPDATE "model_registry"
SET "notes" = '长上下文角色扮演快照；默认复用 DASHSCOPE_BASE_URL，若 workspace 提供独立兼容地址，可通过 DASHSCOPE_CHARACTER_BASE_URL 覆盖。',
    "updated_at" = NOW()
WHERE "provider" = 'dashscope'
  AND "identifier" = 'qwen-flash-character-2026-02-26'
  AND "notes" = '长上下文角色扮演快照；需配置百炼 workspace 专属兼容接口地址。';

UPDATE "model_registry"
SET "notes" = 'DeepSeek 官方 API；价格字段已配置，实际账单以 DeepSeek 控制台为准。',
    "updated_at" = NOW()
WHERE "provider" = 'deepseek'
  AND "identifier" IN ('deepseek-v4-pro', 'deepseek-v4-flash')
  AND "notes" = 'Direct DeepSeek API; fill pricing manually in admin.';
