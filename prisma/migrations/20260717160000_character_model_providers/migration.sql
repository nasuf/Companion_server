ALTER TABLE "system_config"
    ADD COLUMN IF NOT EXISTS "remote_chat_provider" TEXT,
    ADD COLUMN IF NOT EXISTS "remote_small_provider" TEXT;

ALTER TABLE "agent_config_overrides"
    ADD COLUMN IF NOT EXISTS "remote_chat_provider" TEXT,
    ADD COLUMN IF NOT EXISTS "remote_small_provider" TEXT;

-- Preserve the former one-provider behavior for existing system and agent rows.
UPDATE "system_config"
SET "remote_chat_provider" = COALESCE("remote_chat_provider", "remote_provider"),
    "remote_small_provider" = COALESCE("remote_small_provider", "remote_provider")
WHERE "remote_provider" IS NOT NULL
  AND ("remote_chat_provider" IS NULL OR "remote_small_provider" IS NULL);

UPDATE "agent_config_overrides"
SET "remote_chat_provider" = COALESCE("remote_chat_provider", "remote_provider"),
    "remote_small_provider" = COALESCE("remote_small_provider", "remote_provider")
WHERE "remote_provider" IS NOT NULL
  AND ("remote_chat_provider" IS NULL OR "remote_small_provider" IS NULL);

-- Stable public model identifiers can be seeded. Ark Character uses an
-- account-specific model/endpoint id, so admins register the exact console id.
INSERT INTO "model_registry" (
    "id", "identifier", "display_name", "provider", "enabled",
    "context_window", "input_cost_per_million", "output_cost_per_million",
    "notes", "created_at", "updated_at"
) VALUES
    (
        gen_random_uuid(), 'qwen-plus-character', 'Qwen Plus Character',
        'dashscope', true, 32768, 0.8, 2.0,
        '角色扮演模型；需配置百炼 workspace 专属兼容接口地址。', NOW(), NOW()
    ),
    (
        gen_random_uuid(), 'qwen-flash-character', 'Qwen Flash Character',
        'dashscope', true, 8192, 0.25, 1.5,
        '低延迟角色扮演模型；需配置百炼 workspace 专属兼容接口地址。', NOW(), NOW()
    ),
    (
        gen_random_uuid(), 'qwen-flash-character-2026-02-26',
        'Qwen Flash Character 2026-02-26', 'dashscope', true, 262144, 0.18, 1.5,
        '长上下文角色扮演快照；需配置百炼 workspace 专属兼容接口地址。', NOW(), NOW()
    ),
    (
        gen_random_uuid(), 'M2-her', 'MiniMax M2-her',
        'minimax', true, 65536, NULL, NULL,
        'MiniMax 官方角色扮演模型；费用以开放平台控制台为准。', NOW(), NOW()
    ),
    (
        gen_random_uuid(), 'ERNIE-Character-8K', 'ERNIE Character 8K（已退役）',
        'qianfan', false, 8192, NULL, NULL,
        '百度预置服务已于 2026-06-09 退役，仅留作历史记录；请注册账号内仍可用的自定义接入点。', NOW(), NOW()
    )
ON CONFLICT ("provider", "identifier") DO NOTHING;
