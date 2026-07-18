-- Complete the M2-her metadata shown in Admin → System Settings → Model Library.
-- Preserve any prices that an admin already customized in the Web console.
INSERT INTO "model_registry" (
    "id", "identifier", "display_name", "provider", "enabled",
    "context_window", "input_cost_per_million", "output_cost_per_million",
    "cached_input_cost_per_million", "notes", "created_at", "updated_at"
) VALUES (
    gen_random_uuid(),
    'M2-her',
    'MiniMax M2-her',
    'minimax',
    true,
    65536,
    2.1,
    8.4,
    NULL,
    'MiniMax 官方角色扮演模型；64K 上下文，最大输出 2048 tokens。官方按量价：输入 ¥2.1/百万 tokens、输出 ¥8.4/百万 tokens，不支持提示缓存计费。',
    NOW(),
    NOW()
)
ON CONFLICT ("provider", "identifier") DO UPDATE SET
    "display_name" = COALESCE("model_registry"."display_name", EXCLUDED."display_name"),
    "context_window" = COALESCE("model_registry"."context_window", EXCLUDED."context_window"),
    "input_cost_per_million" = COALESCE(
        "model_registry"."input_cost_per_million",
        EXCLUDED."input_cost_per_million"
    ),
    "output_cost_per_million" = COALESCE(
        "model_registry"."output_cost_per_million",
        EXCLUDED."output_cost_per_million"
    ),
    "notes" = CASE
        WHEN "model_registry"."notes" IS NULL
          OR "model_registry"."notes" = 'MiniMax 官方角色扮演模型；费用以开放平台控制台为准。'
        THEN EXCLUDED."notes"
        ELSE "model_registry"."notes"
    END,
    "updated_at" = NOW();
