INSERT INTO "model_registry" (
    "id", "identifier", "display_name", "provider", "enabled",
    "context_window", "input_cost_per_million", "output_cost_per_million",
    "cached_input_cost_per_million", "notes", "created_at", "updated_at"
) VALUES (
    gen_random_uuid(),
    'doubao-seed-character-260628',
    'Doubao Seed Character 260628',
    'ark',
    true,
    131072,
    0.8,
    2.0,
    0.16,
    '豆包角色扮演模型；需先在火山方舟开通模型服务。官方按量起始价：输入 ¥0.8/百万 tokens、输出 ¥2/百万 tokens、缓存命中 ¥0.16/百万 tokens；缓存存储另按 ¥0.017/百万 tokens/小时计费。',
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
    "cached_input_cost_per_million" = COALESCE(
        "model_registry"."cached_input_cost_per_million",
        EXCLUDED."cached_input_cost_per_million"
    ),
    "notes" = CASE
        WHEN "model_registry"."notes" IS NULL
          OR "model_registry"."notes" = '豆包角色扮演模型；需先在火山方舟开通模型服务。价格为官方按量付费起始价，实际账单以控制台为准。'
        THEN EXCLUDED."notes"
        ELSE "model_registry"."notes"
    END,
    "updated_at" = NOW();
