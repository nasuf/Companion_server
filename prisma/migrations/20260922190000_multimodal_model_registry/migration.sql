-- Register multimodal models so runtime routing can use provider/model dropdowns.

INSERT INTO "model_registry" (
    "id",
    "identifier",
    "display_name",
    "provider",
    "enabled",
    "model_kind",
    "billing_unit",
    "context_window",
    "input_cost_per_million",
    "output_cost_per_million",
    "notes",
    "updated_at"
) VALUES
(
    gen_random_uuid(),
    'doubao-1-5-vision-pro-32k-250115',
    'Doubao 1.5 Vision Pro 32K（即将下线）',
    'ark',
    false,
    'vision',
    'per_million_tokens',
    32768,
    3,
    9,
    '旧版视觉理解模型；仅保留历史配置识别，不再允许新选择。',
    CURRENT_TIMESTAMP
),
(
    gen_random_uuid(),
    'doubao-1-5-vision-pro-250328',
    'Doubao 1.5 Vision Pro',
    'ark',
    true,
    'vision',
    'per_million_tokens',
    131072,
    3,
    9,
    '视觉理解、OCR 和场景摘要；用于聊天图片分析。',
    CURRENT_TIMESTAMP
),
(
    gen_random_uuid(),
    'fun-asr-flash-2026-06-15',
    'Fun-ASR Flash',
    'dashscope',
    true,
    'asr',
    'per_million_tokens',
    NULL,
    NULL,
    NULL,
    '实时语音转写；使用 DashScope 多模态生成端点。',
    CURRENT_TIMESTAMP
)
ON CONFLICT ("provider", "identifier") DO UPDATE SET
    "display_name" = EXCLUDED."display_name",
    "enabled" = EXCLUDED."enabled",
    "model_kind" = EXCLUDED."model_kind",
    "billing_unit" = EXCLUDED."billing_unit",
    "context_window" = EXCLUDED."context_window",
    "input_cost_per_million" = EXCLUDED."input_cost_per_million",
    "output_cost_per_million" = EXCLUDED."output_cost_per_million",
    "notes" = EXCLUDED."notes",
    "updated_at" = CURRENT_TIMESTAMP;

UPDATE "system_config"
SET "vision_model" = 'doubao-1-5-vision-pro-250328',
    "updated_at" = CURRENT_TIMESTAMP
WHERE "vision_model" = 'doubao-1-5-vision-pro-32k-250115';
