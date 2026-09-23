-- Chat image understanding uses Doubao-Seed-2.0-mini. The 1.5 vision ids
-- are unreachable on this account (404 / retiring), so drop them after the
-- global vision setting no longer points at them.
--
-- Stored rates are the 0-32K tier, which covers a chat image:
-- input 0.2, output 2, cache hit 0.04 CNY per million tokens.
-- Longer contexts are 0.4/4 (32-128K) and 0.8/8 (128-256K).

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
    "cached_input_cost_per_million",
    "notes",
    "updated_at"
) VALUES (
    gen_random_uuid(),
    'doubao-seed-2-0-mini-260428',
    'Doubao Seed 2.0 Mini',
    'ark',
    true,
    'vision',
    'per_million_tokens',
    262144,
    0.2,
    2,
    0.04,
    '聊天识图。按 0–32K 上下文计价：输入 0.2 元/百万 tokens，输出 2 元/百万 tokens，缓存命中 0.04。',
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
    "cached_input_cost_per_million" = EXCLUDED."cached_input_cost_per_million",
    "notes" = EXCLUDED."notes",
    "updated_at" = CURRENT_TIMESTAMP;

UPDATE "system_config"
SET "vision_model" = 'doubao-seed-2-0-mini-260428',
    "updated_at" = CURRENT_TIMESTAMP
WHERE "vision_model" IS NULL
   OR btrim("vision_model") = ''
   OR "vision_model" IN (
        'doubao-1-5-vision-pro-32k-250115',
        'doubao-1-5-vision-pro-250328',
        'doubao-1.5-vision-pro-250328'
   );

DELETE FROM "model_registry"
WHERE "provider" = 'ark'
  AND "model_kind" = 'vision'
  AND "identifier" IN (
        'doubao-1-5-vision-pro-32k-250115',
        'doubao-1-5-vision-pro-250328'
  );
