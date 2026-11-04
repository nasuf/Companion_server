BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
-- Realtime voice call model, chosen in 系统设置 like the other media models.
-- NULL keeps STEPFUN_REALTIME_MODEL. SQL-only: absent from the Prisma client.
ALTER TABLE system_config ADD COLUMN realtime_model TEXT;
-- StepFun list price (CNY per million tokens): input 13, cached input 3,
-- output 90. Calls are metered from response.done usage into llm_usage
-- (scope realtime_voice) and priced from this row.
INSERT INTO model_registry (
  id, identifier, display_name, provider, enabled, model_kind, billing_unit,
  input_cost_per_million, cached_input_cost_per_million, output_cost_per_million,
  notes, created_at, updated_at
) VALUES (
  gen_random_uuid(), 'stepaudio-2.5-realtime', 'StepAudio 2.5 Realtime', 'stepfun', true,
  'realtime', 'per_million_tokens', 13, 3, 90,
  '实时语音通话；按 response.done 返回的 token 计费。', NOW(), NOW()
)
ON CONFLICT (provider, identifier) DO NOTHING;
COMMIT;
