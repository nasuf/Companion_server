-- Qwen assistant speech output: global runtime config, stable per-agent voice,
-- model-registry modality/pricing, and immutable usage ledger.

ALTER TABLE "ai_agents"
    ADD COLUMN IF NOT EXISTS "tts_voice_id" TEXT;

ALTER TABLE "system_config"
    ADD COLUMN IF NOT EXISTS "tts_model" TEXT,
    ADD COLUMN IF NOT EXISTS "tts_output_probability" INTEGER;

ALTER TABLE "system_config"
    DROP CONSTRAINT IF EXISTS "system_config_tts_output_probability_check";
ALTER TABLE "system_config"
    ADD CONSTRAINT "system_config_tts_output_probability_check"
    CHECK (
        "tts_output_probability" IS NULL
        OR "tts_output_probability" BETWEEN 0 AND 100
    );

ALTER TABLE "model_registry"
    ADD COLUMN IF NOT EXISTS "model_kind" TEXT NOT NULL DEFAULT 'llm',
    ADD COLUMN IF NOT EXISTS "billing_unit" TEXT NOT NULL DEFAULT 'per_million_tokens',
    ADD COLUMN IF NOT EXISTS "unit_price_cny" DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS "model_registry_model_kind_enabled_idx"
    ON "model_registry"("model_kind", "enabled");

INSERT INTO "model_registry" (
    "id",
    "identifier",
    "display_name",
    "provider",
    "enabled",
    "model_kind",
    "billing_unit",
    "unit_price_cny",
    "notes",
    "updated_at"
) VALUES (
    gen_random_uuid(),
    'qwen3-tts-instruct-flash-2026-01-26',
    'Qwen3 TTS Instruct Flash（语音输出）',
    'dashscope',
    true,
    'tts',
    'per_10k_characters',
    0.8,
    'Flutter Agent 语音气泡；系统音色 + 中文语气指令，HTTP 非实时输出。',
    CURRENT_TIMESTAMP
)
ON CONFLICT ("provider", "identifier") DO UPDATE SET
    "display_name" = EXCLUDED."display_name",
    "enabled" = true,
    "model_kind" = EXCLUDED."model_kind",
    "billing_unit" = EXCLUDED."billing_unit",
    "unit_price_cny" = EXCLUDED."unit_price_cny",
    "notes" = EXCLUDED."notes",
    "updated_at" = CURRENT_TIMESTAMP;

CREATE TABLE IF NOT EXISTS "tts_usage" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "message_id" TEXT,
    "source" TEXT NOT NULL,
    "provider" TEXT NOT NULL DEFAULT 'dashscope',
    "model" TEXT NOT NULL,
    "voice_id" TEXT NOT NULL,
    "request_id" TEXT,
    "raw_characters" INTEGER NOT NULL,
    "billable_characters" INTEGER NOT NULL,
    "duration_milliseconds" INTEGER NOT NULL,
    "audio_bytes" INTEGER NOT NULL,
    "unit_price_cny" DOUBLE PRECISION NOT NULL,
    "cost_cny" DOUBLE PRECISION NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "tts_usage_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "tts_usage_request_id_key"
    ON "tts_usage"("request_id");
CREATE INDEX IF NOT EXISTS "tts_usage_created_at_idx"
    ON "tts_usage"("created_at");
CREATE INDEX IF NOT EXISTS "tts_usage_user_id_created_at_idx"
    ON "tts_usage"("user_id", "created_at");
CREATE INDEX IF NOT EXISTS "tts_usage_agent_id_created_at_idx"
    ON "tts_usage"("agent_id", "created_at");
CREATE INDEX IF NOT EXISTS "tts_usage_conversation_id_created_at_idx"
    ON "tts_usage"("conversation_id", "created_at");
CREATE INDEX IF NOT EXISTS "tts_usage_message_id_idx"
    ON "tts_usage"("message_id");
