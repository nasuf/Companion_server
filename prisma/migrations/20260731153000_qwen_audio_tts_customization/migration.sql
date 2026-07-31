ALTER TABLE "ai_agents"
    ADD COLUMN IF NOT EXISTS "tts_rate" DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    ADD COLUMN IF NOT EXISTS "tts_pitch" DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    ADD COLUMN IF NOT EXISTS "tts_volume" INTEGER NOT NULL DEFAULT 50,
    ADD COLUMN IF NOT EXISTS "tts_seed" INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS "tts_instruction" TEXT,
    ADD COLUMN IF NOT EXISTS "tts_auto_emotion" BOOLEAN NOT NULL DEFAULT true,
    ADD COLUMN IF NOT EXISTS "tts_emotion_scale" DOUBLE PRECISION NOT NULL DEFAULT 1.0;

ALTER TABLE "ai_agents" DROP CONSTRAINT IF EXISTS "ai_agents_tts_rate_check";
ALTER TABLE "ai_agents" ADD CONSTRAINT "ai_agents_tts_rate_check"
    CHECK ("tts_rate" BETWEEN 0.5 AND 2.0);
ALTER TABLE "ai_agents" DROP CONSTRAINT IF EXISTS "ai_agents_tts_pitch_check";
ALTER TABLE "ai_agents" ADD CONSTRAINT "ai_agents_tts_pitch_check"
    CHECK ("tts_pitch" BETWEEN 0.5 AND 2.0);
ALTER TABLE "ai_agents" DROP CONSTRAINT IF EXISTS "ai_agents_tts_volume_check";
ALTER TABLE "ai_agents" ADD CONSTRAINT "ai_agents_tts_volume_check"
    CHECK ("tts_volume" BETWEEN 0 AND 100);
ALTER TABLE "ai_agents" DROP CONSTRAINT IF EXISTS "ai_agents_tts_seed_check";
ALTER TABLE "ai_agents" ADD CONSTRAINT "ai_agents_tts_seed_check"
    CHECK ("tts_seed" BETWEEN 0 AND 65535);
ALTER TABLE "ai_agents" DROP CONSTRAINT IF EXISTS "ai_agents_tts_emotion_scale_check";
ALTER TABLE "ai_agents" ADD CONSTRAINT "ai_agents_tts_emotion_scale_check"
    CHECK ("tts_emotion_scale" BETWEEN 0 AND 2.0);

ALTER TABLE "tts_usage"
    ALTER COLUMN "conversation_id" DROP NOT NULL;

CREATE TABLE IF NOT EXISTS "tts_voice_profiles" (
    "id" TEXT NOT NULL,
    "display_name" TEXT NOT NULL,
    "provider" TEXT NOT NULL DEFAULT 'dashscope',
    "model" TEXT NOT NULL,
    "voice_id" TEXT NOT NULL,
    "gender" TEXT NOT NULL,
    "source" TEXT NOT NULL,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "provider_request_id" TEXT,
    "consent_confirmed_at" TIMESTAMP(3),
    "consent_confirmed_by" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "tts_voice_profiles_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "tts_voice_profiles_gender_check" CHECK ("gender" IN ('female', 'male')),
    CONSTRAINT "tts_voice_profiles_source_check" CHECK ("source" IN ('system', 'cloned'))
);

CREATE UNIQUE INDEX IF NOT EXISTS "tts_voice_profiles_provider_model_voice_id_key"
    ON "tts_voice_profiles"("provider", "model", "voice_id");
CREATE INDEX IF NOT EXISTS "tts_voice_profiles_model_gender_enabled_idx"
    ON "tts_voice_profiles"("model", "gender", "enabled");

INSERT INTO "tts_voice_profiles" (
    "id", "display_name", "provider", "model", "voice_id",
    "gender", "source", "enabled", "updated_at"
)
VALUES
    (
        gen_random_uuid(), '龙安灵心', 'dashscope',
        'qwen-audio-3.0-tts-plus', 'longanlingxin',
        'female', 'system', true, CURRENT_TIMESTAMP
    ),
    (
        gen_random_uuid(), '龙安鲁风', 'dashscope',
        'qwen-audio-3.0-tts-plus', 'longanlufeng',
        'male', 'system', true, CURRENT_TIMESTAMP
    )
ON CONFLICT ("provider", "model", "voice_id") DO UPDATE SET
    "display_name" = EXCLUDED."display_name",
    "gender" = EXCLUDED."gender",
    "enabled" = true,
    "updated_at" = CURRENT_TIMESTAMP;

UPDATE "ai_agents"
SET "tts_voice_id" = CASE
    WHEN LOWER(COALESCE("gender", '')) = 'male' THEN 'longanlufeng'
    ELSE 'longanlingxin'
END
WHERE "tts_voice_id" IS NULL
   OR "tts_voice_id" NOT IN ('longanlingxin', 'longanlufeng');

UPDATE "system_config"
SET "tts_model" = 'qwen-audio-3.0-tts-plus',
    "updated_at" = CURRENT_TIMESTAMP
WHERE "id" = 1
  AND (
      "tts_model" IS NULL
      OR "tts_model" LIKE 'qwen3-tts%'
  );

UPDATE "model_registry"
SET "enabled" = false,
    "updated_at" = CURRENT_TIMESTAMP
WHERE "provider" = 'dashscope'
  AND "identifier" LIKE 'qwen3-tts%'
  AND "model_kind" = 'tts';

INSERT INTO "model_registry" (
    "id", "identifier", "display_name", "provider", "enabled",
    "model_kind", "billing_unit", "unit_price_cny", "notes", "updated_at"
)
VALUES (
    gen_random_uuid(),
    'qwen-audio-3.0-tts-plus',
    'Qwen Audio 3.0 TTS Plus（语音输出）',
    'dashscope',
    true,
    'tts',
    'per_10k_characters',
    1.12413,
    '高质量 Agent 语音；支持声音复刻、指令控制和细粒度情绪标签。',
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
