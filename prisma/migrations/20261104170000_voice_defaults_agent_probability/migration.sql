-- Admin-chosen default voice per gender replaces the display names that were
-- hardcoded in voices.py. New agents of that gender bind to it, and realtime
-- calls fall back to it when an agent voice is rejected.
ALTER TABLE "tts_voice_profiles"
    ADD COLUMN IF NOT EXISTS "is_default" BOOLEAN NOT NULL DEFAULT false;
CREATE UNIQUE INDEX IF NOT EXISTS "tts_voice_profiles_one_default_per_gender"
    ON "tts_voice_profiles"("provider", "model", "gender")
    WHERE "is_default";
UPDATE "tts_voice_profiles" profile
SET "is_default" = true, "updated_at" = NOW()
WHERE profile."provider" = 'stepfun'
  AND profile."model" = 'stepaudio-2.5-tts'
  AND profile."enabled"
  AND profile."voice_id" IN ('elegantgentle-female', 'wenrounansheng')
  AND NOT EXISTS (
      SELECT 1 FROM "tts_voice_profiles" other
      WHERE other."provider" = profile."provider"
        AND other."model" = profile."model"
        AND other."gender" = profile."gender"
        AND other."is_default"
  );

-- NULL inherits system_config.tts_output_probability.
ALTER TABLE "ai_agents"
    ADD COLUMN IF NOT EXISTS "tts_output_probability" INTEGER;
ALTER TABLE "ai_agents" DROP CONSTRAINT IF EXISTS "ai_agents_tts_output_probability_check";
ALTER TABLE "ai_agents" ADD CONSTRAINT "ai_agents_tts_output_probability_check"
    CHECK ("tts_output_probability" BETWEEN 0 AND 100);

-- Library auditions are billed to the admin without an agent.
ALTER TABLE "tts_usage" ALTER COLUMN "agent_id" DROP NOT NULL;
