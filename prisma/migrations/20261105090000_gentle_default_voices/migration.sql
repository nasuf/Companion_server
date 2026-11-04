-- Product decision 2026-11-05: every agent speaks with 温柔女声
-- (wenrounvsheng) or 温柔男声 (wenrounansheng), and these are the gender
-- defaults for new agents. Admins can change defaults later in the library.
UPDATE "tts_voice_profiles"
SET "enabled" = true, "updated_at" = NOW()
WHERE "provider" = 'stepfun' AND "model" = 'stepaudio-2.5-tts'
  AND "voice_id" IN ('wenrounvsheng', 'wenrounansheng')
  AND NOT "enabled";

-- Clear first: the partial unique index allows one default per gender.
UPDATE "tts_voice_profiles"
SET "is_default" = false, "updated_at" = NOW()
WHERE "provider" = 'stepfun' AND "model" = 'stepaudio-2.5-tts'
  AND "is_default"
  AND "voice_id" NOT IN ('wenrounvsheng', 'wenrounansheng');
UPDATE "tts_voice_profiles"
SET "is_default" = true, "updated_at" = NOW()
WHERE "provider" = 'stepfun' AND "model" = 'stepaudio-2.5-tts'
  AND (("voice_id" = 'wenrounvsheng' AND "gender" = 'female')
    OR ("voice_id" = 'wenrounansheng' AND "gender" = 'male'))
  AND NOT "is_default";

UPDATE "ai_agents"
SET "tts_voice_id" = CASE WHEN lower(coalesce("gender", '')) = 'male'
                          THEN 'wenrounansheng' ELSE 'wenrounvsheng' END,
    "updated_at" = NOW()
WHERE "tts_voice_id" IS DISTINCT FROM
      CASE WHEN lower(coalesce("gender", '')) = 'male'
           THEN 'wenrounansheng' ELSE 'wenrounvsheng' END
  AND EXISTS (
      SELECT 1 FROM "tts_voice_profiles"
      WHERE "provider" = 'stepfun' AND "model" = 'stepaudio-2.5-tts'
        AND "voice_id" = 'wenrounvsheng'
  );
