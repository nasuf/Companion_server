BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
-- Voice messages move from DashScope Qwen Audio to StepFun stepaudio-2.5-tts so
-- they share one voice with realtime calls. DashScope rows stay for history
-- and usage records but leave every selectable pool.
INSERT INTO model_registry (
  id, identifier, display_name, provider, enabled, model_kind, billing_unit,
  unit_price_cny, notes, created_at, updated_at
) VALUES (
  gen_random_uuid(), 'stepaudio-2.5-tts', 'StepAudio 2.5 TTS', 'stepfun', true,
  'tts', 'per_10k_characters', 5.8,
  '580 元/百万字符（汉字 1 个，其他字符 2 个计 1 个）；复刻音色同时用于实时通话。',
  NOW(), NOW()
)
ON CONFLICT (provider, identifier) DO UPDATE SET enabled = true, updated_at = NOW();
UPDATE model_registry SET enabled = false, updated_at = NOW()
 WHERE model_kind = 'tts' AND provider = 'dashscope';
UPDATE system_config SET tts_model = 'stepaudio-2.5-tts'
 WHERE tts_model IS NULL OR tts_model <> 'stepaudio-2.5-tts';

UPDATE tts_voice_profiles SET enabled = false, updated_at = NOW()
 WHERE provider = 'dashscope';
-- Official voices accepted by both stepaudio-2.5-tts and stepaudio-2.5-realtime.
-- They stand in until the 女声-2 / 男声-2 clones are recorded again.
INSERT INTO tts_voice_profiles (
  id, display_name, provider, model, voice_id, gender, source, enabled, updated_at
) VALUES
  (gen_random_uuid(), '气质温婉', 'stepfun', 'stepaudio-2.5-tts', 'elegantgentle-female', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '温柔男声', 'stepfun', 'stepaudio-2.5-tts', 'wenrounansheng', 'male', 'system', true, NOW())
ON CONFLICT (provider, model, voice_id) DO UPDATE SET enabled = true, updated_at = NOW();

-- DashScope ids are rejected by StepFun; bind every agent to the official
-- voice of its gender (same rule as speech_output.voices.assign_random_voice).
UPDATE ai_agents SET
  tts_voice_id = CASE WHEN lower(coalesce(gender, '')) = 'male'
                      THEN 'wenrounansheng' ELSE 'elegantgentle-female' END,
  updated_at = NOW()
 WHERE tts_voice_id IS NULL OR tts_voice_id NOT IN (
   SELECT voice_id FROM tts_voice_profiles
    WHERE provider = 'stepfun' AND model = 'stepaudio-2.5-tts'
 );
COMMIT;
