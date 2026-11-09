BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- Female agents still on Serena (the 2026-11-06 migration's default) move to
-- Mia (温柔生活博主-舒然), the admin-chosen default since 2026-11-07. In
-- replays of a 2026-11-09 call Mia spoke lower and steadier than Serena.
-- Agents an admin bound to any other voice keep it. Only runs if Mia is an
-- enabled female voice of the call model, so a retired id is never written.
UPDATE ai_agents
SET tts_voice_id = 'Mia', updated_at = NOW()
WHERE tts_voice_id = 'Serena'
  AND lower(coalesce(gender, '')) = 'female'
  AND EXISTS (
    SELECT 1 FROM tts_voice_profiles
    WHERE provider = 'dashscope' AND model = 'qwen3.8-omni-flash-realtime'
      AND voice_id = 'Mia' AND gender = 'female' AND enabled
  );

COMMIT;
