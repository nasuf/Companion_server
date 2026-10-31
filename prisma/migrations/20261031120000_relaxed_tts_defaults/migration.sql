-- New agents use the approved 0.9x conversational pace.
ALTER TABLE ai_agents ALTER COLUMN tts_rate SET DEFAULT 0.9;

-- Move only the complete old parameter preset. There is no per-field edit
-- history, so this deliberately preserves any distinguishable customization.
-- Keep the chosen voice (including clones), and leave style NULL/blank so it
-- continues to inherit the shared default rather than becoming an override.
UPDATE ai_agents
SET tts_rate = 0.9, updated_at = CURRENT_TIMESTAMP
WHERE tts_rate = 1.0
  AND tts_pitch = 1.0
  AND tts_volume = 50
  AND tts_seed = 0
  AND NULLIF(BTRIM(tts_instruction), '') IS NULL
  AND tts_auto_emotion = true
  AND tts_emotion_scale = 1.0;
