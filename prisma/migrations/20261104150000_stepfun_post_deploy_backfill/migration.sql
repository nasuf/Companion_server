BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
-- The three 2026-11-04 migrations reached prod before the matching code, so
-- the old server kept writing for a while. Deploy stops it before migrating;
-- re-apply the same idempotent backfills to rows it wrote in between.
UPDATE messages SET hidden_from_timeline = true
 WHERE metadata->>'channel' = 'realtime_voice' AND NOT hidden_from_timeline;
UPDATE ai_agents SET
  tts_voice_id = CASE WHEN lower(coalesce(gender, '')) = 'male'
                      THEN 'wenrounansheng' ELSE 'elegantgentle-female' END,
  updated_at = NOW()
 WHERE tts_voice_id IS NULL OR tts_voice_id NOT IN (
   SELECT voice_id FROM tts_voice_profiles
    WHERE provider = 'stepfun' AND model = 'stepaudio-2.5-tts'
 );
COMMIT;
