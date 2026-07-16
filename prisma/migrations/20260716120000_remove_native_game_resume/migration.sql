-- Native games no longer support resuming an unfinished local board. Remove
-- obsolete open sessions and the in-game companion lines stored on old events.
DELETE FROM "game_sessions"
WHERE "provider" = 'native'
  AND "status" IN ('created', 'playing');

UPDATE "game_events" AS event
SET "companion_reply" = NULL
FROM "game_sessions" AS session
WHERE event."session_id" = session."id"
  AND session."provider" = 'native'
  AND event."event_type" NOT IN ('game_finished', 'game_aborted');
