ALTER TABLE "game_sessions"
    ADD COLUMN IF NOT EXISTS "game_key" TEXT,
    ALTER COLUMN "sud_code" DROP NOT NULL,
    ALTER COLUMN "sud_code_expires_at" DROP NOT NULL;

CREATE INDEX IF NOT EXISTS "game_sessions_provider_game_created_idx"
    ON "game_sessions"("provider", "game_key", "created_at" DESC);

ALTER TABLE "game_events"
    ADD COLUMN IF NOT EXISTS "client_event_id" TEXT;

WITH ranked_client_events AS (
    SELECT
        "id",
        "payload"->>'client_event_id' AS client_event_id,
        ROW_NUMBER() OVER (
            PARTITION BY "session_id", "payload"->>'client_event_id'
            ORDER BY "created_at", "id"
        ) AS duplicate_rank
    FROM "game_events"
    WHERE "payload"->>'client_event_id' IS NOT NULL
)
UPDATE "game_events" AS event
SET "client_event_id" = CASE
    WHEN ranked.duplicate_rank = 1 THEN ranked.client_event_id
    ELSE NULL
END
FROM ranked_client_events AS ranked
WHERE event."id" = ranked."id";

CREATE INDEX IF NOT EXISTS "game_events_session_client_event_idx"
    ON "game_events"("session_id", "client_event_id");

CREATE UNIQUE INDEX IF NOT EXISTS "game_events_session_client_event_unique_idx"
    ON "game_events"("session_id", "client_event_id")
    WHERE "client_event_id" IS NOT NULL;
