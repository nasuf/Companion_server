-- game_sessions was created with CREATE TABLE IF NOT EXISTS, so a database
-- whose table already existed never received the foreign keys in
-- 20260531100000_sud_game_sessions. Agent deletion therefore left play
-- history behind. Those rows still have agent_id set, but the agent row is
-- gone, so they cannot be attached to a later friend.
--
-- Drop sessions whose agent or user no longer exists. Events follow the
-- session. Point wallets and game_point_ledger stay: they are account-scoped
-- and have no foreign key to a session.
-- Surviving rows may still point at a workspace or conversation that was
-- removed with the old friend. Those two keys are ON DELETE SET NULL, so
-- clear the dangling pointer instead of deleting the round.

DELETE FROM "game_events" AS event
USING "game_sessions" AS session
WHERE event."session_id" = session."id"
  AND (
    NOT EXISTS (
        SELECT 1 FROM "ai_agents" AS agent WHERE agent."id" = session."agent_id"
    )
    OR NOT EXISTS (
        SELECT 1 FROM "users" AS owner WHERE owner."id" = session."user_id"
    )
  );

DELETE FROM "game_sessions" AS session
WHERE NOT EXISTS (
    SELECT 1 FROM "ai_agents" AS agent WHERE agent."id" = session."agent_id"
)
OR NOT EXISTS (
    SELECT 1 FROM "users" AS owner WHERE owner."id" = session."user_id"
);

UPDATE "game_sessions" AS session
SET "workspace_id" = NULL
WHERE session."workspace_id" IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM "chat_workspaces" AS workspace
      WHERE workspace."id" = session."workspace_id"
  );

UPDATE "game_sessions" AS session
SET "conversation_id" = NULL
WHERE session."conversation_id" IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM "conversations" AS conversation
      WHERE conversation."id" = session."conversation_id"
  );

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'game_sessions_user_id_fkey'
    ) THEN
        ALTER TABLE "game_sessions"
            ADD CONSTRAINT "game_sessions_user_id_fkey"
            FOREIGN KEY ("user_id") REFERENCES "users"("id")
            ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'game_sessions_agent_id_fkey'
    ) THEN
        ALTER TABLE "game_sessions"
            ADD CONSTRAINT "game_sessions_agent_id_fkey"
            FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id")
            ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'game_sessions_workspace_id_fkey'
    ) THEN
        ALTER TABLE "game_sessions"
            ADD CONSTRAINT "game_sessions_workspace_id_fkey"
            FOREIGN KEY ("workspace_id") REFERENCES "chat_workspaces"("id")
            ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'game_sessions_conversation_id_fkey'
    ) THEN
        ALTER TABLE "game_sessions"
            ADD CONSTRAINT "game_sessions_conversation_id_fkey"
            FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id")
            ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;
