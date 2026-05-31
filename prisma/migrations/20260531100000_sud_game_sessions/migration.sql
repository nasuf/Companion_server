CREATE TABLE IF NOT EXISTS "game_sessions" (
    "id" TEXT NOT NULL,
    "provider" TEXT NOT NULL DEFAULT 'sud',
    "status" TEXT NOT NULL DEFAULT 'created',
    "user_id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "workspace_id" TEXT,
    "conversation_id" TEXT,
    "mg_id" TEXT NOT NULL,
    "room_id" TEXT NOT NULL,
    "play_mode" TEXT NOT NULL DEFAULT 'versus',
    "difficulty" TEXT NOT NULL DEFAULT 'newbie',
    "ai_level" INTEGER NOT NULL DEFAULT 1,
    "sdk_enabled" BOOLEAN NOT NULL DEFAULT false,
    "sud_code" TEXT NOT NULL,
    "sud_code_expires_at" TIMESTAMP(3) NOT NULL,
    "user_player" JSONB NOT NULL,
    "ai_player" JSONB NOT NULL,
    "companion_reply" TEXT,
    "result" JSONB,
    "duration_seconds" INTEGER,
    "started_at" TIMESTAMP(3),
    "ended_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "game_sessions_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "game_sessions_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "game_sessions_agent_id_fkey" FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "game_sessions_workspace_id_fkey" FOREIGN KEY ("workspace_id") REFERENCES "chat_workspaces"("id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "game_sessions_conversation_id_fkey" FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id") ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS "game_events" (
    "id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "event_type" TEXT NOT NULL,
    "state" TEXT,
    "source" TEXT NOT NULL DEFAULT 'client',
    "payload" JSONB NOT NULL DEFAULT '{}',
    "companion_reply" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "game_events_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "game_events_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "game_sessions"("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX IF NOT EXISTS "game_sessions_user_created_idx" ON "game_sessions"("user_id", "created_at" DESC);
CREATE INDEX IF NOT EXISTS "game_sessions_agent_created_idx" ON "game_sessions"("agent_id", "created_at" DESC);
CREATE INDEX IF NOT EXISTS "game_sessions_conversation_created_idx" ON "game_sessions"("conversation_id", "created_at" DESC);
CREATE INDEX IF NOT EXISTS "game_sessions_room_idx" ON "game_sessions"("room_id");
CREATE INDEX IF NOT EXISTS "game_events_session_created_idx" ON "game_events"("session_id", "created_at" DESC);
CREATE INDEX IF NOT EXISTS "game_events_type_created_idx" ON "game_events"("event_type", "created_at" DESC);
