CREATE TABLE "music_co_listening_sessions" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "workspace_id" TEXT,
    "conversation_id" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'active',
    "initiated_by" TEXT NOT NULL DEFAULT 'user',
    "track_external_id" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "artist" TEXT NOT NULL DEFAULT 'AudioLib',
    "album" TEXT NOT NULL DEFAULT 'Curated Library',
    "library" TEXT NOT NULL DEFAULT 'audio.focus',
    "audio_url" TEXT NOT NULL DEFAULT '',
    "duration_sec" INTEGER NOT NULL DEFAULT 0,
    "cover_key" TEXT NOT NULL DEFAULT 'music-cover-01.jpg',
    "accent_a" TEXT NOT NULL DEFAULT '#1f6fff',
    "accent_b" TEXT NOT NULL DEFAULT '#18c6c0',
    "source" TEXT NOT NULL DEFAULT 'audiolib',
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "position_seconds" INTEGER NOT NULL DEFAULT 0,
    "is_playing" BOOLEAN NOT NULL DEFAULT true,
    "ended_reason" TEXT,
    "ended_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "music_co_listening_sessions_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "music_co_listening_sessions_conversation_id_key"
    ON "music_co_listening_sessions"("conversation_id");

CREATE INDEX "music_co_listening_user_agent_updated_idx"
    ON "music_co_listening_sessions"("user_id", "agent_id", "updated_at" DESC);

CREATE INDEX "music_co_listening_workspace_updated_idx"
    ON "music_co_listening_sessions"("workspace_id", "updated_at" DESC);

CREATE INDEX "music_co_listening_status_updated_idx"
    ON "music_co_listening_sessions"("status", "updated_at" DESC);

ALTER TABLE "music_co_listening_sessions"
    ADD CONSTRAINT "music_co_listening_sessions_user_id_fkey"
    FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "music_co_listening_sessions"
    ADD CONSTRAINT "music_co_listening_sessions_agent_id_fkey"
    FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "music_co_listening_sessions"
    ADD CONSTRAINT "music_co_listening_sessions_workspace_id_fkey"
    FOREIGN KEY ("workspace_id") REFERENCES "chat_workspaces"("id") ON DELETE SET NULL ON UPDATE CASCADE;

ALTER TABLE "music_co_listening_sessions"
    ADD CONSTRAINT "music_co_listening_sessions_conversation_id_fkey"
    FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id") ON DELETE CASCADE ON UPDATE CASCADE;
