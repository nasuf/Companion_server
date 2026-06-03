CREATE TABLE IF NOT EXISTS "music_favorites" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "workspace_id" TEXT,
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
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "music_favorites_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "music_favorites_duration_nonnegative" CHECK ("duration_sec" >= 0),
    CONSTRAINT "music_favorites_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "music_favorites_agent_id_fkey" FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "music_favorites_workspace_id_fkey" FOREIGN KEY ("workspace_id") REFERENCES "chat_workspaces"("id") ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS "music_favorites_user_agent_track_key"
    ON "music_favorites"("user_id", "agent_id", "track_external_id");
CREATE INDEX IF NOT EXISTS "music_favorites_user_agent_created_idx"
    ON "music_favorites"("user_id", "agent_id", "created_at" DESC);
CREATE INDEX IF NOT EXISTS "music_favorites_workspace_created_idx"
    ON "music_favorites"("workspace_id", "created_at" DESC);

CREATE TABLE IF NOT EXISTS "music_playbacks" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "workspace_id" TEXT,
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
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "music_playbacks_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "music_playbacks_position_nonnegative" CHECK ("position_seconds" >= 0),
    CONSTRAINT "music_playbacks_duration_nonnegative" CHECK ("duration_sec" >= 0),
    CONSTRAINT "music_playbacks_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "music_playbacks_agent_id_fkey" FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "music_playbacks_workspace_id_fkey" FOREIGN KEY ("workspace_id") REFERENCES "chat_workspaces"("id") ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS "music_playbacks_user_agent_key"
    ON "music_playbacks"("user_id", "agent_id");
CREATE INDEX IF NOT EXISTS "music_playbacks_workspace_updated_idx"
    ON "music_playbacks"("workspace_id", "updated_at" DESC);
