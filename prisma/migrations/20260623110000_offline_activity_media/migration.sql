CREATE TABLE IF NOT EXISTS offline_activity_media (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    recommendation_id TEXT NOT NULL REFERENCES offline_activity_recommendations(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind TEXT NOT NULL DEFAULT 'image',
    name TEXT,
    mime TEXT NOT NULL,
    size INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    storage_key TEXT NOT NULL UNIQUE,
    url TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS offline_activity_media_recommendation_idx
    ON offline_activity_media(recommendation_id, created_at DESC);

CREATE INDEX IF NOT EXISTS offline_activity_media_user_idx
    ON offline_activity_media(user_id, created_at DESC);
