CREATE TABLE ops_redis_capacity (
    id TEXT PRIMARY KEY CHECK (id = 'primary'),
    sampled_at TIMESTAMPTZ,
    snapshot JSONB NOT NULL DEFAULT '{}',
    level TEXT NOT NULL DEFAULT 'normal' CHECK (level IN ('normal','warning','unknown')),
    notified_signature TEXT,
    generation INTEGER NOT NULL DEFAULT 0,
    notification_ids JSONB NOT NULL DEFAULT '[]',
    recipient_count INTEGER NOT NULL DEFAULT 0
);
