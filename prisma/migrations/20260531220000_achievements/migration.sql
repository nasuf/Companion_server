CREATE TABLE IF NOT EXISTS achievement_unlocks (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL REFERENCES ai_agents(id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES chat_workspaces(id) ON DELETE SET NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    achievement_id INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    notified_at TIMESTAMP(3),
    unlocked_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS achievement_unlocks_user_agent_achievement_key
    ON achievement_unlocks(user_id, agent_id, achievement_id);

CREATE INDEX IF NOT EXISTS achievement_unlocks_user_agent_idx
    ON achievement_unlocks(user_id, agent_id, unlocked_at DESC);

CREATE TABLE IF NOT EXISTS achievement_events (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL REFERENCES ai_agents(id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES chat_workspaces(id) ON DELETE SET NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    source_id TEXT,
    value_int INTEGER,
    value_text TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS achievement_events_source_unique
    ON achievement_events(user_id, agent_id, event_type, source_id)
    WHERE source_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS achievement_events_scope_type_time_idx
    ON achievement_events(user_id, agent_id, event_type, occurred_at DESC);

CREATE INDEX IF NOT EXISTS achievement_events_workspace_type_time_idx
    ON achievement_events(workspace_id, event_type, occurred_at DESC);
