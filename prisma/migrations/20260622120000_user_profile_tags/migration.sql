CREATE TABLE IF NOT EXISTS user_profile_tags (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL REFERENCES ai_agents(id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES chat_workspaces(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'preference',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    source_memory_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'llm',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT user_profile_tags_confidence_check
        CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT user_profile_tags_evidence_count_check
        CHECK (evidence_count >= 0)
);

CREATE INDEX IF NOT EXISTS user_profile_tags_active_idx
    ON user_profile_tags(user_id, workspace_id, is_active, confidence DESC);

CREATE INDEX IF NOT EXISTS user_profile_tags_agent_idx
    ON user_profile_tags(agent_id, is_active, updated_at DESC);

CREATE INDEX IF NOT EXISTS user_profile_tags_label_idx
    ON user_profile_tags(label);
