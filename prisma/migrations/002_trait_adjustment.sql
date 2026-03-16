-- Add seven-dim personality traits to AiAgent
ALTER TABLE ai_agents ADD COLUMN IF NOT EXISTS seven_dim_traits JSONB;
ALTER TABLE ai_agents ADD COLUMN IF NOT EXISTS current_traits JSONB;
ALTER TABLE ai_agents ADD COLUMN IF NOT EXISTS traits_history JSONB;

-- Add mention_count to memories
ALTER TABLE memories ADD COLUMN IF NOT EXISTS mention_count INTEGER NOT NULL DEFAULT 0;

-- Create trait feedback log table
CREATE TABLE IF NOT EXISTS trait_feedback_logs (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    agent_id TEXT NOT NULL REFERENCES ai_agents(id),
    dimension TEXT NOT NULL,
    delta DOUBLE PRECISION NOT NULL,
    source TEXT NOT NULL,
    reason TEXT,
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trait_feedback_agent ON trait_feedback_logs(agent_id);
CREATE INDEX IF NOT EXISTS idx_trait_feedback_created ON trait_feedback_logs(created_at);
