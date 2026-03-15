-- Phase 0: Add new tables for PRD 2-7
-- Run this migration against the Supabase database
-- Note: public schema uses TEXT for IDs (not UUID)

-- 0.7: User Portrait
CREATE TABLE IF NOT EXISTS user_portraits (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL REFERENCES users(id),
    agent_id TEXT NOT NULL,
    version INT NOT NULL DEFAULT 1,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 0.8: Memory Changelog
CREATE TABLE IF NOT EXISTS memory_changelogs (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL REFERENCES users(id),
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    operation VARCHAR(20) NOT NULL, -- 'insert', 'update', 'delete'
    old_value TEXT,
    new_value TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 0.9: Intimacy
CREATE TABLE IF NOT EXISTS intimacies (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    agent_id TEXT NOT NULL REFERENCES ai_agents(id),
    user_id TEXT NOT NULL REFERENCES users(id),
    topic_intimacy INT NOT NULL DEFAULT 50,
    topic_level VARCHAR(10) NOT NULL DEFAULT 'L3',
    topic_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    growth_intimacy INT NOT NULL DEFAULT 500,
    growth_level VARCHAR(10) NOT NULL DEFAULT 'G5',
    growth_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(agent_id, user_id)
);

-- 0.10: AI Daily Schedule
CREATE TABLE IF NOT EXISTS ai_daily_schedules (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    agent_id TEXT NOT NULL REFERENCES ai_agents(id),
    date DATE NOT NULL,
    schedule_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(agent_id, date)
);

-- 0.11: Schedule Adjust Log
CREATE TABLE IF NOT EXISTS schedule_adjust_logs (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    agent_id TEXT NOT NULL,
    adjust_type VARCHAR(50) NOT NULL,
    old_value TEXT,
    new_value TEXT,
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 0.12: Proactive Chat Log
CREATE TABLE IF NOT EXISTS proactive_chat_logs (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    agent_id TEXT NOT NULL REFERENCES ai_agents(id),
    user_id TEXT NOT NULL,
    message TEXT NOT NULL,
    event_type VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 0.13: Extend AiAgent with life overview fields
ALTER TABLE ai_agents
    ADD COLUMN IF NOT EXISTS life_overview TEXT,
    ADD COLUMN IF NOT EXISTS life_overview_data JSONB,
    ADD COLUMN IF NOT EXISTS age INT,
    ADD COLUMN IF NOT EXISTS occupation VARCHAR(100),
    ADD COLUMN IF NOT EXISTS city VARCHAR(100);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_user_portraits_user_id ON user_portraits(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_changelogs_user_id ON memory_changelogs(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_changelogs_created_at ON memory_changelogs(created_at);
CREATE INDEX IF NOT EXISTS idx_intimacies_agent_user ON intimacies(agent_id, user_id);
CREATE INDEX IF NOT EXISTS idx_ai_daily_schedules_agent_date ON ai_daily_schedules(agent_id, date);
CREATE INDEX IF NOT EXISTS idx_proactive_chat_logs_agent ON proactive_chat_logs(agent_id, created_at);
