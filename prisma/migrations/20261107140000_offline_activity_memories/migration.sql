BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- VOICE01.10: one user-side and one AI-side memory per activity the user responded to,
-- rewritten from the activity's current state on every status change (offline/activity_memory.py).
-- The row only records which memories belong to the activity; it goes with the activity.
CREATE TABLE offline_activity_memories (
    activity_id TEXT PRIMARY KEY REFERENCES offline_activity_recommendations(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES chat_workspaces(id) ON DELETE CASCADE,
    user_memory_id TEXT,
    ai_memory_id TEXT,
    state_key TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX offline_activity_memories_user_idx ON offline_activity_memories(user_id);

COMMIT;
