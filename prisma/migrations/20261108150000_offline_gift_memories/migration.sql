-- AI 寄给用户的礼物的状态记忆链接表 (B01.08)。与活动记忆对称 (offline_activity_memories)。
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

CREATE TABLE offline_gift_memories (
    gift_id        TEXT PRIMARY KEY REFERENCES real_world_gifts(id) ON DELETE CASCADE,
    user_id        TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id   TEXT REFERENCES chat_workspaces(id) ON DELETE CASCADE,
    user_memory_id TEXT,
    ai_memory_id   TEXT,
    state_key      TEXT NOT NULL DEFAULT '',
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX offline_gift_memories_user_idx ON offline_gift_memories(user_id);

COMMIT;
