-- Consecutive interaction days (user messages + makeup cards) per workspace.

CREATE TABLE IF NOT EXISTS workspace_interaction_days (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    workspace_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    local_date DATE NOT NULL,
    source TEXT NOT NULL,
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT workspace_interaction_days_workspace_id_fkey
        FOREIGN KEY (workspace_id) REFERENCES chat_workspaces(id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT workspace_interaction_days_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT workspace_interaction_days_source_check
        CHECK (source IN ('user_message', 'makeup'))
);

CREATE UNIQUE INDEX IF NOT EXISTS workspace_interaction_days_ws_date_key
    ON workspace_interaction_days (workspace_id, local_date);

CREATE INDEX IF NOT EXISTS workspace_interaction_days_ws_date_idx
    ON workspace_interaction_days (workspace_id, local_date DESC);

-- Seed from existing user messages. created_at is TIMESTAMP WITHOUT TIME ZONE
-- stored as UTC wall clock (Prisma default); interpret as UTC then fold to
-- Asia/Shanghai so the ledger matches local_activity_date().
INSERT INTO workspace_interaction_days (id, workspace_id, user_id, local_date, source, created_at)
SELECT gen_random_uuid()::text,
       c.workspace_id,
       c.user_id,
       ((m.created_at AT TIME ZONE 'UTC') AT TIME ZONE 'Asia/Shanghai')::date,
       'user_message',
       MIN(m.created_at)
FROM messages m
JOIN conversations c ON c.id = m.conversation_id
WHERE m.role = 'user'
  AND c.workspace_id IS NOT NULL
GROUP BY c.workspace_id, c.user_id,
         ((m.created_at AT TIME ZONE 'UTC') AT TIME ZONE 'Asia/Shanghai')::date
ON CONFLICT (workspace_id, local_date) DO NOTHING;
