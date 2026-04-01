CREATE TABLE IF NOT EXISTS "proactive_states" (
    "id" TEXT PRIMARY KEY,
    "workspace_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "conversation_id" TEXT,
    "status" TEXT NOT NULL DEFAULT 'idle',
    "stage" TEXT NOT NULL DEFAULT 'cold_start',
    "silence_level_n" INTEGER NOT NULL DEFAULT 0,
    "followup_plan_type" TEXT NOT NULL DEFAULT 'normal',
    "remaining_forced_triggers" INTEGER,
    "current_window_index" INTEGER,
    "window_due_at" TIMESTAMP(3),
    "response_deadline_at" TIMESTAMP(3),
    "t0_at" TIMESTAMP(3),
    "last_proactive_at" TIMESTAMP(3),
    "last_user_reply_at" TIMESTAMP(3),
    "last_assistant_reply_at" TIMESTAMP(3),
    "last_attempt_at" TIMESTAMP(3),
    "daily_scene_triggered_at" TIMESTAMP(3),
    "stop_reason" TEXT,
    "metadata" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "proactive_states_workspace_id_fkey"
        FOREIGN KEY ("workspace_id") REFERENCES "chat_workspaces"("id")
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "proactive_states_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users"("id")
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "proactive_states_agent_id_fkey"
        FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id")
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "proactive_states_conversation_id_fkey"
        FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id")
        ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS "proactive_states_workspace_id_key"
    ON "proactive_states"("workspace_id");
CREATE INDEX IF NOT EXISTS "proactive_states_user_id_status_idx"
    ON "proactive_states"("user_id", "status");
CREATE INDEX IF NOT EXISTS "proactive_states_agent_id_status_idx"
    ON "proactive_states"("agent_id", "status");
CREATE INDEX IF NOT EXISTS "proactive_states_status_window_due_at_idx"
    ON "proactive_states"("status", "window_due_at");

CREATE TABLE IF NOT EXISTS "proactive_event_logs" (
    "id" TEXT PRIMARY KEY,
    "state_id" TEXT NOT NULL,
    "workspace_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "conversation_id" TEXT,
    "event_type" TEXT NOT NULL,
    "window_index" INTEGER,
    "window_name" TEXT,
    "trigger_type" TEXT,
    "payload" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "proactive_event_logs_state_id_fkey"
        FOREIGN KEY ("state_id") REFERENCES "proactive_states"("id")
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "proactive_event_logs_workspace_id_fkey"
        FOREIGN KEY ("workspace_id") REFERENCES "chat_workspaces"("id")
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "proactive_event_logs_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users"("id")
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "proactive_event_logs_agent_id_fkey"
        FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id")
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "proactive_event_logs_conversation_id_fkey"
        FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id")
        ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE INDEX IF NOT EXISTS "proactive_event_logs_workspace_id_created_at_idx"
    ON "proactive_event_logs"("workspace_id", "created_at");
CREATE INDEX IF NOT EXISTS "proactive_event_logs_state_id_created_at_idx"
    ON "proactive_event_logs"("state_id", "created_at");
CREATE INDEX IF NOT EXISTS "proactive_event_logs_event_type_created_at_idx"
    ON "proactive_event_logs"("event_type", "created_at");

ALTER TABLE "proactive_chat_logs"
    ADD COLUMN IF NOT EXISTS "workspace_id" TEXT,
    ADD COLUMN IF NOT EXISTS "conversation_id" TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'proactive_chat_logs_workspace_id_fkey'
    ) THEN
        ALTER TABLE "proactive_chat_logs"
            ADD CONSTRAINT "proactive_chat_logs_workspace_id_fkey"
            FOREIGN KEY ("workspace_id") REFERENCES "chat_workspaces"("id")
            ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'proactive_chat_logs_conversation_id_fkey'
    ) THEN
        ALTER TABLE "proactive_chat_logs"
            ADD CONSTRAINT "proactive_chat_logs_conversation_id_fkey"
            FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id")
            ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS "idx_proactive_chat_logs_workspace"
    ON "proactive_chat_logs"("workspace_id", "created_at");
CREATE INDEX IF NOT EXISTS "idx_proactive_chat_logs_conversation"
    ON "proactive_chat_logs"("conversation_id", "created_at");
