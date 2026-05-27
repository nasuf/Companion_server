CREATE TABLE "time_capsules" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "workspace_id" TEXT,
    "title" TEXT,
    "content" TEXT NOT NULL,
    "media" JSONB,
    "skin" TEXT NOT NULL DEFAULT 'paper',
    "open_date" TIMESTAMP(3),
    "status" TEXT NOT NULL DEFAULT 'draft',
    "sealed_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "time_capsules_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "time_capsules_user_agent_state_idx"
    ON "time_capsules"("user_id", "agent_id", "status", "open_date");

CREATE INDEX "time_capsules_workspace_state_idx"
    ON "time_capsules"("workspace_id", "status", "open_date");

ALTER TABLE "time_capsules"
    ADD CONSTRAINT "time_capsules_user_id_fkey"
    FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "time_capsules"
    ADD CONSTRAINT "time_capsules_agent_id_fkey"
    FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "time_capsules"
    ADD CONSTRAINT "time_capsules_workspace_id_fkey"
    FOREIGN KEY ("workspace_id") REFERENCES "chat_workspaces"("id") ON DELETE SET NULL ON UPDATE CASCADE;
