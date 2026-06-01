-- Time capsules are user-owned, not agent-owned. Keep the agent as optional
-- creation context only, and preserve capsules when an agent is deleted.
ALTER TABLE "time_capsules"
    DROP CONSTRAINT IF EXISTS "time_capsules_agent_id_fkey";

ALTER TABLE "time_capsules"
    ALTER COLUMN "agent_id" DROP NOT NULL;

ALTER TABLE "time_capsules"
    ADD CONSTRAINT "time_capsules_agent_id_fkey"
    FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id")
    ON DELETE SET NULL ON UPDATE CASCADE;

CREATE INDEX IF NOT EXISTS "time_capsules_user_state_idx"
    ON "time_capsules"("user_id", "status", "open_date");

CREATE INDEX IF NOT EXISTS "time_capsules_user_opened_idx"
    ON "time_capsules"("user_id", "opened_at");
