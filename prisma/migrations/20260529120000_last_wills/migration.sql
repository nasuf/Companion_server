ALTER TABLE "users"
    ADD COLUMN IF NOT EXISTS "last_seen_at" TIMESTAMP(3);

CREATE TABLE IF NOT EXISTS "user_daily_activity" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "local_date" DATE NOT NULL,
    "source" TEXT NOT NULL,
    "seen_count" INTEGER NOT NULL DEFAULT 1,
    "first_seen_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "last_seen_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "user_daily_activity_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "user_daily_activity_user_date_key"
    ON "user_daily_activity"("user_id", "local_date");

CREATE INDEX IF NOT EXISTS "user_daily_activity_user_date_idx"
    ON "user_daily_activity"("user_id", "local_date");

ALTER TABLE "user_daily_activity"
    ADD CONSTRAINT "user_daily_activity_user_id_fkey"
    FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

CREATE TABLE IF NOT EXISTS "last_wills" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "workspace_id" TEXT,
    "content" TEXT NOT NULL,
    "inactivity_days" INTEGER NOT NULL DEFAULT 30,
    "contacts" JSONB NOT NULL DEFAULT '[]'::jsonb,
    "status" TEXT NOT NULL DEFAULT 'draft',
    "started_at" TIMESTAMP(3),
    "triggered_at" TIMESTAMP(3),
    "delivered_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "last_wills_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "last_wills_user_agent_status_idx"
    ON "last_wills"("user_id", "agent_id", "status");

CREATE UNIQUE INDEX IF NOT EXISTS "last_wills_user_agent_unique"
    ON "last_wills"("user_id", "agent_id");

CREATE INDEX IF NOT EXISTS "last_wills_status_started_idx"
    ON "last_wills"("status", "started_at");

ALTER TABLE "last_wills"
    ADD CONSTRAINT "last_wills_user_id_fkey"
    FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "last_wills"
    ADD CONSTRAINT "last_wills_agent_id_fkey"
    FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "last_wills"
    ADD CONSTRAINT "last_wills_workspace_id_fkey"
    FOREIGN KEY ("workspace_id") REFERENCES "chat_workspaces"("id") ON DELETE SET NULL ON UPDATE CASCADE;

CREATE TABLE IF NOT EXISTS "last_will_deliveries" (
    "id" TEXT NOT NULL,
    "last_will_id" TEXT NOT NULL,
    "channel" TEXT NOT NULL,
    "contact" JSONB NOT NULL,
    "dedupe_key" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "last_will_deliveries_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "last_will_deliveries_unique_contact_channel"
    ON "last_will_deliveries"("last_will_id", "channel", "dedupe_key");

CREATE INDEX IF NOT EXISTS "last_will_deliveries_status_idx"
    ON "last_will_deliveries"("status", "created_at");

ALTER TABLE "last_will_deliveries"
    ADD CONSTRAINT "last_will_deliveries_last_will_id_fkey"
    FOREIGN KEY ("last_will_id") REFERENCES "last_wills"("id") ON DELETE CASCADE ON UPDATE CASCADE;
