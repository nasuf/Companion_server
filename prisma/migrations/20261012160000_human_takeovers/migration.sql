CREATE TABLE "human_takeovers" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "conversation_id" TEXT,
    "agent_id" TEXT,
    "workspace_id" TEXT,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "owner_admin_id" TEXT,
    "claimed_at" TIMESTAMP(3),
    "status_override" TEXT,
    "unread_count" INTEGER NOT NULL DEFAULT 0,
    "last_user_message_at" TIMESTAMP(3),
    "last_message_preview" TEXT,
    "admin_last_read_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "human_takeovers_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "human_takeovers_user_id_key" ON "human_takeovers"("user_id");
CREATE INDEX "human_takeovers_status_created_idx" ON "human_takeovers"("status", "created_at" DESC);
CREATE INDEX "human_takeovers_owner_status_idx" ON "human_takeovers"("owner_admin_id", "status");
CREATE INDEX "human_takeovers_conversation_idx" ON "human_takeovers"("conversation_id");

ALTER TABLE "human_takeovers" ADD CONSTRAINT "human_takeovers_user_id_fkey"
    FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "human_takeovers" ADD CONSTRAINT "human_takeovers_owner_admin_id_fkey"
    FOREIGN KEY ("owner_admin_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;

CREATE TABLE "human_takeover_dismissals" (
    "id" TEXT NOT NULL,
    "takeover_id" TEXT NOT NULL,
    "admin_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "human_takeover_dismissals_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "human_takeover_dismissals_takeover_admin_key"
    ON "human_takeover_dismissals"("takeover_id", "admin_id");

ALTER TABLE "human_takeover_dismissals" ADD CONSTRAINT "human_takeover_dismissals_takeover_id_fkey"
    FOREIGN KEY ("takeover_id") REFERENCES "human_takeovers"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "human_takeover_dismissals" ADD CONSTRAINT "human_takeover_dismissals_admin_id_fkey"
    FOREIGN KEY ("admin_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
