-- P3 memory repair queue: operational fallback for memory issues that cannot
-- be safely resolved by the hot-path contradiction flow.
CREATE TABLE "memory_repair_items" (
    "id" TEXT NOT NULL,
    "source_type" TEXT NOT NULL,
    "source_id" TEXT,
    "status" TEXT NOT NULL DEFAULT 'open',
    "severity" TEXT NOT NULL DEFAULT 'medium',
    "user_id" TEXT,
    "agent_id" TEXT,
    "workspace_id" TEXT,
    "conversation_id" TEXT,
    "message_id" TEXT,
    "memory_id" TEXT,
    "memory_source" TEXT,
    "reason" TEXT,
    "suggested_action" TEXT,
    "evidence" JSONB,
    "resolution_note" TEXT,
    "resolved_by_id" TEXT,
    "resolved_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "memory_repair_items_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "memory_repair_items_status_created_at_idx"
    ON "memory_repair_items"("status", "created_at" DESC);
CREATE INDEX "memory_repair_items_source_type_source_id_idx"
    ON "memory_repair_items"("source_type", "source_id");
CREATE INDEX "memory_repair_items_memory_id_status_idx"
    ON "memory_repair_items"("memory_id", "status");
CREATE INDEX "memory_repair_items_user_id_status_idx"
    ON "memory_repair_items"("user_id", "status");

CREATE UNIQUE INDEX "memory_repair_items_open_source_uidx"
    ON "memory_repair_items"("source_type", "source_id")
    WHERE "status" = 'open' AND "source_id" IS NOT NULL;
