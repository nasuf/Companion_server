CREATE TABLE "memory_daily_reviews" (
  "id" TEXT PRIMARY KEY,
  "workspace_id" TEXT NOT NULL REFERENCES "chat_workspaces"("id") ON DELETE CASCADE,
  "local_date" DATE NOT NULL,
  "status" TEXT NOT NULL CHECK ("status" IN ('running','completed','failed','capacity_skipped')),
  "memory_ids" JSONB NOT NULL DEFAULT '[]'::jsonb,
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX "memory_daily_reviews_workspace_id_local_date_key"
  ON "memory_daily_reviews"("workspace_id","local_date");
CREATE INDEX "memory_daily_reviews_retention_idx" ON "memory_daily_reviews"("created_at");
