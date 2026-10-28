CREATE TABLE "ops_provider_health" (
  "id" TEXT PRIMARY KEY,
  "snapshot" JSONB NOT NULL DEFAULT '{}',
  "sampled_at" TIMESTAMPTZ
);
