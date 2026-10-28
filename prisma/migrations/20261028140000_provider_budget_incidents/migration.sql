CREATE TABLE "llm_provider_incidents" (
  "provider" TEXT NOT NULL,
  "model" TEXT NOT NULL,
  "code" TEXT NOT NULL,
  "first_failure_at" TIMESTAMPTZ NOT NULL,
  "last_failure_at" TIMESTAMPTZ NOT NULL,
  "last_success_started_at" TIMESTAMPTZ,
  PRIMARY KEY ("provider", "model")
);
