-- Queryable LLM runtime health metrics.
-- Values are session-level aggregates written with each llm_usage row.
ALTER TABLE "llm_usage"
    ADD COLUMN "latency_ms_total" INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN "latency_count" INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN "failure_count" INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN "fallback_count" INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN "circuit_open_count" INTEGER NOT NULL DEFAULT 0;
