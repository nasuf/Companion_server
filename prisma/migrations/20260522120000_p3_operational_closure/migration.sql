-- P3 operational closure:
-- 1. Materialized memory quality state.
-- 2. Automatic memory consolidation run audit.
-- 3. Visible memory use events.
-- 4. Structured crisis event counters.

CREATE TABLE IF NOT EXISTS "memory_quality_states" (
    "memory_id" TEXT NOT NULL,
    "memory_source" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "workspace_id" TEXT,
    "confidence" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "evidence_message_ids" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    "last_verified_at" TIMESTAMP(3),
    "verified_by" TEXT,
    "contradiction_state" TEXT NOT NULL DEFAULT 'none',
    "user_corrected_count" INTEGER NOT NULL DEFAULT 0,
    "admin_repaired_count" INTEGER NOT NULL DEFAULT 0,
    "access_count" INTEGER NOT NULL DEFAULT 0,
    "last_repair_item_id" TEXT,
    "superseded_by_memory_id" TEXT,
    "signals" JSONB NOT NULL DEFAULT '{}'::jsonb,
    "source_updated_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "memory_quality_states_pkey" PRIMARY KEY ("memory_id", "memory_source")
);

CREATE INDEX IF NOT EXISTS "memory_quality_states_user_workspace_idx"
    ON "memory_quality_states"("user_id", "workspace_id", "confidence");
CREATE INDEX IF NOT EXISTS "memory_quality_states_contradiction_idx"
    ON "memory_quality_states"("contradiction_state", "updated_at");
CREATE INDEX IF NOT EXISTS "memory_quality_states_superseded_idx"
    ON "memory_quality_states"("superseded_by_memory_id");

CREATE TABLE IF NOT EXISTS "memory_consolidation_runs" (
    "id" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'succeeded',
    "user_id" TEXT,
    "workspace_id" TEXT,
    "source" TEXT,
    "scopes" INTEGER NOT NULL DEFAULT 0,
    "checked" INTEGER NOT NULL DEFAULT 0,
    "archived" INTEGER NOT NULL DEFAULT 0,
    "merged" INTEGER NOT NULL DEFAULT 0,
    "updated" INTEGER NOT NULL DEFAULT 0,
    "errors" INTEGER NOT NULL DEFAULT 0,
    "changes" JSONB NOT NULL DEFAULT '[]'::jsonb,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "memory_consolidation_runs_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "memory_consolidation_runs_created_idx"
    ON "memory_consolidation_runs"("created_at" DESC);
CREATE INDEX IF NOT EXISTS "memory_consolidation_runs_scope_idx"
    ON "memory_consolidation_runs"("user_id", "workspace_id", "created_at" DESC);

CREATE TABLE IF NOT EXISTS "memory_visible_use_events" (
    "id" TEXT NOT NULL,
    "message_id" TEXT,
    "conversation_id" TEXT NOT NULL,
    "agent_id" TEXT,
    "user_id" TEXT,
    "workspace_id" TEXT,
    "trace_id" TEXT,
    "method" TEXT NOT NULL DEFAULT 'lexical_overlap_v1',
    "selected_count" INTEGER NOT NULL DEFAULT 0,
    "likely_used_count" INTEGER NOT NULL DEFAULT 0,
    "likely_unused_count" INTEGER NOT NULL DEFAULT 0,
    "visible_use_rate" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "unsupported_reference_count" INTEGER NOT NULL DEFAULT 0,
    "warning_count" INTEGER NOT NULL DEFAULT 0,
    "has_prompt_dilution" BOOLEAN NOT NULL DEFAULT false,
    "has_final_gate_drop" BOOLEAN NOT NULL DEFAULT false,
    "memory_ids" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    "payload" JSONB NOT NULL DEFAULT '{}'::jsonb,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "memory_visible_use_events_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "memory_visible_use_events_created_idx"
    ON "memory_visible_use_events"("created_at" DESC);
CREATE INDEX IF NOT EXISTS "memory_visible_use_events_scope_idx"
    ON "memory_visible_use_events"("agent_id", "user_id", "created_at" DESC);
CREATE INDEX IF NOT EXISTS "memory_visible_use_events_conversation_idx"
    ON "memory_visible_use_events"("conversation_id", "created_at" DESC);

CREATE TABLE IF NOT EXISTS "crisis_events" (
    "id" TEXT NOT NULL,
    "message_id" TEXT,
    "conversation_id" TEXT NOT NULL,
    "agent_id" TEXT,
    "user_id" TEXT,
    "workspace_id" TEXT,
    "trace_id" TEXT,
    "status" TEXT NOT NULL,
    "category" TEXT,
    "severity" TEXT,
    "handler_path" TEXT,
    "aftercare_triggered" BOOLEAN NOT NULL DEFAULT false,
    "safety_check_mode" TEXT,
    "semantic_checked" BOOLEAN NOT NULL DEFAULT false,
    "semantic_detected" BOOLEAN NOT NULL DEFAULT false,
    "payload" JSONB NOT NULL DEFAULT '{}'::jsonb,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "crisis_events_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "crisis_events_created_idx"
    ON "crisis_events"("created_at" DESC);
CREATE INDEX IF NOT EXISTS "crisis_events_scope_idx"
    ON "crisis_events"("agent_id", "user_id", "created_at" DESC);
CREATE INDEX IF NOT EXISTS "crisis_events_status_idx"
    ON "crisis_events"("status", "created_at" DESC);
