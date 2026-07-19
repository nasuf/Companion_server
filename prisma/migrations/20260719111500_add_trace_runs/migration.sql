-- Local trace collection table (self-hosted replacement for LangSmith run storage).
-- One row per langchain run; root run maintained by LocalTracer enter/close.
CREATE TABLE "trace_runs" (
    "run_id" TEXT NOT NULL,
    "trace_id" TEXT NOT NULL,
    "parent_id" TEXT,
    "name" TEXT NOT NULL,
    "run_type" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'running',
    "error" TEXT,
    "started_at" TIMESTAMP(3) NOT NULL,
    "ended_at" TIMESTAMP(3),
    "first_token_at" TIMESTAMP(3),
    "model_name" TEXT,
    "prompt_tokens" INTEGER,
    "completion_tokens" INTEGER,
    "total_tokens" INTEGER,
    "prompt_token_details" JSONB,
    "inputs_json" JSONB,
    "outputs_json" JSONB,
    "extra_json" JSONB,
    "events_json" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "trace_runs_pkey" PRIMARY KEY ("run_id")
);

CREATE INDEX "trace_runs_trace_id_idx" ON "trace_runs"("trace_id");

CREATE INDEX "trace_runs_created_at_idx" ON "trace_runs"("created_at");
