-- Cached-input pricing for prefix cache hits (e.g. DeepSeek bills cache-hit
-- input at ~1/40-1/120 of the miss price). NULL = not configured; cost
-- estimation conservatively falls back to the full input price.
ALTER TABLE model_registry
    ADD COLUMN IF NOT EXISTS cached_input_cost_per_million DOUBLE PRECISION;

-- Message list joins per-turn usage by trace_id (tokens/cache/cost next to the
-- Trace button on each AI reply).
CREATE INDEX IF NOT EXISTS "llm_usage_trace_id_idx" ON "llm_usage"("trace_id");
