-- Ark web search plugin calls per turn. Billed per call with a monthly free
-- quota, so the cost is aggregated month-to-date in admin stats rather than
-- folded into llm_usage.cost_cny (which is token-based).
ALTER TABLE "llm_usage" ADD COLUMN IF NOT EXISTS "web_search_calls" INTEGER NOT NULL DEFAULT 0;

-- Month-to-date quota accounting scans by created_at and only cares about
-- rows that actually searched.
CREATE INDEX IF NOT EXISTS "llm_usage_web_search_idx"
  ON "llm_usage"("created_at" DESC)
  WHERE "web_search_calls" > 0;
