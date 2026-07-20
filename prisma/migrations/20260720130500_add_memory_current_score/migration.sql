-- Spec §1.5.2 L2 dynamic score gets its own column so `importance` can stay
-- the immutable initial score. The old implementation wrote the computed
-- current_score back into importance, compounding the time/frequency factors
-- nightly (upward inflation for frequently accessed rows, downward spiral for
-- idle ones). Retrieval ranking reads COALESCE(current_score, importance).
ALTER TABLE "memories_user" ADD COLUMN "current_score" DOUBLE PRECISION;
ALTER TABLE "memories_ai" ADD COLUMN "current_score" DOUBLE PRECISION;
