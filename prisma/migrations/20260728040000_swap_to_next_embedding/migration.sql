-- Second half of the bge-m3 → qwen3-embedding:0.6b switch. Run only after
-- scripts/reembed_memories.py reports zero rows left, and re-run it just
-- before deploying to pick up anything written in between.
--
-- Deployment stops the server before migrating, so nothing reads the table
-- while the columns change hands. That is what keeps the switch atomic: the
-- old vectors serve right up to the stop, and the new ones serve from the
-- start.
--
-- The NOT NULL below is the safety catch. If any row is still missing its new
-- vector the migration fails and the deploy stops, which is the outcome we
-- want — shipping half the corpus in one vector space and half in another
-- would leave retrieval silently returning noise.

-- 1. Fail loudly rather than ship a partial swap.
ALTER TABLE memory_embeddings
    ALTER COLUMN embedding_next SET NOT NULL;

-- 2. The ivfflat index is bound to the old column; it has to go first.
DROP INDEX IF EXISTS idx_memory_embeddings_vector;

-- 3. Hand over.
ALTER TABLE memory_embeddings DROP COLUMN embedding;
ALTER TABLE memory_embeddings RENAME COLUMN embedding_next TO embedding;

-- 4. Rebuild the index. lists=100 wants roughly 3900 rows to be effective and
--    we have ~8200, so the same setting as before still applies.
CREATE INDEX idx_memory_embeddings_vector
    ON memory_embeddings
    USING ivfflat (embedding extensions.vector_cosine_ops)
    WITH (lists = 100);
