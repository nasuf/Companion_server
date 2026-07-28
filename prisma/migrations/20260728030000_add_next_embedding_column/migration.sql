-- Switching embedding model: bge-m3 → qwen3-embedding:0.6b (both 1024d).
--
-- The two models put vectors in different spaces, so queries embedded with one
-- and memories embedded with the other produce noise, not weak results. The
-- switch therefore has to be atomic across 8181 rows.
--
-- The previous model switch (20260417100000) simply truncated and re-embedded,
-- which left retrieval dead for the duration. Dimensions match this time, so a
-- second column can be filled in the background and swapped in during the
-- deploy window instead — see the paired migration
-- 20260728040000_swap_to_next_embedding.
--
-- Nullable on purpose: rows fill in progressively while the old column keeps
-- serving. The swap migration is what enforces completeness.

ALTER TABLE memory_embeddings
    ADD COLUMN embedding_next extensions.vector(1024);
