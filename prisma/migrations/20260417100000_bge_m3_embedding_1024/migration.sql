-- Switch embedding model: nomic-embed-text (768d) → bge-m3 (1024d)
-- Old embeddings are incompatible with the new dimension, so truncate first.

-- 1. Drop the old IVFFlat index (dimension-specific)
DROP INDEX IF EXISTS idx_memory_embeddings_vector;

-- 2. Clear all existing embeddings (768d, incompatible with 1024d)
TRUNCATE TABLE memory_embeddings;

-- 3. Alter column from vector(768) to vector(1024)
ALTER TABLE memory_embeddings
    ALTER COLUMN embedding TYPE extensions.vector(1024)
    USING embedding::extensions.vector(1024);

-- 4. Recreate IVFFlat index for 1024 dimensions
-- Note: IVFFlat requires at least (lists * 39) rows to build properly.
-- With lists=100, that's 3900 rows. For small datasets, the index will be
-- created but may not be effective until enough data accumulates.
CREATE INDEX idx_memory_embeddings_vector
    ON memory_embeddings
    USING ivfflat (embedding extensions.vector_cosine_ops)
    WITH (lists = 100);
