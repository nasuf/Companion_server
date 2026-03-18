CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id TEXT PRIMARY KEY,
    embedding extensions.vector(768) NOT NULL
);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'memory_embeddings'
          AND column_name = 'embedding'
    ) THEN
        ALTER TABLE memory_embeddings
            ALTER COLUMN embedding TYPE extensions.vector(768)
            USING embedding::extensions.vector(768);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_memory_embeddings_vector
    ON memory_embeddings
    USING ivfflat (embedding extensions.vector_cosine_ops)
    WITH (lists = 100);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE table_name = 'memory_changelogs'
          AND constraint_type = 'FOREIGN KEY'
          AND constraint_name = 'memory_changelogs_memory_id_fkey'
    ) THEN
        ALTER TABLE memory_changelogs
            DROP CONSTRAINT memory_changelogs_memory_id_fkey;
    END IF;
END $$;

ALTER TABLE memories_user
    ADD COLUMN IF NOT EXISTS mention_count INTEGER NOT NULL DEFAULT 0;

ALTER TABLE memories_ai
    ADD COLUMN IF NOT EXISTS mention_count INTEGER NOT NULL DEFAULT 0;
