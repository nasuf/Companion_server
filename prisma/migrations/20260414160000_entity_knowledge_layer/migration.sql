-- Entity knowledge layer: canonical entities (people / places / topics /
-- preferences) with a many-to-many mentions edge to memories_user/ai.
-- Single source of truth, cascades naturally on entity delete.

CREATE TABLE "memory_entities" (
    "id"              TEXT        NOT NULL,
    "user_id"         TEXT        NOT NULL,
    "workspace_id"    TEXT,
    "canonical_name"  TEXT        NOT NULL,
    "aliases"         TEXT[]      NOT NULL DEFAULT ARRAY[]::TEXT[],
    "entity_type"     TEXT        NOT NULL,
    "role"            TEXT,
    "mention_count"   INTEGER     NOT NULL DEFAULT 0,
    "last_mentioned_at" TIMESTAMP(3),
    "is_archived"     BOOLEAN     NOT NULL DEFAULT false,
    "metadata"        JSONB,
    "created_at"      TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at"      TIMESTAMP(3) NOT NULL,
    CONSTRAINT "memory_entities_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "memory_entities_user_workspace_type_name_key"
    ON "memory_entities"("user_id", "workspace_id", "entity_type", "canonical_name");

CREATE INDEX "memory_entities_user_workspace_type_idx"
    ON "memory_entities"("user_id", "workspace_id", "entity_type");

CREATE INDEX "memory_entities_user_workspace_mentions_idx"
    ON "memory_entities"("user_id", "workspace_id", "mention_count" DESC);

CREATE INDEX "memory_entities_archive_scan_idx"
    ON "memory_entities"("user_id", "workspace_id", "is_archived", "last_mentioned_at");

-- many-to-many memory ↔ entity
CREATE TABLE "memory_mentions" (
    "memory_id"     TEXT NOT NULL,
    "memory_source" TEXT NOT NULL,            -- 'user' | 'ai'
    "entity_id"     TEXT NOT NULL,
    "user_id"       TEXT NOT NULL,
    "workspace_id"  TEXT,
    "created_at"    TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "memory_mentions_pkey" PRIMARY KEY ("memory_id", "entity_id")
);

CREATE INDEX "memory_mentions_entity_idx" ON "memory_mentions"("entity_id");
CREATE INDEX "memory_mentions_user_ws_idx" ON "memory_mentions"("user_id", "workspace_id");
CREATE INDEX "memory_mentions_memory_idx" ON "memory_mentions"("memory_id");

-- Cascade on entity delete (removing an entity forgets all mentions)
ALTER TABLE "memory_mentions"
    ADD CONSTRAINT "memory_mentions_entity_id_fkey"
    FOREIGN KEY ("entity_id") REFERENCES "memory_entities"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;

-- When a memory is deleted (from either memories_user or memories_ai),
-- drop its mentions. Same pattern as the orphan-cleanup triggers for
-- embeddings/changelogs (we can't use a single FK because memory_id
-- spans two tables).
CREATE OR REPLACE FUNCTION cleanup_memory_mentions()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM memory_mentions WHERE memory_id = OLD.id;
    RETURN OLD;
END;
$$;

DROP TRIGGER IF EXISTS trg_cleanup_memory_user_mentions ON memories_user;
CREATE TRIGGER trg_cleanup_memory_user_mentions
AFTER DELETE ON memories_user
FOR EACH ROW EXECUTE FUNCTION cleanup_memory_mentions();

DROP TRIGGER IF EXISTS trg_cleanup_memory_ai_mentions ON memories_ai;
CREATE TRIGGER trg_cleanup_memory_ai_mentions
AFTER DELETE ON memories_ai
FOR EACH ROW EXECUTE FUNCTION cleanup_memory_mentions();
