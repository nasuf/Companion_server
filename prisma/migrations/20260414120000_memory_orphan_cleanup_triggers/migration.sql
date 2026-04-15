-- Defense-in-depth: when a memory row is deleted from either memories_user
-- or memories_ai, cascade-delete its embedding and changelog rows.
-- App code (memory_repo.delete) already does this explicitly, but any
-- direct SQL DELETE or future code path that bypasses the repo will now
-- leave no orphans.
--
-- Note: memory_embeddings and memory_changelogs reference memories by id
-- across TWO tables, which is why we can't use a single FK. A trigger per
-- source table is the cleanest alternative.

CREATE OR REPLACE FUNCTION cleanup_memory_dependents()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM memory_embeddings WHERE memory_id = OLD.id;
    DELETE FROM memory_changelogs WHERE memory_id = OLD.id;
    RETURN OLD;
END;
$$;

DROP TRIGGER IF EXISTS trg_cleanup_memory_user_dependents ON memories_user;
CREATE TRIGGER trg_cleanup_memory_user_dependents
AFTER DELETE ON memories_user
FOR EACH ROW EXECUTE FUNCTION cleanup_memory_dependents();

DROP TRIGGER IF EXISTS trg_cleanup_memory_ai_dependents ON memories_ai;
CREATE TRIGGER trg_cleanup_memory_ai_dependents
AFTER DELETE ON memories_ai
FOR EACH ROW EXECUTE FUNCTION cleanup_memory_dependents();

-- Additional safety net: clean up any embeddings / changelogs whose
-- memory no longer exists. Idempotent and safe to re-run.
DELETE FROM memory_embeddings e
WHERE NOT EXISTS (SELECT 1 FROM memories_user u WHERE u.id = e.memory_id)
  AND NOT EXISTS (SELECT 1 FROM memories_ai a   WHERE a.id = e.memory_id);

DELETE FROM memory_changelogs c
WHERE NOT EXISTS (SELECT 1 FROM memories_user u WHERE u.id = c.memory_id)
  AND NOT EXISTS (SELECT 1 FROM memories_ai a   WHERE a.id = c.memory_id);
