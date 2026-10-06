-- Additive foundation only: current workers do not yet consume these fences.
-- Stop the release if locks cannot be acquired promptly; never leave partial DDL.
BEGIN;
SET LOCAL lock_timeout = '2s';
SET LOCAL statement_timeout = '15s';

ALTER TABLE users ADD COLUMN execution_generation TEXT NOT NULL DEFAULT gen_random_uuid()::text;
ALTER TABLE ai_agents ADD COLUMN execution_generation TEXT NOT NULL DEFAULT gen_random_uuid()::text;
ALTER TABLE chat_workspaces ADD COLUMN execution_generation TEXT NOT NULL DEFAULT gen_random_uuid()::text;
ALTER TABLE conversations ADD COLUMN execution_generation TEXT NOT NULL DEFAULT gen_random_uuid()::text;

CREATE FUNCTION rotate_execution_generation() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE watched_column TEXT;
BEGIN
    -- Old application inserts omit the new column. Explicit ID reuse must never
    -- resurrect an old scope, even if an import supplies its old generation.
    IF TG_OP = 'INSERT' THEN
        NEW.execution_generation := gen_random_uuid()::text;
        RETURN NEW;
    END IF;
    IF NEW.execution_generation IS DISTINCT FROM OLD.execution_generation THEN
        NEW.execution_generation := gen_random_uuid()::text;
        RETURN NEW;
    END IF;
    FOREACH watched_column IN ARRAY TG_ARGV LOOP
        IF (to_jsonb(NEW) -> watched_column) IS DISTINCT FROM
           (to_jsonb(OLD) -> watched_column) THEN
            NEW.execution_generation := gen_random_uuid()::text;
            EXIT;
        END IF;
    END LOOP;
    RETURN NEW;
END;
$$;

CREATE TRIGGER users_execution_generation BEFORE INSERT OR UPDATE OF id, status, archived_at, role, execution_generation ON users
FOR EACH ROW EXECUTE FUNCTION rotate_execution_generation('id', 'status', 'archived_at', 'role');
CREATE TRIGGER ai_agents_execution_generation BEFORE INSERT OR UPDATE OF id, user_id, status, archived_at, execution_generation ON ai_agents
FOR EACH ROW EXECUTE FUNCTION rotate_execution_generation('id', 'user_id', 'status', 'archived_at');
CREATE TRIGGER chat_workspaces_execution_generation BEFORE INSERT OR UPDATE OF id, user_id, agent_id, status, archived_at, execution_generation ON chat_workspaces
FOR EACH ROW EXECUTE FUNCTION rotate_execution_generation('id', 'user_id', 'agent_id', 'status', 'archived_at');
CREATE TRIGGER conversations_execution_generation BEFORE INSERT OR UPDATE OF id, user_id, agent_id, workspace_id, is_deleted, archived_at, execution_generation ON conversations
FOR EACH ROW EXECUTE FUNCTION rotate_execution_generation('id', 'user_id', 'agent_id', 'workspace_id', 'is_deleted', 'archived_at');

COMMIT;
