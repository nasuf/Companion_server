-- Additive provenance store. No historical text, scores or levels are rewritten.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
CREATE TABLE memory_evidence_links (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    memory_source TEXT NOT NULL CHECK (memory_source IN ('user', 'ai')),
    user_memory_id TEXT REFERENCES memories_user(id) ON DELETE CASCADE,
    ai_memory_id TEXT REFERENCES memories_ai(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES chat_workspaces(id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL REFERENCES ai_agents(id) ON DELETE CASCADE,
    content_version TEXT NOT NULL CHECK (content_version ~ '^[a-f0-9]{64}$'),
    source_kind TEXT NOT NULL CHECK (source_kind IN ('message','memory','import','unlinked')),
    source_ref TEXT NOT NULL,
    source_memory_side TEXT CHECK (source_memory_side IN ('user','ai')),
    source_version TEXT,
    source_user_id TEXT,
    source_workspace_id TEXT,
    source_role TEXT,
    source_status TEXT,
    source_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
    parent_user_memory_id TEXT REFERENCES memories_user(id) ON DELETE SET NULL,
    parent_ai_memory_id TEXT REFERENCES memories_ai(id) ON DELETE SET NULL,
    relation TEXT NOT NULL CHECK (relation IN ('extracted_from','derived_from','template_copy','recorded_from')),
    extractor_version TEXT NOT NULL,
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT memory_evidence_target_check CHECK (
        (memory_source='user' AND user_memory_id IS NOT NULL AND user_memory_id=memory_id AND ai_memory_id IS NULL)
        OR (memory_source='ai' AND ai_memory_id IS NOT NULL AND ai_memory_id=memory_id AND user_memory_id IS NULL)),
    CONSTRAINT memory_evidence_message_check CHECK (
        source_message_id IS NULL OR (source_kind='message' AND source_message_id=source_ref)),
    CONSTRAINT memory_evidence_parent_check CHECK (
        NOT (parent_user_memory_id IS NOT NULL AND parent_ai_memory_id IS NOT NULL)
        AND (parent_user_memory_id IS NULL OR (source_kind='memory' AND parent_user_memory_id=source_ref))
        AND (parent_ai_memory_id IS NULL OR (source_kind='memory' AND parent_ai_memory_id=source_ref)))
);
CREATE INDEX memory_evidence_target_idx ON memory_evidence_links
    (workspace_id, memory_source, memory_id, id);
CREATE INDEX memory_evidence_message_idx ON memory_evidence_links (source_message_id);
CREATE INDEX memory_evidence_parent_user_idx ON memory_evidence_links (parent_user_memory_id);
CREATE INDEX memory_evidence_parent_ai_idx ON memory_evidence_links (parent_ai_memory_id);

-- Scope validation also applies to raw/bulk inserts, independently of ORM code.
-- SHARE locks serialize binding against deletion/rebinding. These triggers
-- run only on new evidence, never on normal retrieval or lifecycle score writes.
CREATE FUNCTION validate_memory_evidence_scope() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE target RECORD; origin RECORD; space RECORD;
BEGIN
    SELECT user_id, workspace_id, content INTO target FROM memories_user
      WHERE NEW.memory_source='user' AND id=NEW.memory_id FOR SHARE;
    IF NOT FOUND THEN
      SELECT user_id, workspace_id, content INTO target FROM memories_ai
        WHERE NEW.memory_source='ai' AND id=NEW.memory_id FOR SHARE;
    END IF;
    IF NOT FOUND OR target.user_id IS DISTINCT FROM NEW.user_id
       OR target.workspace_id IS DISTINCT FROM NEW.workspace_id
       OR encode(sha256(convert_to(target.content,'UTF8')),'hex') <> NEW.content_version THEN
      RAISE EXCEPTION 'memory_evidence_target_scope_or_version' USING ERRCODE='23514';
    END IF;
    SELECT user_id, agent_id INTO space FROM chat_workspaces WHERE id=NEW.workspace_id FOR SHARE;
    IF space.user_id IS DISTINCT FROM NEW.user_id OR space.agent_id IS DISTINCT FROM NEW.agent_id THEN
      RAISE EXCEPTION 'memory_evidence_workspace_scope' USING ERRCODE='23514';
    END IF;
    IF NEW.source_kind='message' THEN
      SELECT c.user_id, c.workspace_id, c.agent_id, c.is_deleted, m.role, m.content INTO origin
        FROM messages m JOIN conversations c ON c.id=m.conversation_id
        WHERE m.id=NEW.source_ref FOR SHARE OF m,c;
      IF NOT FOUND OR NEW.source_message_id IS DISTINCT FROM NEW.source_ref OR origin.is_deleted
         OR origin.user_id IS DISTINCT FROM NEW.user_id
         OR origin.workspace_id IS DISTINCT FROM NEW.workspace_id
         OR origin.agent_id IS DISTINCT FROM NEW.agent_id
         OR origin.role IS DISTINCT FROM NEW.source_role
         OR encode(sha256(convert_to(origin.content,'UTF8')),'hex') IS DISTINCT FROM NEW.source_version THEN
        RAISE EXCEPTION 'memory_evidence_message_scope_or_version' USING ERRCODE='23514';
      END IF;
    END IF;
    IF NEW.source_kind='memory' THEN
      IF (NEW.parent_user_memory_id IS NOT NULL AND NEW.source_memory_side IS DISTINCT FROM 'user')
         OR (NEW.parent_ai_memory_id IS NOT NULL AND NEW.source_memory_side IS DISTINCT FROM 'ai') THEN
        RAISE EXCEPTION 'memory_evidence_parent_side' USING ERRCODE='23514';
      END IF;
      SELECT user_id, workspace_id, content INTO origin FROM memories_user
        WHERE id=NEW.parent_user_memory_id FOR SHARE;
      IF NOT FOUND THEN
        SELECT user_id, workspace_id, content INTO origin FROM memories_ai
          WHERE id=NEW.parent_ai_memory_id FOR SHARE;
      END IF;
      IF NOT FOUND OR origin.user_id IS DISTINCT FROM NEW.source_user_id
         OR origin.workspace_id IS DISTINCT FROM NEW.source_workspace_id
         OR encode(sha256(convert_to(origin.content,'UTF8')),'hex') IS DISTINCT FROM NEW.source_version THEN
        RAISE EXCEPTION 'memory_evidence_parent_scope_or_version' USING ERRCODE='23514';
      END IF;
      IF origin.user_id IS DISTINCT FROM NEW.user_id OR origin.workspace_id IS DISTINCT FROM NEW.workspace_id THEN
        IF NEW.relation <> 'template_copy' OR NEW.parent_ai_memory_id IS NULL
           OR NOT EXISTS (
             SELECT 1 FROM ai_agents a JOIN chat_workspaces w ON w.agent_id=a.source_template_id
             WHERE a.id=NEW.agent_id AND w.id=origin.workspace_id AND w.user_id=origin.user_id
           ) THEN
          RAISE EXCEPTION 'memory_evidence_parent_cross_scope' USING ERRCODE='23514';
        END IF;
      END IF;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER memory_evidence_validate BEFORE INSERT ON memory_evidence_links
    FOR EACH ROW EXECUTE FUNCTION validate_memory_evidence_scope();
-- Only referential deletion may null a live source pointer. Preserve the
-- original reference, side, scope and content hashes as immutable snapshots.
CREATE FUNCTION preserve_memory_evidence_snapshot() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(NEW)-'source_message_id'-'parent_user_memory_id'-'parent_ai_memory_id')
       IS DISTINCT FROM (to_jsonb(OLD)-'source_message_id'-'parent_user_memory_id'-'parent_ai_memory_id')
       OR (NEW.source_message_id IS NOT NULL AND NEW.source_message_id IS DISTINCT FROM OLD.source_message_id)
       OR (NEW.parent_user_memory_id IS NOT NULL AND NEW.parent_user_memory_id IS DISTINCT FROM OLD.parent_user_memory_id)
       OR (NEW.parent_ai_memory_id IS NOT NULL AND NEW.parent_ai_memory_id IS DISTINCT FROM OLD.parent_ai_memory_id) THEN
      RAISE EXCEPTION 'memory_evidence_immutable' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER memory_evidence_preserve BEFORE UPDATE ON memory_evidence_links
    FOR EACH ROW EXECUTE FUNCTION preserve_memory_evidence_snapshot();
COMMIT;
