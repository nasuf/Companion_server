-- Additive import/edit receipts. Historic import/unlinked evidence stays unknown.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

CREATE TABLE memory_input_origins (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES chat_workspaces(id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL REFERENCES ai_agents(id) ON DELETE CASCADE,
    source_version TEXT NOT NULL CHECK (source_version ~ '^[a-f0-9]{64}$'),
    source_status TEXT NOT NULL CHECK (source_status IN ('knowledge_document','admin_edit','user_edit')),
    format_version TEXT NOT NULL CHECK (format_version='memory-input-v1'),
    payload_text TEXT NOT NULL CHECK ((octet_length(payload_text)<=8388608
        AND jsonb_typeof(payload_text::jsonb)='object'
        AND jsonb_typeof(payload_text::jsonb->'input')='object'
        AND payload_text::jsonb->>'format'=format_version
        AND payload_text::jsonb->>'kind'=source_status
        AND encode(sha256(convert_to(payload_text,'UTF8')),'hex')=source_version) IS TRUE),
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX memory_input_scope_idx ON memory_input_origins (workspace_id,agent_id,id);
CREATE FUNCTION validate_memory_input_origin() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP='UPDATE' THEN
        RAISE EXCEPTION 'memory_input_origin_immutable' USING ERRCODE='23514';
    END IF;
    PERFORM 1 FROM chat_workspaces w JOIN ai_agents a ON a.id=w.agent_id
      WHERE w.id=NEW.workspace_id AND w.user_id=NEW.user_id
        AND a.id=NEW.agent_id AND a.user_id=NEW.user_id FOR SHARE OF w,a;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'memory_input_origin_scope' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER memory_input_origin_validate BEFORE INSERT OR UPDATE ON memory_input_origins
    FOR EACH ROW EXECUTE FUNCTION validate_memory_input_origin();

ALTER TABLE memory_evidence_links DROP CONSTRAINT memory_evidence_links_source_kind_check;
ALTER TABLE memory_evidence_links ADD CONSTRAINT memory_evidence_links_source_kind_check
    CHECK (source_kind IN ('message','memory','import','unlinked','profile','input'));
ALTER TABLE memory_evidence_links ADD COLUMN source_input_id TEXT
    REFERENCES memory_input_origins(id) ON DELETE SET NULL;
ALTER TABLE memory_evidence_links ADD CONSTRAINT memory_evidence_input_check CHECK (
    source_input_id IS NULL OR (source_kind='input' AND source_input_id=source_ref));
CREATE INDEX memory_evidence_input_idx ON memory_evidence_links (source_input_id);

CREATE FUNCTION validate_memory_input_evidence() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE origin RECORD;
BEGIN
    IF NEW.source_kind='input' THEN
        SELECT * INTO origin FROM memory_input_origins WHERE id=NEW.source_ref FOR SHARE;
        IF NOT FOUND OR NEW.source_input_id IS DISTINCT FROM origin.id
           OR NEW.relation<>'recorded_from'
           OR NEW.user_id IS DISTINCT FROM origin.user_id
           OR NEW.workspace_id IS DISTINCT FROM origin.workspace_id
           OR NEW.agent_id IS DISTINCT FROM origin.agent_id
           OR NEW.source_user_id IS DISTINCT FROM origin.user_id
           OR NEW.source_workspace_id IS DISTINCT FROM origin.workspace_id
           OR NEW.source_version IS DISTINCT FROM origin.source_version
           OR NEW.source_status IS DISTINCT FROM origin.source_status
           OR NEW.source_role IS NOT NULL OR NEW.source_memory_side IS NOT NULL
           OR NEW.source_message_id IS NOT NULL OR NEW.source_profile_id IS NOT NULL
           OR NEW.parent_user_memory_id IS NOT NULL OR NEW.parent_ai_memory_id IS NOT NULL THEN
            RAISE EXCEPTION 'memory_input_evidence_scope_or_version' USING ERRCODE='23514';
        END IF;
        IF origin.source_status='knowledge_document' AND (
            NEW.memory_source<>'ai' OR NOT EXISTS (
                SELECT 1 FROM memories_ai WHERE id=NEW.memory_id AND provenance='knowledge_seed' FOR SHARE)) THEN
            RAISE EXCEPTION 'memory_input_knowledge_target' USING ERRCODE='23514';
        END IF;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER memory_evidence_input_validate BEFORE INSERT ON memory_evidence_links
    FOR EACH ROW EXECUTE FUNCTION validate_memory_input_evidence();

-- Only live FK pointers may become NULL when a source is deleted. All source
-- versions, owner snapshots and content hashes remain immutable for history.
CREATE OR REPLACE FUNCTION preserve_memory_evidence_snapshot() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(NEW)-'source_message_id'-'parent_user_memory_id'-'parent_ai_memory_id'-'source_profile_id'-'source_input_id')
       IS DISTINCT FROM (to_jsonb(OLD)-'source_message_id'-'parent_user_memory_id'-'parent_ai_memory_id'-'source_profile_id'-'source_input_id')
       OR (NEW.source_message_id IS NOT NULL AND NEW.source_message_id IS DISTINCT FROM OLD.source_message_id)
       OR (NEW.parent_user_memory_id IS NOT NULL AND NEW.parent_user_memory_id IS DISTINCT FROM OLD.parent_user_memory_id)
       OR (NEW.parent_ai_memory_id IS NOT NULL AND NEW.parent_ai_memory_id IS DISTINCT FROM OLD.parent_ai_memory_id)
       OR (NEW.source_profile_id IS NOT NULL AND NEW.source_profile_id IS DISTINCT FROM OLD.source_profile_id)
       OR (NEW.source_input_id IS NOT NULL AND NEW.source_input_id IS DISTINCT FROM OLD.source_input_id) THEN
        RAISE EXCEPTION 'memory_evidence_immutable' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END $$;
COMMIT;
