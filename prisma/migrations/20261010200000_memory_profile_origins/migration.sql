-- Additive initialization evidence; no historic persona content is backfilled.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

CREATE TABLE memory_profile_origins (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES chat_workspaces(id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL REFERENCES ai_agents(id) ON DELETE CASCADE,
    source_version TEXT NOT NULL CHECK (source_version ~ '^[a-f0-9]{64}$'),
    source_status TEXT NOT NULL CHECK (source_status IN ('generated_profile','imported_profile','provided_profile')),
    input_status TEXT NOT NULL CHECK (input_status IN ('recorded','uncollected')),
    format_version TEXT NOT NULL CHECK (format_version='persona-profile-v1'),
    payload_text TEXT NOT NULL CHECK ((octet_length(payload_text)<=262144
        AND jsonb_typeof(payload_text::jsonb)='object'
        AND jsonb_typeof(payload_text::jsonb->'profile')='object'
        AND jsonb_typeof(payload_text::jsonb->'career') IN ('object','null')
        AND payload_text::jsonb->>'format'=format_version
        AND payload_text::jsonb->>'kind'=source_status
        AND CASE WHEN input_status='recorded' THEN jsonb_typeof(payload_text::jsonb->'invocation_inputs')='object'
                 ELSE payload_text::jsonb->'invocation_inputs'='null'::jsonb END
        AND encode(sha256(convert_to(payload_text,'UTF8')),'hex')=source_version) IS TRUE),
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX memory_profile_scope_idx ON memory_profile_origins (workspace_id,agent_id,id);
CREATE FUNCTION validate_memory_profile_origin() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP='UPDATE' THEN
        RAISE EXCEPTION 'memory_profile_origin_immutable' USING ERRCODE='23514';
    END IF;
    PERFORM 1 FROM chat_workspaces w JOIN ai_agents a ON a.id=w.agent_id
      WHERE w.id=NEW.workspace_id AND w.user_id=NEW.user_id
        AND a.id=NEW.agent_id AND a.user_id=NEW.user_id FOR SHARE OF w,a;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'memory_profile_origin_scope' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER memory_profile_origin_validate BEFORE INSERT OR UPDATE ON memory_profile_origins
    FOR EACH ROW EXECUTE FUNCTION validate_memory_profile_origin();

ALTER TABLE memory_evidence_links DROP CONSTRAINT memory_evidence_links_source_kind_check;
ALTER TABLE memory_evidence_links ADD CONSTRAINT memory_evidence_links_source_kind_check
    CHECK (source_kind IN ('message','memory','import','unlinked','profile'));
ALTER TABLE memory_evidence_links ADD COLUMN source_profile_id TEXT
    REFERENCES memory_profile_origins(id) ON DELETE SET NULL;
ALTER TABLE memory_evidence_links ADD CONSTRAINT memory_evidence_profile_check CHECK (
    source_profile_id IS NULL OR (source_kind='profile' AND source_profile_id=source_ref));
CREATE INDEX memory_evidence_profile_idx ON memory_evidence_links (source_profile_id);

CREATE FUNCTION validate_memory_profile_evidence() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE origin RECORD;
BEGIN
    IF NEW.source_kind='profile' THEN
        SELECT * INTO origin FROM memory_profile_origins WHERE id=NEW.source_ref FOR SHARE;
        IF NOT FOUND OR NEW.source_profile_id IS DISTINCT FROM origin.id
           OR NEW.memory_source<>'ai' OR NEW.relation<>'derived_from'
           OR NEW.user_id IS DISTINCT FROM origin.user_id
           OR NEW.workspace_id IS DISTINCT FROM origin.workspace_id
           OR NEW.agent_id IS DISTINCT FROM origin.agent_id
           OR NEW.source_user_id IS DISTINCT FROM origin.user_id
           OR NEW.source_workspace_id IS DISTINCT FROM origin.workspace_id
           OR NEW.source_version IS DISTINCT FROM origin.source_version
           OR NEW.source_status IS DISTINCT FROM origin.source_status
           OR NEW.source_role IS NOT NULL OR NEW.source_memory_side IS NOT NULL
           OR NEW.source_message_id IS NOT NULL
           OR NEW.parent_user_memory_id IS NOT NULL OR NEW.parent_ai_memory_id IS NOT NULL
           OR NOT EXISTS (SELECT 1 FROM memories_ai WHERE id=NEW.memory_id AND provenance='profile_seed' FOR SHARE) THEN
            RAISE EXCEPTION 'memory_profile_evidence_scope_or_version' USING ERRCODE='23514';
        END IF;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER memory_evidence_profile_validate BEFORE INSERT ON memory_evidence_links
    FOR EACH ROW EXECUTE FUNCTION validate_memory_profile_evidence();

-- Extend immutable link protection solely for FK deletion of the live profile
-- pointer. The original ref/version remain recorded; resurrection is forbidden.
CREATE OR REPLACE FUNCTION preserve_memory_evidence_snapshot() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(NEW)-'source_message_id'-'parent_user_memory_id'-'parent_ai_memory_id'-'source_profile_id')
       IS DISTINCT FROM (to_jsonb(OLD)-'source_message_id'-'parent_user_memory_id'-'parent_ai_memory_id'-'source_profile_id')
       OR (NEW.source_message_id IS NOT NULL AND NEW.source_message_id IS DISTINCT FROM OLD.source_message_id)
       OR (NEW.parent_user_memory_id IS NOT NULL AND NEW.parent_user_memory_id IS DISTINCT FROM OLD.parent_user_memory_id)
       OR (NEW.parent_ai_memory_id IS NOT NULL AND NEW.parent_ai_memory_id IS DISTINCT FROM OLD.parent_ai_memory_id)
       OR (NEW.source_profile_id IS NOT NULL AND NEW.source_profile_id IS DISTINCT FROM OLD.source_profile_id) THEN
        RAISE EXCEPTION 'memory_evidence_immutable' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END $$;
-- Legacy dependencies are keyed by ID without a memory side. Never remove
-- a surviving opposite-side memory's vector/audit when replacing a persona.
CREATE OR REPLACE FUNCTION cleanup_memory_dependents() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (TG_TABLE_NAME='memories_ai' AND EXISTS (SELECT 1 FROM memories_user WHERE id=OLD.id))
       OR (TG_TABLE_NAME='memories_user' AND EXISTS (SELECT 1 FROM memories_ai WHERE id=OLD.id)) THEN
        RETURN OLD;
    END IF;
    DELETE FROM memory_embeddings WHERE memory_id=OLD.id;
    DELETE FROM memory_changelogs WHERE memory_id=OLD.id;
    RETURN OLD;
END $$;
COMMIT;
