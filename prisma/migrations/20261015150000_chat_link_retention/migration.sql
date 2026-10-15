-- No historical source inference or mass backfill. Existing message projections
-- are checked directly through narrow indexes before a resource is retired.
CREATE INDEX messages_link_cache_ref_idx ON messages USING hash ((metadata #>> '{link_card,id}'));
CREATE INDEX messages_link_snapshot_ref_idx ON messages USING hash ((metadata #>> '{link_card,snapshot_id}'));
CREATE INDEX messages_component_link_ref_idx ON messages USING hash ((metadata #>> '{component_card,payload,link_id}'));
CREATE INDEX messages_component_snapshot_ref_idx ON messages USING hash ((metadata #>> '{component_card,payload,snapshot_id}'));
CREATE INDEX messages_link_image_ref_idx ON messages USING hash ((metadata #>> '{link_card,image_url}'));
CREATE INDEX messages_component_image_ref_idx ON messages USING hash ((metadata #>> '{component_card,payload,image_url}'));
CREATE INDEX chat_link_cards_image_idx ON chat_link_cards USING hash(image_url);
CREATE INDEX chat_link_snapshots_image_idx ON chat_link_snapshots USING hash ((payload ->> 'image_url'));

CREATE TABLE chat_link_storage_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    reserved_bytes BIGINT NOT NULL DEFAULT 0 CHECK (reserved_bytes >= 0),
    cursors JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_preview JSONB,
    last_run JSONB,
    run_id TEXT,
    running_since TIMESTAMPTZ(6)
);
INSERT INTO chat_link_storage_state(id) VALUES (1);

-- Only files reserved through this ledger are eligible for automatic unlink.
-- Existing unregistered files and user-uploaded media are never swept by glob.
CREATE TABLE chat_link_cover_objects (
    storage_key TEXT PRIMARY KEY,
    image_url TEXT NOT NULL UNIQUE,
    byte_size BIGINT NOT NULL CHECK (byte_size > 0),
    state TEXT NOT NULL DEFAULT 'live' CHECK (state IN ('live','deleting')),
    created_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE FUNCTION guard_chat_link_resource_write() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    image TEXT;
    cache_id TEXT;
    version_id TEXT;
BEGIN
    -- Only link writes contend with the bounded maintenance transaction. No
    -- network or filesystem IO is permitted while this lock is held.
    IF TG_TABLE_NAME = 'messages' THEN
        IF TG_OP = 'UPDATE' AND
           (NEW.metadata->'link_card') IS NOT DISTINCT FROM (OLD.metadata->'link_card') AND
           (NEW.metadata->'component_card') IS NOT DISTINCT FROM (OLD.metadata->'component_card') THEN
            RETURN NEW;
        END IF;
        IF jsonb_typeof(NEW.metadata->'link_card') IS DISTINCT FROM 'object' AND
           NEW.metadata #>> '{component_card,type}' IS DISTINCT FROM 'external_link' THEN
            RETURN NEW;
        END IF;
    END IF;
    PERFORM pg_advisory_xact_lock(87231615);
    IF TG_TABLE_NAME = 'messages' THEN
        -- Both projections are protected; conflicting identities must not slip
        -- through one reader while another reader sees a different resource.
        FOR cache_id, version_id, image IN
            SELECT NEW.metadata #>> '{link_card,id}', NEW.metadata #>> '{link_card,snapshot_id}',
                   NEW.metadata #>> '{link_card,image_url}'
            UNION ALL
            SELECT NEW.metadata #>> '{component_card,payload,link_id}',
                   NEW.metadata #>> '{component_card,payload,snapshot_id}',
                   NEW.metadata #>> '{component_card,payload,image_url}'
        LOOP
            -- Malformed ancient projections remain readable. New string cache
            -- identities must resolve in this conversation, including reshares.
            IF cache_id IS NOT NULL AND
               (jsonb_typeof(NEW.metadata #> '{link_card,id}') = 'string' OR
                jsonb_typeof(NEW.metadata #> '{component_card,payload,link_id}') = 'string') AND
               NOT EXISTS (SELECT 1 FROM chat_link_cards l WHERE l.id=cache_id
                           AND l.conversation_id=NEW.conversation_id) THEN
                RAISE EXCEPTION 'chat_link_resource_unavailable';
            END IF;
            IF version_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM chat_link_snapshots s JOIN chat_link_cards l ON l.id=s.link_card_id
                WHERE s.id=version_id AND l.id=cache_id AND l.conversation_id=NEW.conversation_id
            ) THEN
                RAISE EXCEPTION 'chat_link_resource_unavailable';
            END IF;
            IF EXISTS (SELECT 1 FROM chat_link_cover_objects WHERE image_url=image AND state='deleting') THEN
                RAISE EXCEPTION 'chat_link_cover_retiring';
            END IF;
        END LOOP;
    ELSE
        IF TG_TABLE_NAME = 'chat_link_cards' THEN image := NEW.image_url;
        ELSE image := NEW.payload->>'image_url'; END IF;
        IF EXISTS (SELECT 1 FROM chat_link_cover_objects WHERE image_url=image AND state='deleting') THEN
            RAISE EXCEPTION 'chat_link_cover_retiring';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER chat_link_cards_retention_guard BEFORE INSERT OR UPDATE ON chat_link_cards
    FOR EACH ROW EXECUTE FUNCTION guard_chat_link_resource_write();
CREATE TRIGGER chat_link_snapshots_retention_guard BEFORE INSERT ON chat_link_snapshots
    FOR EACH ROW EXECUTE FUNCTION guard_chat_link_resource_write();
CREATE TRIGGER messages_link_retention_guard BEFORE INSERT OR UPDATE OF metadata ON messages
    FOR EACH ROW EXECUTE FUNCTION guard_chat_link_resource_write();
