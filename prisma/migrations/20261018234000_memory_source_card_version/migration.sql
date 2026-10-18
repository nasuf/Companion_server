-- Additive metadata version for structured message sources; old writers remain valid.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
ALTER TABLE memory_evidence_links ADD COLUMN source_card_version TEXT
    CHECK (source_card_version IS NULL OR source_card_version ~ '^[a-f0-9]{64}$');

CREATE FUNCTION validate_memory_evidence_card_version() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE current_version TEXT;
BEGIN
    IF NEW.source_card_version IS NOT NULL THEN
        IF NEW.source_kind <> 'message' THEN
            RAISE EXCEPTION 'memory_evidence_card_requires_message' USING ERRCODE='23514';
        END IF;
        SELECT encode(sha256(convert_to((metadata->'component_card')::text,'UTF8')),'hex')
          INTO current_version FROM messages WHERE id=NEW.source_message_id FOR SHARE;
        IF NOT FOUND OR current_version IS DISTINCT FROM NEW.source_card_version THEN
            RAISE EXCEPTION 'memory_evidence_card_version_changed' USING ERRCODE='23514';
        END IF;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER memory_evidence_card_validate BEFORE INSERT ON memory_evidence_links
    FOR EACH ROW EXECUTE FUNCTION validate_memory_evidence_card_version();
-- The existing generic immutable-snapshot trigger also protects this column.
COMMIT;
