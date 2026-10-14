-- Cache IDs remain stable for existing clients. Prepared versions are append-only.
CREATE TABLE chat_link_snapshots (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    link_card_id TEXT NOT NULL REFERENCES chat_link_cards(id) ON DELETE CASCADE,
    payload JSONB NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    recorded_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK ((payload ->> 'id' = link_card_id) IS TRUE),
    CHECK (octet_length(payload::text) <= 131072)
);
CREATE INDEX chat_link_snapshots_cache_recorded_idx
    ON chat_link_snapshots(link_card_id, recorded_at DESC, id);

CREATE FUNCTION reject_chat_link_snapshot_update() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'chat_link_snapshot_is_immutable';
END;
$$;
CREATE TRIGGER chat_link_snapshots_immutable BEFORE UPDATE ON chat_link_snapshots
    FOR EACH ROW EXECUTE FUNCTION reject_chat_link_snapshot_update();

-- A cache may now serve many messages. Deleting its old pointer must not delete
-- the cache (or the prepared snapshots) still referenced by other messages.
ALTER TABLE chat_link_cards DROP CONSTRAINT chat_link_cards_message_id_fkey;
ALTER TABLE chat_link_cards ADD CONSTRAINT chat_link_cards_message_id_fkey
    FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE SET NULL;

-- No speculative historical backfill: older messages already contain their
-- own link_card projection; their actual capture time remains unknown.
CREATE INDEX messages_link_history_idx ON messages(conversation_id, created_at DESC, id)
    WHERE jsonb_typeof(metadata -> 'link_card') = 'object';
