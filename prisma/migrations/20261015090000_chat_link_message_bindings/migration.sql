-- Additive relation only: old message projections are not fabricated/backfilled.
CREATE TABLE chat_link_message_bindings (
    message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE ON UPDATE RESTRICT,
    snapshot_id TEXT NOT NULL REFERENCES chat_link_snapshots(id) ON DELETE NO ACTION ON UPDATE RESTRICT
        DEFERRABLE INITIALLY DEFERRED,
    bound_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX chat_link_message_bindings_snapshot_idx ON chat_link_message_bindings(snapshot_id);
-- A live share protects its version. Deferral permits owner/conversation deletion
-- to cascade messages and snapshots within one transaction regardless of order.
CREATE FUNCTION reject_chat_link_binding_update() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'chat_link_binding_is_immutable';
END;
$$;
CREATE TRIGGER chat_link_message_bindings_immutable BEFORE UPDATE ON chat_link_message_bindings
    FOR EACH ROW EXECUTE FUNCTION reject_chat_link_binding_update();
