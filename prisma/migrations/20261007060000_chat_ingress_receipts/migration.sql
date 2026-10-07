-- Staged SQL ingress only. No legacy messages, queues or runtime rows are changed.
BEGIN;
SET LOCAL lock_timeout = '2s';
SET LOCAL statement_timeout = '15s';

CREATE TABLE chat_ingress_receipts (
    id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    request_key TEXT NOT NULL,
    source TEXT NOT NULL,
    client_id TEXT,
    input_fingerprint TEXT NOT NULL,
    request_input JSONB NOT NULL,
    prepared_input JSONB NOT NULL,
    message_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    ordinal BIGSERIAL NOT NULL,
    received_at TIMESTAMPTZ(6) NOT NULL,
    accepted_at TIMESTAMPTZ(6) NOT NULL,
    CONSTRAINT chat_ingress_receipts_pkey PRIMARY KEY (id),
    CONSTRAINT chat_ingress_receipts_shape CHECK (
        ordinal > 0
        AND length(btrim(request_key)) BETWEEN 1 AND 256
        AND input_fingerprint ~ '^[0-9a-f]{64}$'
        AND jsonb_typeof(request_input) = 'object'
        AND jsonb_typeof(prepared_input) = 'object'
        AND ((source = 'client' AND client_id IS NOT NULL
              AND length(btrim(client_id)) BETWEEN 1 AND 249
              AND request_key = 'client:' || client_id)
             OR (source = 'wechat' AND client_id IS NULL
                 AND request_key LIKE 'wechat:%' AND length(request_key) > 7))
    ),
    CONSTRAINT chat_ingress_receipts_conversation_id_fkey FOREIGN KEY (conversation_id)
        REFERENCES conversations(id) ON DELETE CASCADE ON UPDATE RESTRICT,
    CONSTRAINT chat_ingress_receipts_message_id_fkey FOREIGN KEY (message_id)
        REFERENCES messages(id) ON DELETE CASCADE ON UPDATE RESTRICT,
    CONSTRAINT chat_ingress_receipts_run_id_fkey FOREIGN KEY (run_id)
        REFERENCES agent_runs(id) ON DELETE CASCADE ON UPDATE RESTRICT,
    CONSTRAINT chat_ingress_receipts_job_id_fkey FOREIGN KEY (job_id)
        REFERENCES runtime_jobs(id) ON DELETE CASCADE ON UPDATE RESTRICT
);
CREATE UNIQUE INDEX chat_ingress_receipts_request_key
    ON chat_ingress_receipts(conversation_id, request_key);
CREATE UNIQUE INDEX chat_ingress_receipts_message_key ON chat_ingress_receipts(message_id);
CREATE UNIQUE INDEX chat_ingress_receipts_conversation_order
    ON chat_ingress_receipts(conversation_id, ordinal);
CREATE INDEX chat_ingress_receipts_run_order ON chat_ingress_receipts(run_id, ordinal);
CREATE INDEX chat_ingress_receipts_job_idx ON chat_ingress_receipts(job_id);

CREATE FUNCTION runtime_preserve_chat_receipt() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW IS DISTINCT FROM OLD THEN
        RAISE EXCEPTION 'chat ingress receipt is immutable' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER chat_ingress_receipts_preserve_snapshot BEFORE UPDATE ON chat_ingress_receipts
    FOR EACH ROW EXECUTE FUNCTION runtime_preserve_chat_receipt();

CREATE FUNCTION runtime_validate_chat_receipt() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    run_record agent_runs%ROWTYPE;
    job_record runtime_jobs%ROWTYPE;
BEGIN
    -- The application additionally validates ownership/generations in a short
    -- scoped_transaction. These guards establish row association, not authority.
    SELECT * INTO run_record FROM agent_runs WHERE id=NEW.run_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'chat ingress run is missing' USING ERRCODE = '23503';
    END IF;
    SELECT * INTO job_record FROM runtime_jobs WHERE id=NEW.job_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'chat ingress job is missing' USING ERRCODE = '23503';
    END IF;
    IF run_record.conversation_id IS DISTINCT FROM NEW.conversation_id
       OR run_record.kind <> 'chat' OR run_record.status <> 'queued'
       OR COALESCE(run_record.state #>> '{ingress,version}', '') <> '1'
       OR COALESCE(jsonb_typeof(run_record.state #> '{ingress,version}'), '') <> 'number'
       OR COALESCE(run_record.state #>> '{ingress,phase}', '') NOT IN ('collecting','ready')
       OR COALESCE(jsonb_typeof(run_record.state #> '{ingress,message_count}'), '') <> 'number'
       OR COALESCE((run_record.state #>> '{ingress,message_count}')::integer, -1) NOT BETWEEN 0 AND 31
       OR (run_record.state #>> '{ingress,phase}' = 'ready'
           AND (run_record.state #>> '{ingress,message_count}')::integer > 0)
       OR job_record.run_id IS DISTINCT FROM NEW.run_id
       OR job_record.handler <> 'chat.execute.v1' OR job_record.job_key <> 'chat:0'
       OR job_record.status <> 'pending' OR job_record.attempts <> 0 THEN
        RAISE EXCEPTION 'chat ingress execution association mismatch' USING ERRCODE = '23514';
    END IF;
    PERFORM 1 FROM messages WHERE id=NEW.message_id
        AND conversation_id=NEW.conversation_id AND role='user' FOR SHARE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'chat ingress source association mismatch' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER chat_ingress_receipts_validate_association BEFORE INSERT ON chat_ingress_receipts
    FOR EACH ROW EXECUTE FUNCTION runtime_validate_chat_receipt();

COMMIT;
