BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
DO $$ BEGIN
  IF EXISTS (
    SELECT 1 FROM human_takeovers t WHERE t.unread_count > (
      SELECT count(*) FROM messages m WHERE m.conversation_id=t.conversation_id
        AND m.role='user' AND m.metadata->>'human_takeover'='true'
    )
  ) THEN RAISE EXCEPTION 'Takeover unread history requires reconciliation'; END IF;
END $$;
ALTER TABLE human_takeovers
  ADD COLUMN user_message_seq BIGINT NOT NULL DEFAULT 0,
  ADD COLUMN admin_read_seq BIGINT NOT NULL DEFAULT 0;

-- Preserve the existing unread count while assigning immutable, per-case
-- delivery positions. Content and memory records are untouched.
WITH ranked AS (
  SELECT m.id, row_number() OVER (PARTITION BY t.id ORDER BY m.created_at, m.id) AS seq
  FROM messages m JOIN human_takeovers t ON t.conversation_id=m.conversation_id
  WHERE m.role='user' AND m.metadata->>'human_takeover'='true'
)
UPDATE messages m SET metadata=m.metadata || jsonb_build_object('takeover_sequence', r.seq)
FROM ranked r WHERE m.id=r.id;
WITH totals AS (
  SELECT t.id, count(m.id)::bigint AS total
  FROM human_takeovers t LEFT JOIN messages m ON m.conversation_id=t.conversation_id
    AND m.role='user' AND m.metadata->>'human_takeover'='true'
  GROUP BY t.id
)
UPDATE human_takeovers t SET user_message_seq=x.total,
  admin_read_seq=x.total-t.unread_count
FROM totals x WHERE t.id=x.id;

-- During a rolling deploy (or rollback), the previous release only updates
-- unread_count. Bridge that release's two known operations under the same row
-- lock, so the additive migration does not break its message transactions.
CREATE FUNCTION takeover_legacy_read_cursor() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.user_message_seq=OLD.user_message_seq AND NEW.admin_read_seq=OLD.admin_read_seq
     AND NEW.unread_count<>OLD.unread_count THEN
    IF NEW.unread_count=0 THEN
      NEW.admin_read_seq := NEW.user_message_seq;
    ELSIF NEW.unread_count=OLD.unread_count+1 THEN
      NEW.user_message_seq := OLD.user_message_seq+1;
      UPDATE messages SET metadata=metadata || jsonb_build_object('takeover_sequence',NEW.user_message_seq)
      WHERE id=(SELECT id FROM messages WHERE conversation_id=NEW.conversation_id
        AND role='user' AND metadata->>'human_takeover'='true'
        AND NOT metadata ? 'takeover_sequence' ORDER BY created_at DESC,id DESC LIMIT 1);
      IF NOT FOUND THEN RAISE EXCEPTION 'Missing legacy takeover message'; END IF;
    ELSE
      RAISE EXCEPTION 'Unsupported legacy takeover unread update';
    END IF;
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER human_takeovers_legacy_read_cursor BEFORE UPDATE ON human_takeovers
  FOR EACH ROW EXECUTE FUNCTION takeover_legacy_read_cursor();
ALTER TABLE human_takeovers ADD CONSTRAINT human_takeovers_read_cursor_check
  CHECK (admin_read_seq >= 0 AND user_message_seq >= admin_read_seq
    AND unread_count = user_message_seq-admin_read_seq);
COMMIT;
