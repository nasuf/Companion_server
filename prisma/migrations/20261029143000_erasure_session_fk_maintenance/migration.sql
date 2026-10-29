BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- Account erasure marks the owner 'deleting' before it removes rows. PostgreSQL
-- then clears conversation/workspace/agent pointers with ON DELETE SET NULL.
-- Those updates hit guard_erasure_owner_active and used to abort the deletion,
-- leaving the account stuck in 'deleting'. The erasure transaction sets
-- app.account_erasure_user for itself only. Updates that keep the same owner,
-- and inserts of that owner's own rows, are allowed. Clearing or moving
-- user_id is not, so a SET NULL on the owner column cannot leave an anonymous
-- copy behind. Sessions that do not set the GUC still receive Account unavailable.
CREATE OR REPLACE FUNCTION guard_erasure_owner_active() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE binding record; resource text; bound text; erasure_user text;
BEGIN
  erasure_user := current_setting('app.account_erasure_user', true);
  IF erasure_user IS NOT NULL AND erasure_user <> '' THEN
    IF TG_OP = 'UPDATE'
       AND OLD.user_id IS NOT DISTINCT FROM erasure_user
       AND NEW.user_id IS NOT DISTINCT FROM OLD.user_id THEN
      RETURN NEW;
    ELSIF TG_OP = 'INSERT' AND NEW.user_id IS NOT DISTINCT FROM erasure_user THEN
      RETURN NEW;
    END IF;
  END IF;
  IF TG_OP = 'UPDATE' AND OLD.user_id IS DISTINCT FROM NEW.user_id AND OLD.user_id IS NOT NULL THEN
    PERFORM 1 FROM users WHERE id = OLD.user_id AND status = 'deleting' FOR SHARE;
    IF FOUND THEN RAISE EXCEPTION 'Account unavailable' USING ERRCODE = '23503'; END IF;
  END IF;
  IF NEW.user_id IS NOT NULL THEN
    PERFORM 1 FROM users WHERE id = NEW.user_id AND status <> 'deleting' FOR SHARE;
    IF NOT FOUND THEN RAISE EXCEPTION 'Account unavailable' USING ERRCODE = '23503'; END IF;
  ELSE
    -- Nullable system measurements may still contain a private resource ID.
    -- Real UUID/hex resources must exist; global pseudo-conversation labels
    -- remain valid, as they do for the Trace root guard.
    FOR binding IN SELECT * FROM (VALUES ('conversation_id', 'conversations'),
        ('workspace_id', 'chat_workspaces'), ('agent_id', 'ai_agents')) AS v(field_name, table_name) LOOP
      resource := to_jsonb(NEW) ->> binding.field_name;
      IF resource ~ '^[0-9a-fA-F-]{32,36}$' THEN
        EXECUTE format('SELECT r.id FROM public.%I r JOIN users u ON u.id = r.user_id WHERE r.id = $1 AND u.status <> $2 FOR SHARE OF r, u', binding.table_name)
          INTO bound USING resource, 'deleting';
        IF bound IS NULL THEN RAISE EXCEPTION 'Account resource unavailable' USING ERRCODE = '23503'; END IF;
      END IF;
    END LOOP;
  END IF;
  RETURN NEW;
END $$;

COMMIT;
