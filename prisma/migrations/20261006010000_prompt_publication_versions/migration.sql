BEGIN;
-- Block audit inserts during backfill so no concurrent publication is missed.
LOCK TABLE prompt_template_versions IN SHARE ROW EXCLUSIVE MODE;
-- Public Web publication numbers are independent of optimistic-lock revisions.
-- Keep every existing audit row unchanged; the new table only attaches metadata.
CREATE TABLE prompt_publication_counters (
  prompt_key TEXT PRIMARY KEY,
  last_version INTEGER NOT NULL CHECK (last_version >= 0)
);
CREATE TABLE prompt_publication_versions (
  version_id TEXT PRIMARY KEY REFERENCES prompt_template_versions(id) ON DELETE CASCADE,
  prompt_key TEXT NOT NULL,
  number INTEGER NOT NULL CHECK (number > 0),
  UNIQUE (prompt_key, number)
);
INSERT INTO prompt_publication_versions(version_id, prompt_key, number)
SELECT id, prompt_key,
       ROW_NUMBER() OVER (PARTITION BY prompt_key ORDER BY created_at, id)::INTEGER
FROM prompt_template_versions
WHERE change_type IN ('manual_save', 'reset_default') OR change_type LIKE 'restore:%';
INSERT INTO prompt_publication_counters(prompt_key, last_version)
SELECT prompt_key, MAX(number) FROM prompt_publication_versions GROUP BY prompt_key;

CREATE FUNCTION number_prompt_publication() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE next_number INTEGER;
BEGIN
  IF NEW.change_type IN ('manual_save', 'reset_default') OR NEW.change_type LIKE 'restore:%' THEN
    INSERT INTO prompt_publication_counters(prompt_key, last_version)
    VALUES (NEW.prompt_key, 1)
    ON CONFLICT (prompt_key) DO UPDATE
      SET last_version = prompt_publication_counters.last_version + 1
    RETURNING last_version INTO next_number;
    INSERT INTO prompt_publication_versions(version_id, prompt_key, number)
    VALUES (NEW.id, NEW.prompt_key, next_number);
  END IF;
  RETURN NEW;
END $$;
-- Also covers an old application client after rollback. Counter and audit share
-- the insert transaction; a failure cannot leave a published version or a gap.
CREATE TRIGGER prompt_publication_number AFTER INSERT ON prompt_template_versions
FOR EACH ROW EXECUTE FUNCTION number_prompt_publication();

COMMIT;
