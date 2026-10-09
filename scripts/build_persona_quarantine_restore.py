"""Generate a reviewed Prisma data migration to reverse the MEMLOC quarantine.

No database connection, no execution. Place the output in a NEW Prisma migration
only when restoring the reviewed rows is intended; never edit applied history.
Rows changed since quarantine are deliberately excluded by compare-and-swap.
"""
from pathlib import Path
import argparse

RESTORE_SQL = """-- Restore only unchanged rows archived by MEMLOC-20261009.
BEGIN;
SET LOCAL lock_timeout = '3s';
SET LOCAL statement_timeout = '15s';
WITH locked AS (
    SELECT m.*, c.old_value AS original_snapshot
    FROM memories_ai m
    JOIN memory_changelogs c ON c.memory_id=m.id
      AND c.id='persona-quarantine-20261009-' || m.id
      AND c.operation='persona_quarantine'
      AND c.user_id=m.user_id AND c.workspace_id=m.workspace_id
    WHERE m.is_archived AND m.updated_at=c.created_at
      AND c.old_value::jsonb->>'is_archived'='false'
      AND (to_jsonb(m)-'is_archived'-'updated_at') =
          (c.old_value::jsonb-'is_archived'-'updated_at')
    FOR UPDATE OF m
), audit AS (
    INSERT INTO memory_changelogs(id,user_id,workspace_id,memory_id,operation,old_value,new_value,created_at)
    SELECT 'persona-quarantine-restored-20261009-' || id,user_id,workspace_id,id,
           'persona_quarantine_restored',
           (to_jsonb(locked)-'original_snapshot')::text,
           '{"is_archived":false,"batch":"MEMLOC-20261009","reason":"reviewed_restore"}',CURRENT_TIMESTAMP
    FROM locked RETURNING memory_id
)
UPDATE memories_ai m SET is_archived=FALSE,updated_at=CURRENT_TIMESTAMP
FROM audit WHERE m.id=audit.memory_id;
COMMIT;
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Exclusive creation prevents accidentally rewriting an applied migration.
    with args.output.open('x') as output:
        output.write(RESTORE_SQL)
    print('Restore migration generated; no database operations performed.')


if __name__ == '__main__':
    main()
