Active migrations for new environments live in this directory.

- `000_baseline.sql`: current production-ready baseline for bootstrapping an empty database.
- Future migrations should be incremental SQL files added after the baseline.

Historical migrations that depended on the old unified `memories` table were moved to
`prisma/migrations_archive/` so new deployments do not replay obsolete steps.
