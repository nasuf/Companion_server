This directory stores historical SQL migrations kept for audit/reference only.

They reflect the path from the old unified `memories` table to the current split-table
design, but they are not safe to replay for new environments.

For a fresh database, use `prisma/migrations/000_baseline.sql` and then apply any newer
incremental migrations from `prisma/migrations/`.
