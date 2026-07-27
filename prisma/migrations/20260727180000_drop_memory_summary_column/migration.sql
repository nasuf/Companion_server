-- Drop the redundant `summary` column from both memory tables. Production data
-- proves it never carried information of its own: across all 8181 rows zero had
-- a summary differing from content and zero were NULL. `content` is now the
-- single source of truth for memory text.
-- No index or constraint depends on the column, so these are metadata-only
-- ALTERs with no table rewrite. Deployment is stop server → migrate deploy →
-- start server, so no running code ever meets the table without the column.
ALTER TABLE "memories_user" DROP COLUMN "summary";
ALTER TABLE "memories_ai" DROP COLUMN "summary";
