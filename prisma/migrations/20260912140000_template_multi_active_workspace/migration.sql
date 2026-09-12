-- Allow the template system user to own many concurrent active workspaces.
--
-- chat_workspaces_user_id_active_key was UNIQUE(user_id) WHERE status='active',
-- encoding "one companion per user". Admin template agents all belong to the
-- reserved user __companion_template_system__ and must coexist (one is the
-- default; others are drafts / alternatives). Creating a second template
-- therefore hit UniqueViolationError on activate_workspace
-- (POST /admin-api/agent-templates).
--
-- Regular users stay constrained: the unique index now skips rows with
-- allow_multiple_active=TRUE, which only the template provisioning path sets.

ALTER TABLE "chat_workspaces"
    ADD COLUMN IF NOT EXISTS "allow_multiple_active" BOOLEAN NOT NULL DEFAULT FALSE;

UPDATE "chat_workspaces" AS w
SET "allow_multiple_active" = TRUE
FROM "users" AS u
WHERE w."user_id" = u."id"
  AND u."username" = '__companion_template_system__';

DROP INDEX IF EXISTS "chat_workspaces_user_id_active_key";

CREATE UNIQUE INDEX "chat_workspaces_user_id_active_key"
    ON "chat_workspaces"("user_id")
    WHERE "status" = 'active' AND "allow_multiple_active" = FALSE;
