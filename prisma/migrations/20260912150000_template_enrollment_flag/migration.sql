-- Template enrollment flag: many templates can be open at once for new-user
-- matching. Stopping a template must NOT archive it (that would hide it from
-- admin and historically cascaded sibling-archive bugs) and must NOT touch
-- already-cloned user agents.
--
-- status='active' still means "fully provisioned". template_enabled=FALSE
-- means "do not clone this onto new signups". Default TRUE so existing
-- provisioned templates stay in the pool; archived leftovers are closed.

ALTER TABLE "ai_agents"
    ADD COLUMN IF NOT EXISTS "template_enabled" BOOLEAN NOT NULL DEFAULT TRUE;

UPDATE "ai_agents" AS a
SET "template_enabled" = FALSE
FROM "users" AS u
WHERE a."user_id" = u."id"
  AND u."username" = '__companion_template_system__'
  AND a."status" <> 'active';
