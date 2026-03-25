-- Introduce chat_workspaces as the isolation boundary for resettable chat state.
-- This migration is tracked by Prisma in _prisma_migrations and will execute once per database.

-- User / agent archival metadata
ALTER TABLE "users"
    ADD COLUMN IF NOT EXISTS "status" TEXT NOT NULL DEFAULT 'active',
    ADD COLUMN IF NOT EXISTS "archived_at" TIMESTAMP(3);

ALTER TABLE "ai_agents"
    ADD COLUMN IF NOT EXISTS "status" TEXT NOT NULL DEFAULT 'active',
    ADD COLUMN IF NOT EXISTS "archived_at" TIMESTAMP(3);

-- Workspace root table
CREATE TABLE IF NOT EXISTS "chat_workspaces" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "agent_id" TEXT,
    "status" TEXT NOT NULL DEFAULT 'active',
    "archived_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "chat_workspaces_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "chat_workspaces_user_id_status_idx"
    ON "chat_workspaces"("user_id", "status");
CREATE INDEX IF NOT EXISTS "chat_workspaces_agent_id_status_idx"
    ON "chat_workspaces"("agent_id", "status");

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'chat_workspaces_user_id_fkey'
    ) THEN
        ALTER TABLE "chat_workspaces"
            ADD CONSTRAINT "chat_workspaces_user_id_fkey"
            FOREIGN KEY ("user_id") REFERENCES "users"("id")
            ON DELETE RESTRICT ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'chat_workspaces_agent_id_fkey'
    ) THEN
        ALTER TABLE "chat_workspaces"
            ADD CONSTRAINT "chat_workspaces_agent_id_fkey"
            FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id")
            ON DELETE RESTRICT ON UPDATE CASCADE;
    END IF;
END $$;

-- Conversation workspace / archive fields
ALTER TABLE "conversations"
    ADD COLUMN IF NOT EXISTS "workspace_id" TEXT,
    ADD COLUMN IF NOT EXISTS "archived_at" TIMESTAMP(3);

CREATE INDEX IF NOT EXISTS "conversations_workspace_id_idx"
    ON "conversations"("workspace_id");

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'conversations_workspace_id_fkey'
    ) THEN
        ALTER TABLE "conversations"
            ADD CONSTRAINT "conversations_workspace_id_fkey"
            FOREIGN KEY ("workspace_id") REFERENCES "chat_workspaces"("id")
            ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

-- User-scoped state becomes workspace-aware
ALTER TABLE "memories_user"
    ADD COLUMN IF NOT EXISTS "workspace_id" TEXT;
CREATE INDEX IF NOT EXISTS "memories_user_workspace_id_idx"
    ON "memories_user"("workspace_id");

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'memories_user_workspace_id_fkey'
    ) THEN
        ALTER TABLE "memories_user"
            ADD CONSTRAINT "memories_user_workspace_id_fkey"
            FOREIGN KEY ("workspace_id") REFERENCES "chat_workspaces"("id")
            ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

ALTER TABLE "memories_ai"
    ADD COLUMN IF NOT EXISTS "workspace_id" TEXT;
CREATE INDEX IF NOT EXISTS "memories_ai_workspace_id_idx"
    ON "memories_ai"("workspace_id");

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'memories_ai_workspace_id_fkey'
    ) THEN
        ALTER TABLE "memories_ai"
            ADD CONSTRAINT "memories_ai_workspace_id_fkey"
            FOREIGN KEY ("workspace_id") REFERENCES "chat_workspaces"("id")
            ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

ALTER TABLE "memory_changelogs"
    ADD COLUMN IF NOT EXISTS "workspace_id" TEXT;
CREATE INDEX IF NOT EXISTS "memory_changelogs_workspace_id_idx"
    ON "memory_changelogs"("workspace_id");

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'memory_changelogs_workspace_id_fkey'
    ) THEN
        ALTER TABLE "memory_changelogs"
            ADD CONSTRAINT "memory_changelogs_workspace_id_fkey"
            FOREIGN KEY ("workspace_id") REFERENCES "chat_workspaces"("id")
            ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

ALTER TABLE "user_profiles"
    ADD COLUMN IF NOT EXISTS "workspace_id" TEXT;

DROP INDEX IF EXISTS "user_profiles_user_id_key";
CREATE INDEX IF NOT EXISTS "user_profiles_user_id_idx"
    ON "user_profiles"("user_id");
CREATE UNIQUE INDEX IF NOT EXISTS "user_profiles_workspace_id_key"
    ON "user_profiles"("workspace_id");

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'user_profiles_workspace_id_fkey'
    ) THEN
        ALTER TABLE "user_profiles"
            ADD CONSTRAINT "user_profiles_workspace_id_fkey"
            FOREIGN KEY ("workspace_id") REFERENCES "chat_workspaces"("id")
            ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

-- Backfill one workspace per existing agent so old data remains queryable.
INSERT INTO "chat_workspaces" (
    "id",
    "user_id",
    "agent_id",
    "status",
    "created_at",
    "updated_at"
)
SELECT
    md5(random()::text || clock_timestamp()::text || a."id"),
    a."user_id",
    a."id",
    CASE
        WHEN a."archived_at" IS NULL AND a."status" = 'active' THEN 'active'
        ELSE 'archived'
    END,
    a."created_at",
    COALESCE(a."updated_at", CURRENT_TIMESTAMP)
FROM "ai_agents" a
WHERE NOT EXISTS (
    SELECT 1
    FROM "chat_workspaces" w
    WHERE w."agent_id" = a."id"
);

-- Imported legacy workspaces keep pre-workspace user-scoped state visible in admin
INSERT INTO "chat_workspaces" (
    "id",
    "user_id",
    "agent_id",
    "status",
    "archived_at",
    "created_at",
    "updated_at"
)
SELECT
    md5('legacy:' || u."id"),
    u."id",
    NULL,
    'imported',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
FROM "users" u
WHERE EXISTS (
    SELECT 1 FROM "memories_user" mu WHERE mu."user_id" = u."id"
    UNION
    SELECT 1 FROM "memories_ai" ma WHERE ma."user_id" = u."id"
    UNION
    SELECT 1 FROM "memory_changelogs" mc WHERE mc."user_id" = u."id"
    UNION
    SELECT 1 FROM "user_profiles" up WHERE up."user_id" = u."id"
)
AND NOT EXISTS (
    SELECT 1
    FROM "chat_workspaces" w
    WHERE w."id" = md5('legacy:' || u."id")
);

UPDATE "conversations" c
SET "workspace_id" = w."id"
FROM "chat_workspaces" w
WHERE c."agent_id" = w."agent_id"
  AND c."workspace_id" IS NULL;

UPDATE "memories_user" mu
SET "workspace_id" = md5('legacy:' || mu."user_id")
WHERE mu."workspace_id" IS NULL;

UPDATE "memories_ai" ma
SET "workspace_id" = md5('legacy:' || ma."user_id")
WHERE ma."workspace_id" IS NULL;

UPDATE "memory_changelogs" mc
SET "workspace_id" = md5('legacy:' || mc."user_id")
WHERE mc."workspace_id" IS NULL;

UPDATE "user_profiles" up
SET "workspace_id" = md5('legacy:' || up."user_id")
WHERE up."workspace_id" IS NULL;

-- If a user already only has one active agent/workspace, move legacy user-scoped state into that workspace.
UPDATE "memories_user" mu
SET "workspace_id" = w."id"
FROM "chat_workspaces" w
WHERE mu."workspace_id" = md5('legacy:' || mu."user_id")
  AND mu."user_id" = w."user_id"
  AND w."status" = 'active'
  AND (
      SELECT COUNT(*)
      FROM "chat_workspaces" wx
      WHERE wx."user_id" = mu."user_id"
        AND wx."agent_id" IS NOT NULL
  ) = 1
  AND mu."workspace_id" IS NOT NULL;

UPDATE "memories_ai" ma
SET "workspace_id" = w."id"
FROM "chat_workspaces" w
WHERE ma."workspace_id" = md5('legacy:' || ma."user_id")
  AND ma."user_id" = w."user_id"
  AND w."status" = 'active'
  AND (
      SELECT COUNT(*)
      FROM "chat_workspaces" wx
      WHERE wx."user_id" = ma."user_id"
        AND wx."agent_id" IS NOT NULL
  ) = 1
  AND ma."workspace_id" IS NOT NULL;

UPDATE "memory_changelogs" mc
SET "workspace_id" = w."id"
FROM "chat_workspaces" w
WHERE mc."workspace_id" = md5('legacy:' || mc."user_id")
  AND mc."user_id" = w."user_id"
  AND w."status" = 'active'
  AND (
      SELECT COUNT(*)
      FROM "chat_workspaces" wx
      WHERE wx."user_id" = mc."user_id"
        AND wx."agent_id" IS NOT NULL
  ) = 1
  AND mc."workspace_id" IS NOT NULL;

UPDATE "user_profiles" up
SET "workspace_id" = w."id"
FROM "chat_workspaces" w
WHERE up."workspace_id" = md5('legacy:' || up."user_id")
  AND up."user_id" = w."user_id"
  AND w."status" = 'active'
  AND (
      SELECT COUNT(*)
      FROM "chat_workspaces" wx
      WHERE wx."user_id" = up."user_id"
        AND wx."agent_id" IS NOT NULL
  ) = 1
  AND up."workspace_id" IS NOT NULL;

-- Keep only the newest active workspace per user before adding partial unique indexes.
WITH ranked_active_workspaces AS (
    SELECT
        "id",
        ROW_NUMBER() OVER (
            PARTITION BY "user_id"
            ORDER BY "created_at" DESC, "updated_at" DESC, "id" DESC
        ) AS rn
    FROM "chat_workspaces"
    WHERE "status" = 'active'
)
UPDATE "chat_workspaces" w
SET
    "status" = 'archived',
    "archived_at" = COALESCE(w."archived_at", CURRENT_TIMESTAMP),
    "updated_at" = CURRENT_TIMESTAMP
FROM ranked_active_workspaces r
WHERE w."id" = r."id"
  AND r.rn > 1;

UPDATE "ai_agents" a
SET
    "status" = 'archived',
    "archived_at" = COALESCE(a."archived_at", CURRENT_TIMESTAMP)
FROM "chat_workspaces" w
WHERE a."id" = w."agent_id"
  AND w."status" = 'archived'
  AND a."status" <> 'archived';

UPDATE "conversations" c
SET
    "is_deleted" = TRUE,
    "archived_at" = COALESCE(c."archived_at", CURRENT_TIMESTAMP)
FROM "chat_workspaces" w
WHERE c."workspace_id" = w."id"
  AND w."status" = 'archived'
  AND c."is_deleted" = FALSE;

CREATE UNIQUE INDEX IF NOT EXISTS "chat_workspaces_user_id_active_key"
    ON "chat_workspaces"("user_id")
    WHERE "status" = 'active';
CREATE UNIQUE INDEX IF NOT EXISTS "chat_workspaces_agent_id_active_key"
    ON "chat_workspaces"("agent_id")
    WHERE "status" = 'active' AND "agent_id" IS NOT NULL;
