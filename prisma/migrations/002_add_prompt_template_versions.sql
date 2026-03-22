CREATE TABLE IF NOT EXISTS "prompt_template_versions" (
    "id" TEXT NOT NULL,
    "prompt_id" TEXT NOT NULL,
    "prompt_key" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "source" TEXT NOT NULL,
    "change_type" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "prompt_template_versions_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "prompt_template_versions_prompt_id_created_at_idx"
ON "prompt_template_versions"("prompt_id", "created_at" DESC);

CREATE INDEX IF NOT EXISTS "prompt_template_versions_prompt_key_created_at_idx"
ON "prompt_template_versions"("prompt_key", "created_at" DESC);

INSERT INTO "prompt_template_versions" (
    "id",
    "prompt_id",
    "prompt_key",
    "content",
    "source",
    "change_type",
    "created_at"
)
SELECT
    gen_random_uuid()::text,
    pt."id",
    pt."key",
    pt."content",
    'db',
    'bootstrap',
    pt."updated_at"
FROM "prompt_templates" pt
WHERE NOT EXISTS (
    SELECT 1
    FROM "prompt_template_versions" pv
    WHERE pv."prompt_id" = pt."id"
);
