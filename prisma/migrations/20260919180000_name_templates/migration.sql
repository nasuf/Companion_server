-- Agent name library for admin template creation (random-by-gender).
CREATE TABLE "name_templates" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "nickname" TEXT NOT NULL DEFAULT '',
    "gender" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'active',
    "sort_order" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "name_templates_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "name_templates_gender_name_key" ON "name_templates"("gender", "name");
CREATE INDEX "name_templates_gender_status_idx" ON "name_templates"("gender", "status");
