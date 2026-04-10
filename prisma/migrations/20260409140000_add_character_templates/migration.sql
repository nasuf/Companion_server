-- CreateTable
CREATE TABLE "character_templates" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "schema" JSONB NOT NULL,
    "defaults" TEXT,
    "status" TEXT NOT NULL DEFAULT 'active',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "character_templates_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "character_profiles" (
    "id" TEXT NOT NULL,
    "template_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "data" JSONB NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'draft',
    "agent_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "character_profiles_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "character_profiles_template_id_idx" ON "character_profiles"("template_id");

-- CreateIndex
CREATE INDEX "character_profiles_status_idx" ON "character_profiles"("status");

-- AddForeignKey
ALTER TABLE "character_profiles" ADD CONSTRAINT "character_profiles_template_id_fkey" FOREIGN KEY ("template_id") REFERENCES "character_templates"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
