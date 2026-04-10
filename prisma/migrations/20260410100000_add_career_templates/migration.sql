-- CreateTable
CREATE TABLE "career_templates" (
    "id" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "duties" TEXT NOT NULL,
    "outputs" TEXT NOT NULL,
    "social_value" TEXT NOT NULL,
    "clients" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'active',
    "sort_order" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "career_templates_pkey" PRIMARY KEY ("id")
);
