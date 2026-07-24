-- CreateTable
CREATE TABLE "speech_usage" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "display_mode" TEXT NOT NULL,
    "duration_seconds" INTEGER NOT NULL,
    "model" TEXT,
    "request_id" TEXT,
    "source" TEXT NOT NULL DEFAULT 'live',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "speech_usage_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "speech_usage_display_mode_created_at_idx" ON "speech_usage"("display_mode", "created_at");

-- CreateIndex
CREATE INDEX "speech_usage_user_id_created_at_idx" ON "speech_usage"("user_id", "created_at");
