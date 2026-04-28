-- 本地 trace 镜像表. 详见 schema.prisma 注释.
CREATE TABLE "message_traces" (
    "trace_id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "message_id" TEXT,
    "root_message" TEXT,
    "total_duration_ms" INTEGER,
    "total_tokens" INTEGER,
    "llm_step_count" INTEGER,
    "steps_json" JSONB NOT NULL,
    "summary_json" JSONB NOT NULL,
    "share_status" TEXT NOT NULL DEFAULT 'pending',
    "share_url" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "message_traces_pkey" PRIMARY KEY ("trace_id")
);

CREATE INDEX "message_traces_conversation_id_created_at_idx" ON "message_traces"("conversation_id", "created_at" DESC);

CREATE INDEX "message_traces_message_id_idx" ON "message_traces"("message_id");
