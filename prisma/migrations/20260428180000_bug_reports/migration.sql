-- Admin 标记 AI 回复的 bug 报告表. errorTypes 用 TEXT[] 存多选错误类型 key.
CREATE TABLE "bug_reports" (
    "id" TEXT NOT NULL,
    "message_id" TEXT NOT NULL,
    "reporter_id" TEXT NOT NULL,
    "error_types" TEXT[],
    "reason" TEXT,
    "status" TEXT NOT NULL DEFAULT 'open',
    "resolved_at" TIMESTAMP(3),
    "resolved_by_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "bug_reports_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "bug_reports_message_id_idx" ON "bug_reports"("message_id");
CREATE INDEX "bug_reports_status_created_at_idx" ON "bug_reports"("status", "created_at" DESC);

ALTER TABLE "bug_reports" ADD CONSTRAINT "bug_reports_message_id_fkey"
    FOREIGN KEY ("message_id") REFERENCES "messages"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "bug_reports" ADD CONSTRAINT "bug_reports_reporter_id_fkey"
    FOREIGN KEY ("reporter_id") REFERENCES "users"("id")
    ON DELETE RESTRICT ON UPDATE CASCADE;

ALTER TABLE "bug_reports" ADD CONSTRAINT "bug_reports_resolved_by_id_fkey"
    FOREIGN KEY ("resolved_by_id") REFERENCES "users"("id")
    ON DELETE SET NULL ON UPDATE CASCADE;
