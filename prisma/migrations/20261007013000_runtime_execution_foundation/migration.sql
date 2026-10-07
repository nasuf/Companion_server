-- Additive, transactional expansion; no existing rows are rewritten.
BEGIN;
SET LOCAL lock_timeout = '2s';
SET LOCAL statement_timeout = '15s';

-- CreateTable
CREATE TABLE "agent_runs" (
    "id" TEXT NOT NULL,
    "actor_user_id" TEXT NOT NULL,
    "owner_user_id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "workspace_id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "actor_generation" UUID NOT NULL,
    "owner_generation" UUID NOT NULL,
    "agent_generation" UUID NOT NULL,
    "workspace_generation" UUID NOT NULL,
    "conversation_generation" UUID NOT NULL,
    "scope_version" INTEGER NOT NULL DEFAULT 1,
    "parent_run_id" TEXT,
    "kind" TEXT NOT NULL,
    "request_key" TEXT NOT NULL,
    "input_fingerprint" TEXT NOT NULL,
    "input" JSONB NOT NULL,
    "graph_version" TEXT NOT NULL,
    "state_version" INTEGER NOT NULL,
    "config_snapshot" JSONB NOT NULL,
    "prompt_snapshot" JSONB NOT NULL,
    "budget_snapshot" JSONB NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'queued',
    "state" JSONB NOT NULL DEFAULT '{}',
    "result" JSONB NOT NULL DEFAULT '{}',
    "error" JSONB NOT NULL DEFAULT '{}',
    "deadline_at" TIMESTAMPTZ(6),
    "finished_at" TIMESTAMPTZ(6),
    "created_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ(6) NOT NULL,

    CONSTRAINT "agent_runs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "runtime_jobs" (
    "id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "job_key" TEXT NOT NULL,
    "handler" TEXT NOT NULL,
    "payload_version" INTEGER NOT NULL DEFAULT 1,
    "payload" JSONB NOT NULL,
    "queue" TEXT NOT NULL,
    "priority" INTEGER NOT NULL DEFAULT 100,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "max_attempts" INTEGER NOT NULL DEFAULT 3,
    "fencing_token" BIGINT NOT NULL DEFAULT 0,
    "lease_owner" TEXT,
    "lease_expires_at" TIMESTAMPTZ(6),
    "available_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "result" JSONB NOT NULL DEFAULT '{}',
    "error" JSONB NOT NULL DEFAULT '{}',
    "finished_at" TIMESTAMPTZ(6),
    "created_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ(6) NOT NULL,

    CONSTRAINT "runtime_jobs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "agent_actions" (
    "id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "action_key" TEXT NOT NULL,
    "idempotency_key" TEXT NOT NULL,
    "kind" TEXT NOT NULL,
    "input_fingerprint" TEXT NOT NULL,
    "input" JSONB NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'planned',
    "provider" TEXT,
    "provider_ref" TEXT,
    "result" JSONB NOT NULL DEFAULT '{}',
    "error" JSONB NOT NULL DEFAULT '{}',
    "started_at" TIMESTAMPTZ(6),
    "finished_at" TIMESTAMPTZ(6),
    "created_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ(6) NOT NULL,

    CONSTRAINT "agent_actions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "runtime_outbox" (
    "id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "event_key" TEXT NOT NULL,
    "sequence" INTEGER NOT NULL,
    "event_type" TEXT NOT NULL,
    "payload" JSONB NOT NULL,
    "message_id" TEXT,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "fencing_token" BIGINT NOT NULL DEFAULT 0,
    "lease_owner" TEXT,
    "lease_expires_at" TIMESTAMPTZ(6),
    "available_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "delivered_at" TIMESTAMPTZ(6),
    "error" JSONB NOT NULL DEFAULT '{}',
    "created_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ(6) NOT NULL,

    CONSTRAINT "runtime_outbox_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "agent_runs_conversation_created_idx" ON "agent_runs"("conversation_id", "created_at");

-- CreateIndex
CREATE INDEX "agent_runs_status_updated_idx" ON "agent_runs"("status", "updated_at");

-- CreateIndex
CREATE INDEX "agent_runs_parent_idx" ON "agent_runs"("parent_run_id");

-- CreateIndex
CREATE INDEX "agent_runs_owner_idx" ON "agent_runs"("owner_user_id");

-- CreateIndex
CREATE INDEX "agent_runs_actor_idx" ON "agent_runs"("actor_user_id");

-- CreateIndex
CREATE INDEX "agent_runs_agent_idx" ON "agent_runs"("agent_id");

-- CreateIndex
CREATE INDEX "agent_runs_workspace_idx" ON "agent_runs"("workspace_id");

-- CreateIndex
CREATE UNIQUE INDEX "agent_runs_conversation_request_key" ON "agent_runs"("conversation_id", "request_key");

-- CreateIndex
CREATE INDEX "runtime_jobs_claim_idx" ON "runtime_jobs"("queue", "status", "priority", "available_at", "id");

-- CreateIndex
CREATE INDEX "runtime_jobs_lease_idx" ON "runtime_jobs"("status", "lease_expires_at");

-- CreateIndex
CREATE UNIQUE INDEX "runtime_jobs_run_key" ON "runtime_jobs"("run_id", "job_key");

-- CreateIndex
CREATE UNIQUE INDEX "agent_actions_idempotency_key" ON "agent_actions"("idempotency_key");

-- CreateIndex
CREATE INDEX "agent_actions_reconcile_idx" ON "agent_actions"("status", "updated_at");

-- CreateIndex
CREATE UNIQUE INDEX "agent_actions_run_key" ON "agent_actions"("run_id", "action_key");

-- CreateIndex
CREATE INDEX "runtime_outbox_delivery_idx" ON "runtime_outbox"("status", "available_at", "id");

-- CreateIndex
CREATE INDEX "runtime_outbox_lease_idx" ON "runtime_outbox"("status", "lease_expires_at");

-- CreateIndex
CREATE INDEX "runtime_outbox_message_idx" ON "runtime_outbox"("message_id");

-- CreateIndex
CREATE UNIQUE INDEX "runtime_outbox_run_event_key" ON "runtime_outbox"("run_id", "event_key");

-- CreateIndex
CREATE UNIQUE INDEX "runtime_outbox_run_sequence_key" ON "runtime_outbox"("run_id", "sequence");

-- AddForeignKey
ALTER TABLE "agent_runs" ADD CONSTRAINT "agent_runs_actor_user_id_fkey" FOREIGN KEY ("actor_user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE RESTRICT;

-- AddForeignKey
ALTER TABLE "agent_runs" ADD CONSTRAINT "agent_runs_owner_user_id_fkey" FOREIGN KEY ("owner_user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE RESTRICT;

-- AddForeignKey
ALTER TABLE "agent_runs" ADD CONSTRAINT "agent_runs_agent_id_fkey" FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id") ON DELETE CASCADE ON UPDATE RESTRICT;

-- AddForeignKey
ALTER TABLE "agent_runs" ADD CONSTRAINT "agent_runs_workspace_id_fkey" FOREIGN KEY ("workspace_id") REFERENCES "chat_workspaces"("id") ON DELETE CASCADE ON UPDATE RESTRICT;

-- AddForeignKey
ALTER TABLE "agent_runs" ADD CONSTRAINT "agent_runs_conversation_id_fkey" FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id") ON DELETE CASCADE ON UPDATE RESTRICT;

-- AddForeignKey
ALTER TABLE "agent_runs" ADD CONSTRAINT "agent_runs_parent_run_id_fkey" FOREIGN KEY ("parent_run_id") REFERENCES "agent_runs"("id") ON DELETE CASCADE ON UPDATE RESTRICT;

-- AddForeignKey
ALTER TABLE "runtime_jobs" ADD CONSTRAINT "runtime_jobs_run_id_fkey" FOREIGN KEY ("run_id") REFERENCES "agent_runs"("id") ON DELETE CASCADE ON UPDATE RESTRICT;

-- AddForeignKey
ALTER TABLE "agent_actions" ADD CONSTRAINT "agent_actions_run_id_fkey" FOREIGN KEY ("run_id") REFERENCES "agent_runs"("id") ON DELETE CASCADE ON UPDATE RESTRICT;

-- AddForeignKey
ALTER TABLE "runtime_outbox" ADD CONSTRAINT "runtime_outbox_run_id_fkey" FOREIGN KEY ("run_id") REFERENCES "agent_runs"("id") ON DELETE CASCADE ON UPDATE RESTRICT;

-- AddForeignKey
ALTER TABLE "runtime_outbox" ADD CONSTRAINT "runtime_outbox_message_id_fkey" FOREIGN KEY ("message_id") REFERENCES "messages"("id") ON DELETE CASCADE ON UPDATE RESTRICT;


-- These checks/triggers are intentionally migration-managed: Prisma cannot
-- represent CHECK constraints, partial unique indexes or immutable snapshots.
ALTER TABLE agent_runs ADD CONSTRAINT agent_runs_shape CHECK (
    kind IN ('chat','task','background')
    AND status IN ('queued','running','waiting','succeeded','failed','cancelled')
    AND scope_version = 1 AND state_version > 0
    AND length(btrim(request_key)) BETWEEN 1 AND 256
    AND length(btrim(graph_version)) BETWEEN 1 AND 256
    AND input_fingerprint ~ '^[0-9a-f]{64}$'
    AND parent_run_id IS DISTINCT FROM id
    AND jsonb_typeof(input) = 'object'
    AND jsonb_typeof(config_snapshot) = 'object'
    AND jsonb_typeof(prompt_snapshot) = 'object'
    AND jsonb_typeof(budget_snapshot) = 'object'
    AND jsonb_typeof(state) = 'object' AND jsonb_typeof(result) = 'object'
    AND jsonb_typeof(error) = 'object'
    AND ((status IN ('succeeded','failed','cancelled')) = (finished_at IS NOT NULL))
);
-- Waiting runs release the primary chat slot; task/background runs may overlap.
CREATE UNIQUE INDEX agent_runs_one_running_chat
    ON agent_runs(conversation_id) WHERE kind = 'chat' AND status = 'running';

ALTER TABLE runtime_jobs ADD CONSTRAINT runtime_jobs_shape CHECK (
    status IN ('pending','running','retry','succeeded','failed','cancelled')
    AND queue IN ('foreground','background')
    AND length(btrim(job_key)) BETWEEN 1 AND 256
    AND length(btrim(handler)) BETWEEN 1 AND 256
    AND payload_version > 0 AND priority >= 0
    AND attempts >= 0 AND max_attempts > 0 AND attempts <= max_attempts
    AND fencing_token >= 0
    AND jsonb_typeof(payload) = 'object' AND jsonb_typeof(result) = 'object'
    AND jsonb_typeof(error) = 'object'
    AND ((status = 'running' AND lease_owner IS NOT NULL
          AND length(btrim(lease_owner)) BETWEEN 1 AND 256
          AND lease_expires_at IS NOT NULL AND fencing_token > 0 AND attempts > 0)
         OR (status <> 'running' AND lease_owner IS NULL AND lease_expires_at IS NULL))
    AND ((status IN ('succeeded','failed','cancelled')) = (finished_at IS NOT NULL))
);
ALTER TABLE agent_actions ADD CONSTRAINT agent_actions_shape CHECK (
    status IN ('planned','started','unknown','succeeded','failed','cancelled')
    AND length(btrim(action_key)) BETWEEN 1 AND 256
    AND length(btrim(idempotency_key)) BETWEEN 1 AND 256
    AND length(btrim(kind)) BETWEEN 1 AND 256
    AND input_fingerprint ~ '^[0-9a-f]{64}$'
    AND jsonb_typeof(input) = 'object' AND jsonb_typeof(result) = 'object'
    AND jsonb_typeof(error) = 'object'
    AND (status NOT IN ('started','unknown','succeeded') OR started_at IS NOT NULL)
    AND ((status IN ('succeeded','failed','cancelled')) = (finished_at IS NOT NULL))
);
ALTER TABLE runtime_outbox ADD CONSTRAINT runtime_outbox_shape CHECK (
    status IN ('pending','delivering','delivered','failed','cancelled')
    AND length(btrim(event_key)) BETWEEN 1 AND 256
    AND length(btrim(event_type)) BETWEEN 1 AND 256
    AND sequence >= 0 AND attempts >= 0 AND fencing_token >= 0
    AND jsonb_typeof(payload) = 'object' AND jsonb_typeof(error) = 'object'
    AND ((status = 'delivering' AND lease_owner IS NOT NULL
          AND length(btrim(lease_owner)) BETWEEN 1 AND 256
          AND lease_expires_at IS NOT NULL AND fencing_token > 0 AND attempts > 0)
         OR (status <> 'delivering' AND lease_owner IS NULL AND lease_expires_at IS NULL))
    AND ((status = 'delivered') = (delivered_at IS NOT NULL))
);

-- Mutable state may advance; identity, scope, inputs, versions and published
-- event payloads must never be silently rewritten by retries or config changes.
CREATE FUNCTION runtime_preserve_snapshot() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(NEW) - TG_ARGV) IS DISTINCT FROM (to_jsonb(OLD) - TG_ARGV) THEN
        RAISE EXCEPTION 'runtime snapshot is immutable' USING ERRCODE = '23514';
    END IF;
    IF OLD.status IN ('succeeded','failed','cancelled','delivered')
       AND NEW IS DISTINCT FROM OLD THEN
        RAISE EXCEPTION 'runtime terminal record is immutable' USING ERRCODE = '23514';
    END IF;
    IF TG_TABLE_NAME IN ('runtime_jobs','runtime_outbox') THEN
        IF NEW.fencing_token < OLD.fencing_token OR NEW.attempts < OLD.attempts THEN
            RAISE EXCEPTION 'runtime counters cannot decrease' USING ERRCODE = '23514';
        END IF;
    END IF;
    IF TG_TABLE_NAME = 'agent_actions' AND OLD.status = 'unknown'
       AND NEW.status IN ('planned','started') THEN
        RAISE EXCEPTION 'unknown action requires reconciliation' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER agent_runs_preserve_snapshot BEFORE UPDATE ON agent_runs
    FOR EACH ROW EXECUTE FUNCTION runtime_preserve_snapshot(
        'status','state','result','error','finished_at','updated_at');
CREATE TRIGGER runtime_jobs_preserve_snapshot BEFORE UPDATE ON runtime_jobs
    FOR EACH ROW EXECUTE FUNCTION runtime_preserve_snapshot(
        'status','attempts','fencing_token','lease_owner','lease_expires_at',
        'available_at','result','error','finished_at','updated_at');
CREATE TRIGGER agent_actions_preserve_snapshot BEFORE UPDATE ON agent_actions
    FOR EACH ROW EXECUTE FUNCTION runtime_preserve_snapshot(
        'status','provider_ref','result','error','started_at','finished_at','updated_at');
CREATE TRIGGER runtime_outbox_preserve_snapshot BEFORE UPDATE ON runtime_outbox
    FOR EACH ROW EXECUTE FUNCTION runtime_preserve_snapshot(
        'status','attempts','fencing_token','lease_owner','lease_expires_at',
        'available_at','delivered_at','error','updated_at');

-- Parent links cannot change after insertion. Compare the full scope, including
-- generations, so a child cannot be attached to a different/later execution.
CREATE FUNCTION runtime_validate_parent_scope() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE parent agent_runs;
BEGIN
    IF NEW.parent_run_id IS NOT NULL THEN
        SELECT * INTO parent FROM agent_runs WHERE id = NEW.parent_run_id FOR KEY SHARE;
        IF NOT FOUND OR
           ROW(NEW.actor_user_id,NEW.owner_user_id,NEW.agent_id,NEW.workspace_id,NEW.conversation_id,
               NEW.actor_generation,NEW.owner_generation,NEW.agent_generation,
               NEW.workspace_generation,NEW.conversation_generation,NEW.scope_version)
           IS DISTINCT FROM
           ROW(parent.actor_user_id,parent.owner_user_id,parent.agent_id,parent.workspace_id,parent.conversation_id,
               parent.actor_generation,parent.owner_generation,parent.agent_generation,
               parent.workspace_generation,parent.conversation_generation,parent.scope_version) THEN
            RAISE EXCEPTION 'runtime parent scope mismatch' USING ERRCODE = '23514';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER agent_runs_validate_parent BEFORE INSERT ON agent_runs
    FOR EACH ROW EXECUTE FUNCTION runtime_validate_parent_scope();

CREATE FUNCTION runtime_validate_outbox_message() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE run_conversation text; message_conversation text;
BEGIN
    IF NEW.message_id IS NOT NULL THEN
        SELECT conversation_id INTO run_conversation FROM agent_runs WHERE id = NEW.run_id FOR KEY SHARE;
        SELECT conversation_id INTO message_conversation FROM messages WHERE id = NEW.message_id FOR SHARE;
        IF run_conversation IS NULL OR message_conversation IS NULL
           OR run_conversation IS DISTINCT FROM message_conversation THEN
            RAISE EXCEPTION 'runtime outbox message scope mismatch' USING ERRCODE = '23514';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER runtime_outbox_validate_message BEFORE INSERT ON runtime_outbox
    FOR EACH ROW EXECUTE FUNCTION runtime_validate_outbox_message();

COMMIT;
