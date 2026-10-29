BEGIN;
SET LOCAL lock_timeout='5s';
-- Quality workers write only these tables plus unowned llm_usage/notification receipts.
CREATE TABLE quality_jobs (
 id TEXT PRIMARY KEY,
 kind TEXT NOT NULL CHECK(kind IN ('daily','prompt','routing','nightly','calibration')),
 source_key TEXT NOT NULL,
 source_version TEXT NOT NULL,
 local_date DATE NOT NULL,
 status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','running','completed','failed','budget_exhausted','not_configured','skipped','interrupted')),
 not_before TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 lease_token TEXT,
 lease_until TIMESTAMPTZ,
 snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
 report JSONB NOT NULL DEFAULT '{}'::jsonb,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(kind,source_key,source_version)
);
CREATE INDEX quality_jobs_pending_idx ON quality_jobs(not_before) WHERE status='pending';
CREATE INDEX quality_jobs_date_idx ON quality_jobs(local_date,kind,created_at);
CREATE TABLE quality_budget_months (
 month DATE PRIMARY KEY,
 committed_cny NUMERIC(14,8) NOT NULL DEFAULT 0 CHECK(committed_cny>=0)
);
-- Every request reserves a proven upper bound BEFORE contacting a provider.
-- Unknown responses retain their reservation; a timeout is not a free request.
CREATE TABLE quality_charges (
 id TEXT PRIMARY KEY,
 job_id TEXT NOT NULL REFERENCES quality_jobs(id),
 month DATE NOT NULL REFERENCES quality_budget_months(month),
 local_date DATE NOT NULL,
 model TEXT NOT NULL,
 reserved_cny NUMERIC(14,8) NOT NULL CHECK(reserved_cny>0),
 cost_cny NUMERIC(14,8),
 status TEXT NOT NULL DEFAULT 'reserved' CHECK(status IN ('reserved','settled','unknown')),
 usage JSONB,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 CHECK(cost_cny IS NULL OR cost_cny BETWEEN 0 AND reserved_cny)
);
CREATE INDEX quality_charges_job_idx ON quality_charges(job_id);
CREATE INDEX quality_charges_day_idx ON quality_charges(local_date);
CREATE TABLE quality_samples (
 id TEXT PRIMARY KEY,
 job_id TEXT NOT NULL REFERENCES quality_jobs(id) ON DELETE CASCADE,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
 conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE ON UPDATE CASCADE,
 message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE ON UPDATE CASCADE,
 reply_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE ON UPDATE CASCADE,
 trace_id TEXT,
 internal BOOLEAN NOT NULL,
 status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','judged','unknown','budget_exhausted')),
 verdict JSONB,
 review_queue BOOLEAN NOT NULL DEFAULT FALSE,
 review_status TEXT NOT NULL DEFAULT 'unreviewed' CHECK(review_status IN ('unreviewed','confirmed','false_positive','uncertain')),
 review_category TEXT,
 review_note TEXT,
 reviewed_by TEXT REFERENCES users(id) ON DELETE SET NULL ON UPDATE CASCADE,
 reviewed_at TIMESTAMPTZ,
 revision INTEGER NOT NULL DEFAULT 0,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(job_id,message_id)
);
CREATE INDEX quality_samples_review_idx ON quality_samples(job_id,review_queue,review_status);
CREATE INDEX quality_samples_user_idx ON quality_samples(user_id);
CREATE TABLE quality_cases (
 id TEXT PRIMARY KEY,
 sample_id TEXT UNIQUE REFERENCES quality_samples(id) ON DELETE CASCADE,
 case_data JSONB NOT NULL,
 enabled BOOLEAN NOT NULL DEFAULT TRUE,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE quality_results (
 job_id TEXT NOT NULL REFERENCES quality_jobs(id) ON DELETE CASCADE,
 case_id TEXT NOT NULL,
 repetition INTEGER NOT NULL,
 result JSONB NOT NULL,
 PRIMARY KEY(job_id,case_id,repetition)
);
-- One cluster-wide executor, shared by API workers and the CI lease client.
CREATE TABLE quality_executor (
 id TEXT PRIMARY KEY CHECK(id='primary'),
 job_id TEXT REFERENCES quality_jobs(id),
 lease_token TEXT,
 lease_until TIMESTAMPTZ
);
INSERT INTO quality_executor(id) VALUES('primary');
COMMIT;
