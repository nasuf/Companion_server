-- 每通实时语音通话一行诊断计数 (VOICE01.01)。只存计数/耗时/分位数, 不存通话原文。
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

CREATE TABLE realtime_call_sessions (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id    TEXT REFERENCES chat_workspaces(id) ON DELETE SET NULL,
    conversation_id TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    platform        TEXT NOT NULL DEFAULT 'unknown',
    model           TEXT NOT NULL DEFAULT '',
    started_at      TIMESTAMPTZ NOT NULL,
    ended_at        TIMESTAMPTZ NOT NULL,
    duration_s      DOUBLE PRECISION NOT NULL DEFAULT 0,
    stats           JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX realtime_call_sessions_started_idx ON realtime_call_sessions(started_at DESC);
CREATE INDEX realtime_call_sessions_user_idx ON realtime_call_sessions(user_id, started_at DESC);

COMMIT;
