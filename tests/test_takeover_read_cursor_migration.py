"""Execute the exact migration against disposable session-local legacy tables."""
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlsplit

import pytest


@pytest.mark.parametrize('unread',[0,1,3,4])
def test_legacy_history_preservation_and_inconsistent_counter_blocks(unread):
    url=os.environ.get('PROACTIVE_E2E_DATABASE_URL')
    if not url:pytest.skip('isolated PostgreSQL required')
    assert urlsplit(url).hostname in {'127.0.0.1','localhost'}
    cli=shutil.which('psql')
    assert cli, 'native PostgreSQL client required'
    migration=Path('prisma/migrations/20261012220000_takeover_read_cursor/migration.sql').read_text()
    setup=f"""
        SET search_path TO pg_temp,public;
        CREATE TEMP TABLE human_takeovers(id text,conversation_id text,unread_count integer);
        CREATE TEMP TABLE messages(id text,conversation_id text,role text,created_at timestamp,metadata jsonb,content text);
        INSERT INTO human_takeovers VALUES ('case','conv',{unread});
        INSERT INTO messages SELECT n::text,'conv','user','2026-10-12'::timestamp,
            '{{"human_takeover":true}}'::jsonb,'original-'||n FROM generate_series(1,3) n;
    """
    check=f"""
        DO $$ BEGIN
          IF NOT EXISTS (SELECT 1 FROM human_takeovers WHERE user_message_seq=3 AND admin_read_seq=3-{unread} AND unread_count={unread})
          OR (SELECT count(*) FROM messages WHERE content='original-'||id AND (metadata->>'takeover_sequence')::int=id::int)<>3
          THEN RAISE EXCEPTION 'History or unread changed'; END IF;
        END $$;
        INSERT INTO messages VALUES ('4','conv','user','2026-10-13',
          '{{"human_takeover":true}}','original-4');
        UPDATE human_takeovers SET unread_count=unread_count+1 WHERE id='case';
        DO $$ BEGIN
          IF NOT EXISTS (SELECT 1 FROM human_takeovers WHERE user_message_seq=4 AND unread_count={unread}+1)
          OR NOT EXISTS (SELECT 1 FROM messages WHERE id='4' AND metadata->>'takeover_sequence'='4')
          THEN RAISE EXCEPTION 'Rolling legacy writer failed'; END IF;
        END $$;
        UPDATE human_takeovers SET unread_count=0 WHERE id='case';
        UPDATE human_takeovers SET user_message_seq=5,unread_count=1 WHERE id='case';
        DO $$ BEGIN
          IF NOT EXISTS (SELECT 1 FROM human_takeovers WHERE user_message_seq=5 AND admin_read_seq=4 AND unread_count=1)
          THEN RAISE EXCEPTION 'New writer or legacy read failed'; END IF;
        END $$;
    """
    result=subprocess.run([cli,'-Xq','-v','ON_ERROR_STOP=1',url],input=setup+migration+check,
        text=True,capture_output=True,timeout=30)
    if unread<=3:assert result.returncode==0,result.stderr
    else:assert result.returncode!=0 and 'requires reconciliation' in result.stderr
