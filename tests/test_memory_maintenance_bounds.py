"""Real 50k-row scan verifies bounded application allocations, not SQL strings."""
import json
import time
import tracemalloc
from uuid import uuid4
from unittest.mock import AsyncMock,MagicMock

import pytest

from app.services.memory.lifecycle import lazy_update
from tests.test_memory_lifecycle_postgres import memory_flow
from tests.test_runtime_execution_foundation import flow


async def test_fifty_thousand_rows_do_not_load_into_application(memory_flow):
    f=memory_flow
    await f.db.execute_raw("""INSERT INTO memories_user(id,user_id,workspace_id,content,level,importance,current_score,value_updated_at,main_category,sub_category,updated_at)
        SELECT $1||'-'||i::text,$2,$3,repeat('Synthetic ',100),2,0.6,0.6,NOW()-INTERVAL '60 days','生活','工作',NOW()
        FROM generate_series(1,50000) i""",str(uuid4()),f.ids["owner"],f.ids["workspace"])
    tracemalloc.start();started=time.monotonic()
    result=await lazy_update.sweep_stale_values(user_id=f.ids["owner"],sources=("user",))
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    assert result["scanned"]==1000
    assert (await f.db.query_raw("SELECT count(*)::int n FROM memories_user WHERE user_id=$1 AND value_updated_at < NOW()-INTERVAL '30 days'",f.ids["owner"]))[0]["n"]==49000
    assert peak < 20*1024*1024
    print(json.dumps({"synthetic_rows":50000,"scanned":1000,"peak_python_bytes":peak,"duration_seconds":round(time.monotonic()-started,3)}))


async def test_scheduler_decay_failure_cannot_emit_success(monkeypatch):
    from jobs import scheduler
    async def direct(name,ttl,body):await body()
    monkeypatch.setattr(scheduler,"_run_distributed_job",direct)
    monkeypatch.setattr(scheduler,"run_l2_adjustment",AsyncMock(side_effect=RuntimeError("synthetic database failure")))
    failed=MagicMock();monkeypatch.setattr(scheduler,"_job_failed",failed)
    with pytest.raises(RuntimeError,match="synthetic database failure"):
        await scheduler._run_l2_adjustment()
    failed.assert_called_once()
