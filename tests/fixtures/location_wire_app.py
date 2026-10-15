"""Loopback location contract fixture: real admission/queue/storage, fixed providers."""

from unittest.mock import AsyncMock

from tests.fixtures import admin_takeover_wire_app as base
from app.api.admin import runtime_jobs, memory_repairs
from app.services import location_memory
from app.services.runtime import job_queue

app = base.app
app.include_router(runtime_jobs.router)
app.include_router(memory_repairs.router)
base.ws.get_cached_schedule = AsyncMock(return_value={"synthetic": True})
base.ws.get_current_status = lambda _: {
    "status": "idle",
    "activity": "休息",
    "type": "leisure",
}
base.ws.build_reply_timing_context = AsyncMock(return_value={})
# The model boundary is replaced; admission, message planning, ACK and recovery
# use application code and actual PostgreSQL/Redis. No provider calls permitted.
base.ws._queue_reply_or_error = AsyncMock()
location_memory.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
job_queue.get_redis = base.get_redis
for name in (
    "_READY_KEY",
    "_DELAYED_KEY",
    "_RUNNING_KEY",
    "_DLQ_KEY",
    "_SUCCEEDED_KEY",
    "_JOB_KEY_PREFIX",
    "_IDEMP_KEY_PREFIX",
):
    setattr(job_queue, name, "location-wire:" + getattr(job_queue, name))
job_queue.register_job_handler(
    location_memory.JOB_TYPE, location_memory.write_location_share_memories
)


@app.post("/__test__/recover-location")
async def recover():
    dispatched = await location_memory.dispatch_location_events(limit=10)
    processed = await job_queue.process_runtime_jobs(10)
    return {
        "dispatched": dispatched,
        "processed": processed,
        "reply_calls": base.ws._queue_reply_or_error.await_count,
    }
