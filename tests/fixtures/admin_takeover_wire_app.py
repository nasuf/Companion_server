"""Loopback-only wire fixture using real routes, SQL and Redis.

No app lifespan/cron/provider calls. Model entry is a test failure marker.
Never include this application in production routing.
"""
from contextlib import asynccontextmanager
import os
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

from fastapi import FastAPI
from redis.asyncio import Redis

from app.config import settings
from app.db import db
from app.api import deps
from app.api.public import chat, conversations, daily_share
from app.api.realtime import admin_inbox, ws
from app.api.admin import takeovers
from app.services.admin_takeover import service
from app.services.runtime import ws_auth, ws_manager
from app.services.notifications import service as notifications

for name in ('DATABASE_URL','REDIS_URL'):
    assert urlsplit(os.environ[name]).hostname in {'127.0.0.1','localhost'}
settings.redis_url=os.environ['REDIS_URL']
redis=Redis.from_url(settings.redis_url,decode_responses=True)

async def get_redis():return redis

service.get_redis=ws_auth.get_redis=ws_manager.get_redis=get_redis
deps.is_redis_healthy=ws.is_redis_healthy=admin_inbox.is_redis_healthy=lambda:True
ws.send_first_greeting=AsyncMock(return_value=False)
ws.record_ws_online=ws.remove_ws_online=AsyncMock()
ws.fire_background=service.fire_background=lambda coro:coro.close()
notifications.notify_agent_message_created=AsyncMock()
# Manual ingress uses the real wallet quota implementation too. Only mark
# ordinary reply preparation, which a claimed message must never reach.
ws.get_cached_schedule=AsyncMock(side_effect=AssertionError('schedule/model preparation entered'))

@asynccontextmanager
async def lifespan(_):
    await db.connect();await redis.ping();await ws_manager.manager.start_subscriber()
    try:yield
    finally:
        await ws_manager.manager.stop_subscriber();await redis.aclose();await db.disconnect()

app=FastAPI(lifespan=lifespan)
for router in (chat.router,ws.router,takeovers.router,admin_inbox.router,conversations.router,daily_share.router):app.include_router(router)

@app.get('/health')
async def health():return {'status':'ok'}
