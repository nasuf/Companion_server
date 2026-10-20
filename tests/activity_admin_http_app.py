"""Disposable cross-repo browser host: no scheduler, generators or production IO."""

import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI

from app.api.admin.activities import router
from app.db import db
from evals.memory_baseline.safety import loopback_network_fence

assert urlsplit(os.environ["DATABASE_URL"]).hostname in {"127.0.0.1", "localhost"}
assert urlsplit(os.environ["DATABASE_URL"]).path == "/companion_proactive_e2e"


@asynccontextmanager
async def lifespan(_app):
    with loopback_network_fence():
        await db.connect()
        try:
            yield
        finally:
            await db.disconnect()


app = FastAPI(lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health():
    return {"ok": True}
