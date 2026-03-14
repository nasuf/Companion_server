from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import connect_db, disconnect_db
from app.redis_client import get_redis, close_redis
from app.neo4j_client import get_driver, close_neo4j
from app.services.graph.schema import init_graph_schema
from app.middleware import configure_logging, configure_langsmith, RequestTimingMiddleware
from jobs.scheduler import setup_scheduler, shutdown_scheduler

# Configure logging and tracing before anything else
configure_logging()
configure_langsmith()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_db()
    await get_redis()
    await get_driver()
    await init_graph_schema()
    setup_scheduler()
    yield
    # Shutdown
    shutdown_scheduler()
    await disconnect_db()
    await close_redis()
    await close_neo4j()


app = FastAPI(title="AI Companion", version="0.1.0", lifespan=lifespan)

app.add_middleware(RequestTimingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
from app.api.health import router as health_router
from app.api.users import router as users_router
from app.api.agents import router as agents_router
from app.api.conversations import router as conversations_router
from app.api.chat import router as chat_router
from app.api.memories import router as memories_router
from app.api.emotions import router as emotions_router
from app.api.intimacy import router as intimacy_router
from app.api.boundary import router as boundary_router

app.include_router(health_router)
app.include_router(users_router)
app.include_router(agents_router)
app.include_router(conversations_router)
app.include_router(chat_router)
app.include_router(memories_router)
app.include_router(emotions_router)
app.include_router(intimacy_router)
app.include_router(boundary_router)
