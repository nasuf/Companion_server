from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import connect_db, disconnect_db
from app.redis_client import get_redis, close_redis
from app.neo4j_client import get_driver, close_neo4j
from app.services.graph.schema import init_graph_schema
from app.middleware import configure_logging, configure_langsmith, RequestTimingMiddleware
from app.services.prompting.store import ensure_prompt_templates
from app.services.character import ensure_default_template
from app.services.career import ensure_default_careers
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
    await ensure_prompt_templates()
    await ensure_default_template()
    await ensure_default_careers()
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
from app.api.public.health import router as health_router
from app.api.public.users import router as users_router
from app.api.public.agents import router as agents_router
from app.api.public.conversations import router as conversations_router
from app.api.public.chat import router as chat_router
from app.api.public.memories import router as memories_router
from app.api.public.emotions import router as emotions_router
from app.api.public.intimacy import router as intimacy_router
from app.api.public.boundary import router as boundary_router
from app.api.public.stickers import router as stickers_router
from app.api.realtime.ws import router as ws_router
from app.api.admin.prompts import router as admin_prompts_router
from app.api.public.auth import router as auth_router
from app.api.admin.users import router as admin_users_router
from app.api.admin.character import router as admin_character_router
from app.api.admin.career import router as admin_career_router
from app.api.admin.agents import router as admin_agents_router
from app.api.public.traces import router as traces_router

app.include_router(health_router)
app.include_router(users_router)
app.include_router(agents_router)
app.include_router(conversations_router)
app.include_router(chat_router)
app.include_router(memories_router)
app.include_router(emotions_router)
app.include_router(intimacy_router)
app.include_router(boundary_router)
app.include_router(stickers_router)
app.include_router(ws_router)
app.include_router(admin_prompts_router)
app.include_router(auth_router)
app.include_router(admin_users_router)
app.include_router(admin_character_router)
app.include_router(admin_career_router)
app.include_router(admin_agents_router)
app.include_router(traces_router)
