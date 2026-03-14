import asyncio
import logging

from fastapi import APIRouter, HTTPException
from prisma import Json

from app.db import db
from app.models.agent import AgentCreate, AgentUpdate, AgentResponse
from app.services.memory.self_memory import generate_initial_self_memories

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("", response_model=AgentResponse)
async def create_agent(data: AgentCreate):
    create_data: dict = {
        "name": data.name,
        "user": {"connect": {"id": data.user_id}},
    }
    if data.personality is not None:
        create_data["personality"] = Json(data.personality)
    if data.background is not None:
        create_data["background"] = data.background
    if data.values is not None:
        create_data["values"] = Json(data.values)
    agent = await db.aiagent.create(data=create_data)

    # Generate initial self-memories in background
    task = asyncio.create_task(
        generate_initial_self_memories(
            agent_id=agent.id,
            agent_name=agent.name,
            personality=agent.personality or {},
            user_id=data.user_id,
        )
    )
    task.add_done_callback(
        lambda t: logger.error(f"Initial self-memory generation failed: {t.exception()}")
        if not t.cancelled() and t.exception() else None
    )

    return AgentResponse(
        id=agent.id,
        name=agent.name,
        user_id=agent.userId,
        personality=agent.personality,
        background=agent.background,
        values=agent.values,
        created_at=str(agent.createdAt),
    )


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str):
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentResponse(
        id=agent.id,
        name=agent.name,
        user_id=agent.userId,
        personality=agent.personality,
        background=agent.background,
        values=agent.values,
        created_at=str(agent.createdAt),
    )


@router.get("", response_model=list[AgentResponse])
async def list_agents(user_id: str | None = None):
    where = {"userId": user_id} if user_id else {}
    agents = await db.aiagent.find_many(where=where)
    return [
        AgentResponse(
            id=a.id,
            name=a.name,
            user_id=a.userId,
            personality=a.personality,
            background=a.background,
            values=a.values,
            created_at=str(a.createdAt),
        )
        for a in agents
    ]


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: str, data: AgentUpdate):
    update_data = {}
    if data.name is not None:
        update_data["name"] = data.name
    if data.personality is not None:
        update_data["personality"] = Json(data.personality)
    if data.background is not None:
        update_data["background"] = data.background
    if data.values is not None:
        update_data["values"] = Json(data.values)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    agent = await db.aiagent.update(where={"id": agent_id}, data=update_data)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentResponse(
        id=agent.id,
        name=agent.name,
        user_id=agent.userId,
        personality=agent.personality,
        background=agent.background,
        values=agent.values,
        created_at=str(agent.createdAt),
    )
