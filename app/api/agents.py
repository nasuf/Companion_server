import asyncio
import logging

from fastapi import APIRouter, HTTPException
from prisma import Json

from app.db import db
from app.models.agent import AgentCreate, AgentUpdate, AgentResponse
from app.services.memory.self_memory import generate_initial_self_memories
from app.services.emotion import compute_baseline_emotion, save_ai_emotion
from app.services.schedule import generate_life_overview, generate_daily_schedule, get_cached_schedule, get_current_status, save_life_overview

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

    # Initialize baseline emotion from personality
    baseline = compute_baseline_emotion(agent.personality or {})
    emo_task = asyncio.create_task(save_ai_emotion(agent.id, baseline))
    emo_task.add_done_callback(
        lambda t: logger.error(f"Baseline emotion init failed: {t.exception()}")
        if not t.cancelled() and t.exception() else None
    )

    # Generate life overview and initial schedule in background
    async def _init_schedule():
        try:
            overview = await generate_life_overview(agent.name, agent.personality or {})
            await generate_daily_schedule(agent.id, agent.name, agent.personality or {}, overview)
            await save_life_overview(agent.id, overview)
        except Exception as e:
            logger.warning(f"Schedule init failed for agent {agent.id}: {e}")

    sched_task = asyncio.create_task(_init_schedule())
    sched_task.add_done_callback(
        lambda t: logger.error(f"Schedule init failed: {t.exception()}")
        if not t.cancelled() and t.exception() else None
    )

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


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    """删除 Agent 及其所有关联数据（对话、记忆、画像、缓存等）。"""
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    from app.services.data_reset import reset_agent_data

    stats = await reset_agent_data(agent_id, agent.userId)

    # 最后删除 agent 本身
    await db.aiagent.delete(where={"id": agent_id})

    return {"ok": True, "stats": stats}


async def _resolve_schedule(agent_id: str):
    """获取Agent和作息表，不存在则404。"""
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    schedule = await get_cached_schedule(agent_id)
    if not schedule:
        schedule = await generate_daily_schedule(
            agent_id, agent.name, agent.personality or {}
        )
    return agent, schedule


@router.get("/{agent_id}/schedule")
async def get_agent_schedule(agent_id: str):
    """获取Agent当日作息表。"""
    _, schedule = await _resolve_schedule(agent_id)
    return {"agent_id": agent_id, "schedule": schedule}


@router.get("/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """获取Agent当前状态。"""
    _, schedule = await _resolve_schedule(agent_id)
    status = get_current_status(schedule)
    return {"agent_id": agent_id, **status}
