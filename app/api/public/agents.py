import asyncio
import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException
from prisma import Json

from app.db import db
from app.models.agent import AgentCreate, AgentUpdate, AgentResponse
from app.services.relationship.boundary import init_patience
from app.services.relationship.emotion import compute_baseline_emotion_llm, save_ai_emotion
from app.services.trait_model import get_seven_dim
from app.services.life_story import generate_full_life_story, get_progress
from app.services.schedule_domain.schedule import (
    generate_and_save_life_overview,
    generate_daily_schedule,
    get_cached_schedule,
    get_current_status,
    status_label,
    type_label,
)
from app.services.workspace.workspaces import (
    activate_workspace,
    archive_provisioning_workspace,
    archive_workspace,
    create_provisioning_workspace,
    finalize_archived_workspaces,
    get_active_workspace,
    restore_staged_workspaces,
    stage_active_workspaces_for_user,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("", response_model=AgentResponse)
async def create_agent(data: AgentCreate):
    create_data: dict = {
        "name": data.name,
        "user": {"connect": {"id": data.user_id}},
        "status": "provisioning",
        "archivedAt": None,
    }
    if data.personality is not None:
        create_data["personality"] = Json(data.personality)
    if data.background is not None:
        create_data["background"] = data.background
    if data.values is not None:
        create_data["values"] = Json(data.values)
    if data.gender is not None:
        create_data["gender"] = data.gender
    agent = None
    workspace = None
    try:
        agent = await db.aiagent.create(data=create_data)
        workspace = await create_provisioning_workspace(data.user_id, agent.id)
    except Exception:
        if agent is not None:
            await db.aiagent.update(
                where={"id": agent.id},
                data={"status": "archived", "archivedAt": datetime.now(UTC)},
            )
        raise
    try:
        staged_workspaces = await stage_active_workspaces_for_user(data.user_id)
        # status 保持 "provisioning" — 等人生经历生成完成后再设为 "active"
        workspace = await activate_workspace(workspace.id)
        await finalize_archived_workspaces(staged_workspaces)
    except Exception:
        if workspace is not None:
            await archive_provisioning_workspace(workspace.id)
        if agent is not None:
            await db.aiagent.update(
                where={"id": agent.id},
                data={"status": "archived", "archivedAt": datetime.now(UTC)},
            )
        if 'staged_workspaces' in locals():
            await restore_staged_workspaces(staged_workspaces)
        raise

    # Initialize baseline emotion from personality (small model)
    async def _init_emotion():
        try:
            baseline = await compute_baseline_emotion_llm(
                agent.personality or {},
                seven_dim=get_seven_dim(agent),
                name=agent.name,
                background=agent.background or "",
                gender=agent.gender or "",
            )
            await save_ai_emotion(agent.id, baseline)
        except Exception as e:
            logger.error(f"Baseline emotion init failed for agent {agent.id}: {e}")

    asyncio.create_task(_init_emotion())

    # Initialize patience value (Redis + DB)
    asyncio.create_task(init_patience(agent.id, data.user_id))

    # Background: life story (profile selection → memory generation) → schedule
    async def _init_and_generate_story():
        ws_id = workspace.id if workspace else None
        # Step 1: 生成人生经历 (内部会从 Profile 覆盖 age/occupation/city)
        try:
            await generate_full_life_story(
                agent_id=agent.id,
                user_id=data.user_id,
                name=agent.name,
                gender=agent.gender,
                seven_dim=get_seven_dim(agent),
                workspace_id=ws_id,
            )
        except Exception as e:
            logger.error(f"Life story generation failed for {agent.id}: {e}", exc_info=True)
            try:
                a = await db.aiagent.find_unique(where={"id": agent.id})
                if a and a.status != "active":
                    await db.aiagent.update(where={"id": agent.id}, data={"status": "active"})
            except Exception:
                pass

        # Step 2: 生成生活画像 + 作息表 (依赖 Step 1 已覆盖的 identity 字段)
        try:
            updated_agent = await db.aiagent.find_unique(where={"id": agent.id})
            if updated_agent:
                overview_data = await generate_and_save_life_overview(updated_agent)
                await generate_daily_schedule(
                    agent.id, agent.name, get_seven_dim(updated_agent),
                    life_overview=overview_data.get("description"),
                )
        except Exception as e:
            logger.warning(f"Schedule init failed for agent {agent.id}: {e}")

    asyncio.create_task(_init_and_generate_story())

    return AgentResponse(
        id=agent.id,
        name=agent.name,
        user_id=agent.userId,
        workspace_id=workspace.id,
        personality=agent.personality,
        background=agent.background,
        values=agent.values,
        gender=agent.gender,
        life_overview=agent.lifeOverview,
        created_at=str(agent.createdAt),
    )


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str):
    workspace = await get_active_workspace(agent_id=agent_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Agent not found")
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentResponse(
        id=agent.id,
        name=agent.name,
        user_id=agent.userId,
        workspace_id=workspace.id,
        personality=agent.personality,
        background=agent.background,
        values=agent.values,
        gender=agent.gender,
        life_overview=agent.lifeOverview,
        created_at=str(agent.createdAt),
    )


@router.get("", response_model=list[AgentResponse])
async def list_agents(user_id: str | None = None):
    where = {"status": "active"}
    if user_id:
        where["userId"] = user_id
    agents = await db.aiagent.find_many(where=where)
    return [
        AgentResponse(
            id=a.id,
            name=a.name,
            user_id=a.userId,
            workspace_id=None,
            personality=a.personality,
            background=a.background,
            values=a.values,
            gender=a.gender,
            life_overview=a.lifeOverview,
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
        gender=agent.gender,
        life_overview=agent.lifeOverview,
        created_at=str(agent.createdAt),
    )


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    """归档 Agent 当前工作区，并清理运行时状态。"""
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    workspace = await get_active_workspace(agent_id=agent_id)
    if workspace:
        stats = await archive_workspace(workspace.id)
    else:
        archived_at = datetime.now(UTC)
        await db.aiagent.update(
            where={"id": agent_id},
            data={"status": "archived", "archivedAt": archived_at},
        )
        stats = {"agent_id": agent_id, "workspace_id": None, "runtime": {}}

    return {"ok": True, "stats": stats}


async def _resolve_schedule(agent_id: str):
    """获取Agent和作息表，不存在则404。"""
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or getattr(agent, "status", "active") != "active":
        raise HTTPException(status_code=404, detail="Agent not found")
    schedule = await get_cached_schedule(agent_id)
    if not schedule:
        life_overview = None
        if isinstance(agent.lifeOverview, dict):
            life_overview = agent.lifeOverview.get("description")
        schedule = await generate_daily_schedule(
            agent_id, agent.name, get_seven_dim(agent),
            life_overview=life_overview,
        )
    return agent, schedule


@router.get("/{agent_id}/schedule")
async def get_agent_schedule(agent_id: str):
    """获取Agent当日作息表。"""
    _, schedule = await _resolve_schedule(agent_id)
    return {"agent_id": agent_id, "schedule": schedule}


@router.get("/{agent_id}/schedule-history")
async def get_schedule_history(agent_id: str, days: int = 30):
    """获取Agent作息历史（含生活画像）。"""
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or getattr(agent, "status", "active") != "active":
        raise HTTPException(status_code=404, detail="Agent not found")

    since = datetime.now(UTC) - timedelta(days=days)
    records = await db.aidailyschedule.find_many(
        where={"agentId": agent_id, "date": {"gte": since}},
        order={"date": "desc"},
    )
    return {
        "life_overview": agent.lifeOverview,
        "schedules": [
            {"date": str(r.date.date()), "schedule": r.scheduleData}
            for r in records
        ],
    }


@router.get("/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """获取Agent当前状态。"""
    _, schedule = await _resolve_schedule(agent_id)
    status = get_current_status(schedule)
    return {
        "agent_id": agent_id,
        **status,
        "status_label": status_label(str(status.get("status", ""))),
        "type_label": type_label(str(status.get("type", ""))),
    }


@router.get("/{agent_id}/provision-status")
async def get_provision_status(agent_id: str):
    """获取Agent初始化进度（人生经历生成）。

    前端轮询此接口显示进度条，直到 stage=complete。
    """
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # 已激活 → 直接返回完成
    if agent.status == "active":
        return {
            "agent_id": agent_id,
            "status": "active",
            "stage": "complete",
            "percent": 100,
            "message": "初始化完成",
        }

    # 查 Redis 进度
    progress = await get_progress(agent_id)
    if progress:
        return {
            "agent_id": agent_id,
            "status": agent.status,
            **progress,
        }

    # 刚创建，还没开始
    return {
        "agent_id": agent_id,
        "status": agent.status,
        "stage": "initializing",
        "percent": 0,
        "message": "正在初始化...",
    }
