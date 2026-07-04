import asyncio
import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from prisma import Json

from app.api.jwt_auth import require_user
from app.api.ownership import (
    require_agent_owner,
    require_agent_owner_any_status,
    require_user_self,
)
from app.db import db
from app.models.agent import AgentCreate, AgentUpdate, AgentResponse, RegenerateMbtiRequest
from app.services.interaction.boundary import init_patience
from app.services.mbti import build_mbti, get_mbti, seven_dim_to_mbti
from app.services.career import pick_random_active_career
from app.services.character_generation import generate_full_profile
from app.services.agent_avatars import (
    build_cached_avatar_url,
    ensure_cached_avatar,
    pick_agent_avatar,
    warm_avatar_pool,
)
from app.services.life_story import (
    activate_agent,
    generate_l1_coverage,
    get_progress,
    set_progress,
)
from app.services.runtime.data_reset import hard_delete_agent_data
from app.services.proactive.sender import dispatch_first_greeting_for_agent
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
    create_provisioning_workspace,
    finalize_archived_workspaces,
    get_active_workspace,
    restore_staged_workspaces,
    stage_active_workspaces_for_user,
)
from app.services.runtime.job_queue import enqueue_runtime_job, register_job_handler
from app.services.runtime.tasks import fire_background

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


def _agent_response(agent, *, workspace_id: str | None = None) -> AgentResponse:
    avatar_key = getattr(agent, "avatarKey", None)
    return AgentResponse(
        id=agent.id,
        name=agent.name,
        user_id=agent.userId,
        workspace_id=workspace_id,
        mbti=get_mbti(agent),
        background=agent.background,
        values=agent.values,
        gender=agent.gender,
        city=getattr(agent, "city", None),
        life_overview=agent.lifeOverview,
        avatar_key=avatar_key,
        avatar_url=build_cached_avatar_url(avatar_key) or getattr(agent, "avatarUrl", None),
        created_at=str(agent.createdAt),
    )


async def _warm_agent_avatar_cache(avatar_key: str | None) -> None:
    if not avatar_key:
        return
    try:
        await ensure_cached_avatar(avatar_key)
    except Exception as exc:
        logger.warning("[agent-avatar] warm cache failed key=%s error=%s", avatar_key, exc)


async def warm_default_agent_avatars() -> None:
    results = await warm_avatar_pool()
    ok = sum(1 for value in results.values() if value)
    logger.info("[agent-avatar] warmed %s/%s default avatars", ok, len(results))


async def _safe_life_overview(agent) -> str | None:
    try:
        return await generate_and_save_life_overview(agent)
    except Exception as e:
        logger.warning(f"Life overview failed for {agent.id}: {e}")
        return None


async def _run_agent_initialization_job(payload: dict) -> None:
    agent_id = str(payload["agent_id"])
    user_id = str(payload["user_id"])
    workspace_id = payload.get("workspace_id")
    personality_dict = dict(payload.get("personality") or {})
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or getattr(agent, "status", None) == "archived":
        logger.info(f"Skipping agent initialization job for missing/archived agent {agent_id}")
        return

    await init_patience(agent.id, user_id)

    from app.services.llm.usage_tracker import usage_session

    async with usage_session(
        scope="agent_creation",
        conversation_id=None,
        agent_id=agent.id,
        user_id=user_id,
    ):
        await _run_agent_initialization_inner(agent, user_id, workspace_id, personality_dict)


async def _run_agent_initialization_inner(
    agent,
    user_id: str,
    workspace_id: str | None,
    personality_dict: dict,
) -> None:
    """Plan B 主管线（9 段进度）."""
    await set_progress(agent.id, "initializing", message="正在创建空间...")

    await set_progress(agent.id, "mbti_deriving", message="正在推导 MBTI 性格...")
    mbti: dict | None = None
    try:
        mbti_input = await seven_dim_to_mbti(personality_dict)
        summary = mbti_input.pop("summary", "")
        mbti = await build_mbti(mbti_input, summary=summary)
        await db.aiagent.update(
            where={"id": agent.id},
            data={"mbti": Json(mbti), "currentMbti": Json(mbti)},
        )
        agent.mbti = mbti
        agent.currentMbti = mbti
    except Exception as e:
        logger.error(f"MBTI init failed for agent {agent.id}: {e}")
    await set_progress(agent.id, "mbti_done", message="MBTI 推导完成")

    await set_progress(agent.id, "prompt_building", message="正在构建生成提示...")
    try:
        career = await pick_random_active_career()
    except Exception as e:
        logger.warning(f"Career pool query failed for {agent.id}: {e}")
        career = None

    await set_progress(agent.id, "llm_generating", message="正在生成 AI 背景...")
    try:
        profile = await generate_full_profile(
            name=agent.name,
            gender=agent.gender,
            mbti=mbti,
            personality=personality_dict,
            career_template=career,
        )
    except Exception as e:
        logger.error(f"Background generation failed for {agent.id}: {e}", exc_info=True)
        await set_progress(agent.id, "failed", message=f"生成失败: {str(e)[:200]}")
        return
    await set_progress(agent.id, "llm_done", message="背景生成完成, 正在解析...")

    identity = profile.get("identity", {}) if isinstance(profile, dict) else {}
    update_payload: dict = {}
    if career and isinstance(career.get("title"), str) and career["title"].strip():
        update_payload["occupation"] = career["title"].strip()
    city = identity.get("location")
    if isinstance(city, str) and city.strip():
        update_payload["city"] = city.strip()
    derived_age = identity.get("age")
    if isinstance(derived_age, int):
        update_payload["age"] = derived_age
    if update_payload:
        try:
            await db.aiagent.update(where={"id": agent.id}, data=update_payload)
            for k, v in update_payload.items():
                setattr(agent, k, v)
        except Exception as e:
            logger.warning(f"Persisting derived agent fields failed for {agent.id}: {e}")

    memories_failed = False

    async def _run_memories():
        nonlocal memories_failed
        try:
            stored = await generate_l1_coverage(
                agent_id=agent.id,
                user_id=user_id,
                profile=profile,
                career_template=career,
                workspace_id=workspace_id,
            )
        except Exception as e:
            memories_failed = True
            logger.error(f"Life story memories failed for {agent.id}: {e}", exc_info=True)
            await set_progress(agent.id, "failed", message=f"生成失败: {str(e)[:200]}")
            return
        if stored == 0:
            memories_failed = True
            logger.warning(f"L1 generation produced 0 memories for {agent.id} (lock held or empty profile)")
            await set_progress(agent.id, "failed", message="生成失败: 记忆库为空, 请删除重建")

    _, overview_text = await asyncio.gather(_run_memories(), _safe_life_overview(agent))

    if overview_text and not memories_failed:
        try:
            await generate_daily_schedule(
                agent.id,
                agent.name,
                mbti,
                life_overview=overview_text,
            )
        except Exception as e:
            logger.warning(f"Daily schedule init failed for agent {agent.id}: {e}")

    if not memories_failed:
        await set_progress(agent.id, "first_greeting", message="正在准备第一句问候...")
        await activate_agent(agent.id)
        try:
            await dispatch_first_greeting_for_agent(
                agent_id=agent.id,
                user_id=agent.userId,
            )
        except Exception as e:
            logger.warning(f"first_greeting dispatch failed for agent {agent.id}: {e}")
        await set_progress(agent.id, "complete", message="生成完成")


register_job_handler("agent_initialization", _run_agent_initialization_job)


async def _enqueue_agent_initialization(
    *,
    agent_id: str,
    user_id: str,
    workspace_id: str | None,
    personality: dict,
) -> None:
    payload = {
        "agent_id": agent_id,
        "user_id": user_id,
        "workspace_id": workspace_id,
        "personality": personality,
    }
    try:
        job_id = await enqueue_runtime_job(
            "agent_initialization",
            payload,
            idempotency_key=f"agent_initialization:{agent_id}",
            max_attempts=3,
        )
        logger.info(
            f"Queued agent initialization job {job_id} for {agent_id}",
            extra={"event": "runtime_job", "job_type": "agent_initialization", "job_id": job_id},
        )
    except Exception as e:
        logger.warning(
            f"Queue unavailable for agent initialization; falling back to local task: {e}",
            extra={"event": "runtime_job", "job_type": "agent_initialization", "phase": "fallback"},
        )
        fire_background(_run_agent_initialization_job(payload))


async def create_agent_with_provisioning(
    *,
    user_id: str,
    name: str,
    personality: dict,
    background: str | None = None,
    values: dict | None = None,
    gender: str | None = None,
):
    """Create an agent + workspace and enqueue the full provisioning job.

    Shared by the public create_agent endpoint and admin template creation so
    both paths yield identical agent data (persona + L1 memory + embeddings +
    schedule). Returns ``(agent, workspace)``.
    """
    # 防双击 / 重试竞态: 用户已有 provisioning agent 时拒绝新建. 否则两次并发
    # 会 (a) 重复扣 LLM 配额, (b) 后入的 stage_active_workspaces_for_user 把
    # 前一份的 workspace 归档, 造成前一份在已 archived 的 workspace 写记忆.
    pending = await db.aiagent.find_first(
        where={"userId": user_id, "status": "provisioning"},
    )
    if pending is not None:
        raise HTTPException(
            status_code=409,
            detail="已有 AI 伙伴正在生成中, 请等候完成或删除后重试",
        )
    # Per spec §1.2: 7-dim / Big Five 用户输入仅作为 MBTI 计算的临时输入,
    # 不再常驻 DB. MBTI 在 _init_mbti() 里生成并写入.
    create_data: dict = {
        "name": name,
        "user": {"connect": {"id": user_id}},
        "status": "provisioning",
        "archivedAt": None,
    }
    if background is not None:
        create_data["background"] = background
    if values is not None:
        create_data["values"] = Json(values)
    if gender is not None:
        # 前端约定已传 "male"/"female", 直接存
        create_data["gender"] = gender
    avatar = pick_agent_avatar(gender)
    create_data["avatarKey"] = avatar.key
    create_data["avatarUrl"] = avatar.url
    agent = None
    workspace = None
    try:
        agent = await db.aiagent.create(data=create_data)
        workspace = await create_provisioning_workspace(user_id, agent.id)
    except Exception:
        if agent is not None:
            await db.aiagent.update(
                where={"id": agent.id},
                data={"status": "archived", "archivedAt": datetime.now(UTC)},
            )
        raise
    staged_workspaces: list = []
    try:
        staged_workspaces = await stage_active_workspaces_for_user(user_id)
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
        if staged_workspaces:
            await restore_staged_workspaces(staged_workspaces)
        raise

    # Spec §1.3：7 维 → MBTI 4 轴用大模型推导，再 §1.4 单步 LLM 生 background。
    # P0: 长生命周期初始化进入 runtime job queue, 避免进程重启后静默丢任务。
    await set_progress(agent.id, "queued", message="已进入生成队列...")
    await _warm_agent_avatar_cache(getattr(agent, "avatarKey", None))

    await _enqueue_agent_initialization(
        agent_id=agent.id,
        user_id=user_id,
        workspace_id=workspace.id if workspace else None,
        personality=personality,
    )
    return agent, workspace


@router.post("", response_model=AgentResponse)
async def create_agent(
    data: AgentCreate,
    user: dict = Depends(require_user),
):
    if data.user_id != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your user_id")
    agent, workspace = await create_agent_with_provisioning(
        user_id=data.user_id,
        name=data.name,
        personality=data.personality.model_dump(),
        background=data.background,
        values=data.values,
        gender=data.gender,
    )
    return _agent_response(agent, workspace_id=workspace.id)


@router.get("/avatar/{avatar_key}.png")
async def get_agent_avatar(avatar_key: str):
    avatar = await ensure_cached_avatar(avatar_key)
    return Response(
        content=avatar.image_bytes,
        media_type=avatar.content_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str, agent=Depends(require_agent_owner_any_status)):
    """读 agent 详情. 用 _any_status 变体允许 status="provisioning" 阶段也能查
    (life_story 生成期 ~90s 内 agent.status 仍是 provisioning, 但前端需要拿
    agent 信息渲染 chat / inspector UI). archived 仍 404."""
    if getattr(agent, "status", "active") == "archived":
        raise HTTPException(status_code=404, detail="Agent not found")
    workspace = await get_active_workspace(agent_id=agent_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _agent_response(agent, workspace_id=workspace.id)


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str, agent=Depends(require_agent_owner_any_status)):
    """Owner-scoped hard delete for the current agent.

    This mirrors the web admin delete path by delegating to
    hard_delete_agent_data, but keeps the public endpoint scoped to the
    authenticated owner so mobile clients can reset and recreate their own
    agent without admin credentials.
    """
    if getattr(agent, "status", "active") == "archived":
        raise HTTPException(status_code=404, detail="Agent not found")
    stats = await hard_delete_agent_data(agent.id, agent.userId)
    return {"ok": True, "stats": stats}


@router.get("", response_model=list[AgentResponse])
async def list_agents(
    user_id: str = Query(...),
    _user=Depends(require_user_self),
):
    agents = await db.aiagent.find_many(where={"status": "active", "userId": user_id})
    return [_agent_response(a) for a in agents]


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    data: AgentUpdate,
    _agent=Depends(require_agent_owner),
):
    update_data = {}
    if data.name is not None:
        update_data["name"] = data.name
    # MBTI 走 POST /agents/{id}/regenerate-mbti, 不在通用 PATCH 里
    if data.background is not None:
        update_data["background"] = data.background
    if data.values is not None:
        update_data["values"] = Json(data.values)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    agent = await db.aiagent.update(where={"id": agent_id}, data=update_data)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _agent_response(agent)


@router.post("/{agent_id}/regenerate-mbti", response_model=AgentResponse)
async def regenerate_mbti(
    agent_id: str,
    data: RegenerateMbtiRequest,
    _agent=Depends(require_agent_owner),
):
    """重写 agent 的 MBTI 性格。

    body 必填: {"mbti": {EI, NS, TF, JP}} (4 个 0-100 整数)
    后端按这 4 个数字构建新 MBTI (LLM 仅用于生成 summary 文本) 并同时
    覆盖 mbti + currentMbti, trait_adjustment 累计偏移 reset.
    """
    try:
        new_mbti = await build_mbti(data.mbti.model_dump())
    except ValueError as e:
        # _validate_input raised on bad shape (e.g. unknown / missing keys
        # that slipped past Pydantic). Surface as 400 not 500.
        raise HTTPException(status_code=400, detail=str(e))

    refreshed = await db.aiagent.update(
        where={"id": agent_id},
        data={"mbti": Json(new_mbti), "currentMbti": Json(new_mbti)},
    )

    return _agent_response(refreshed)


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
            agent_id, agent.name, get_mbti(agent),
            life_overview=life_overview,
        )
    return agent, schedule


@router.get("/{agent_id}/schedule")
async def get_agent_schedule(
    agent_id: str,
    _agent=Depends(require_agent_owner),
):
    """获取Agent当日作息表。"""
    _, schedule = await _resolve_schedule(agent_id)
    return {"agent_id": agent_id, "schedule": schedule}


@router.get("/{agent_id}/schedule-history")
async def get_schedule_history(
    agent_id: str,
    days: int = 30,
    agent=Depends(require_agent_owner),
):
    """获取Agent作息历史（含生活画像）。"""
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
async def get_agent_status(
    agent_id: str,
    _agent=Depends(require_agent_owner),
):
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
async def get_provision_status(
    agent_id: str,
    agent=Depends(require_agent_owner_any_status),
):
    """获取Agent初始化进度（人生经历生成）。

    前端轮询此接口显示进度条，直到 stage=complete。
    用 _any_status 变体: provisioning 阶段也能查进度.
    """

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
