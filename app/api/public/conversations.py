from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import settings
from app.api.jwt_auth import require_user
from app.api.ownership import require_conversation_owner, require_user_self
from app.db import db
from app.models.conversation import ConversationCreate, ConversationResponse
from app.models.message import MessageResponse
from app.services.achievements.definitions import ACHIEVEMENT_BY_ID
from app.services.achievements.mode import achievement_alerts_enabled
from app.services.achievements.rule_registry import ACHIEVEMENT_RULES
from app.services.chat.crisis_state import get_crisis_care_status
from app.services.schedule_domain.schedule import (
    get_cached_schedule,
    get_current_status,
    status_label,
)
from app.services.workspace.workspaces import ensure_workspace, get_workspace_by_id

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _parse_created_at(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _interaction_days(created_at) -> int | None:
    created = _parse_created_at(created_at)
    if created is None:
        return None
    return max(1, (datetime.now(UTC).date() - created.astimezone(UTC).date()).days + 1)


async def _agent_name(agent_id: str | None) -> str:
    if not agent_id:
        return "我"
    try:
        agent = await db.aiagent.find_unique(where={"id": agent_id})
    except Exception:
        return "我"
    return (getattr(agent, "name", None) or "我").strip() or "我"


async def _current_ai_status(agent_id: str | None) -> dict | None:
    if not agent_id:
        return None
    schedule = None
    try:
        schedule = await get_cached_schedule(agent_id)
    except Exception:
        schedule = None
    if not schedule:
        try:
            local_now = datetime.now(ZoneInfo(settings.schedule_timezone))
            date_only = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
            row = await db.aidailyschedule.find_unique(
                where={"agentId_date": {"agentId": agent_id, "date": date_only}}
            )
            schedule = getattr(row, "scheduleData", None) if row else None
        except Exception:
            schedule = None
    if not schedule:
        return None
    status = get_current_status(schedule)
    code = str(status.get("status") or "")
    return {
        "ai_status": code or None,
        "ai_status_label": status_label(code) if code else None,
        "ai_activity": status.get("activity") or status.get("event"),
    }


async def _conversation_response(conv, *, ensure_idle_music: bool = False) -> ConversationResponse:
    status = await _current_ai_status(getattr(conv, "agentId", None))
    music_co_listening = None
    try:
        from app.services import music as music_service
        from app.services import music_status

        if ensure_idle_music:
            status_code = str((status or {}).get("ai_status") or "idle")
            activity = (
                str((status or {}).get("ai_activity") or "").strip()
                or str((status or {}).get("ai_status_label") or "").strip()
                or "处理自己的事"
            )
            music_co_listening = await music_status.reconcile_co_listening_for_status(
                user_id=conv.userId,
                agent_id=conv.agentId,
                conversation_id=conv.id,
                workspace_id=conv.workspaceId,
                status_code=status_code,
                activity=activity,
                ai_name=await _agent_name(conv.agentId),
            )
        else:
            music_co_listening = await music_service.get_active_co_listening(
                conversation_id=conv.id,
            )
    except Exception:
        music_co_listening = None
    return ConversationResponse(
        id=conv.id,
        user_id=conv.userId,
        agent_id=conv.agentId,
        workspace_id=conv.workspaceId,
        title=conv.title,
        created_at=str(conv.createdAt),
        updated_at=str(conv.updatedAt),
        interaction_days=_interaction_days(conv.createdAt),
        music_co_listening=music_co_listening,
        **(status or {}),
    )


@router.post("", response_model=ConversationResponse)
async def create_conversation(
    data: ConversationCreate,
    user: dict = Depends(require_user),
):
    if data.user_id != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your user_id")
    agent = await db.aiagent.find_unique(where={"id": data.agent_id})
    # 放宽到 active / provisioning: agent 创建后端 LLM 后台 30-60s, 前端在
    # 此期间立刻创建会话, workspace 已 active, conversation 可正常落库;
    # 待 agent 激活后用户即可发送消息. archived 仍 404.
    if not agent or getattr(agent, "status", "active") == "archived":
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent.userId != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your agent")
    workspace = None
    if data.workspace_id:
        workspace = await get_workspace_by_id(data.workspace_id)
        if not workspace or workspace.status != "active":
            raise HTTPException(status_code=404, detail="Workspace not found")
        if workspace.userId != data.user_id or workspace.agentId != data.agent_id:
            raise HTTPException(status_code=400, detail="Workspace does not match user/agent")
    else:
        try:
            workspace = await ensure_workspace(data.user_id, data.agent_id)
        except ValueError as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc

    existing = await db.conversation.find_first(
        where={
            "workspaceId": workspace.id,
            "isDeleted": False,
        },
        order={"updatedAt": "desc"},
    )
    if existing:
        return await _conversation_response(existing, ensure_idle_music=True)

    try:
        conv = await db.conversation.create(
            data={
                "user": {"connect": {"id": data.user_id}},
                "agent": {"connect": {"id": data.agent_id}},
                "workspace": {"connect": {"id": workspace.id}},
                "title": data.title,
            }
        )
    except Exception:
        existing = await db.conversation.find_first(
            where={
                "workspaceId": workspace.id,
                "isDeleted": False,
            },
            order={"updatedAt": "desc"},
        )
        if not existing:
            raise
        conv = existing

    return await _conversation_response(conv, ensure_idle_music=True)


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conv=Depends(require_conversation_owner)):
    return await _conversation_response(conv, ensure_idle_music=True)


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    user_id: str = Query(...),
    workspace_id: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    _user=Depends(require_user_self),
):
    where: dict = {"isDeleted": False, "userId": user_id}
    if workspace_id:
        where["workspaceId"] = workspace_id
    convs = await db.conversation.find_many(
        where=where, order={"createdAt": "desc"}, take=limit, skip=offset
    )
    return [await _conversation_response(c) for c in convs]


@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
async def list_messages(
    conversation_id: str,
    limit: int = Query(default=100, le=100000),
    offset: int = 0,
    include_metadata: bool = Query(default=True),
    include_achievements: bool = Query(default=False),
    include_usage: bool = Query(default=False),
    conv=Depends(require_conversation_owner),
    user: dict = Depends(require_user),
):
    messages = await db.message.find_many(
        where={"conversationId": conversation_id},
        order={"createdAt": "desc"},
        take=limit,
        skip=offset,
    )
    items = [
        MessageResponse(
            id=m.id,
            conversation_id=m.conversationId,
            role=m.role,
            content=m.content,
            metadata=m.metadata if include_metadata else None,
            created_at=str(m.createdAt),
        )
        for m in messages
    ]
    # admin 显式请求时给带 trace_id 的回复消息附本轮 LLM 用量 (tokens/缓存/
    # 费用), 展示在 Trace 按钮旁. 仅 admin: 成本数据属运营信息, 普通用户
    # 的消息列表保持原 payload.
    if include_usage and include_metadata and user.get("role") == "admin":
        await _attach_llm_usage(items)
    # 聊天时间线成就行属于「聊天界面成就提示」: silent 模式下与 WS 弹窗/推送
    # 一并静默, 切回 on 后历史行自动出现 (unlocked_at/conversation_id 已落库).
    if include_achievements and await achievement_alerts_enabled():
        items.extend(
            await _achievement_timeline_items(
                conversation_id=conversation_id,
                user_id=conv.userId,
                agent_id=conv.agentId,
                messages=messages,
                is_latest_page=offset == 0,
                include_metadata=include_metadata,
            )
        )
        items.sort(
            key=lambda item: _parse_timeline_at(item.created_at),
            reverse=True,
        )
    return items


async def _attach_llm_usage(items: list[MessageResponse]) -> None:
    """给带 trace_id 的消息 metadata 注入 llm_usage (本轮 tokens/缓存/费用).

    同一轮多条气泡只有首条挂 trace_id (save_replies i==0), 天然每轮一次.
    聚合失败静默 — 用量是装饰性展示, 不影响消息读取.
    """
    from app.services.llm.usage_repo import aggregate_usage_by_trace_ids

    trace_ids = [
        str(item.metadata.get("trace_id"))
        for item in items
        if isinstance(item.metadata, dict) and item.metadata.get("trace_id")
    ]
    if not trace_ids:
        return
    usage_by_trace = await aggregate_usage_by_trace_ids(trace_ids)
    if not usage_by_trace:
        return
    for item in items:
        if not isinstance(item.metadata, dict):
            continue
        usage = usage_by_trace.get(str(item.metadata.get("trace_id") or ""))
        if usage:
            item.metadata["llm_usage"] = usage


async def _achievement_timeline_items(
    *,
    conversation_id: str,
    user_id: str,
    agent_id: str,
    messages: list,
    is_latest_page: bool,
    include_metadata: bool,
) -> list[MessageResponse]:
    if messages:
        message_times = [m.createdAt for m in messages if m.createdAt]
        if not message_times:
            return []
        lower = min(message_times)
        upper = datetime.now(UTC) if is_latest_page else max(message_times)
    elif not is_latest_page:
        return []
    else:
        lower = datetime.fromtimestamp(0, UTC)
        upper = datetime.now(UTC)

    rows = await db.query_raw(
        """
        SELECT id, achievement_id, unlocked_at
        FROM achievement_unlocks
        WHERE user_id = $1
          AND agent_id = $2
          AND conversation_id = $3
          AND unlocked_at >= $4::timestamp
          AND unlocked_at <= $5::timestamp
        ORDER BY unlocked_at DESC
        """,
        user_id,
        agent_id,
        conversation_id,
        lower,
        upper,
    )
    timeline: list[MessageResponse] = []
    for row in rows:
        achievement_id = int(_row_value(row, "achievement_id"))
        definition = ACHIEVEMENT_BY_ID.get(achievement_id)
        rule = ACHIEVEMENT_RULES.get(achievement_id)
        if not definition or not rule or not rule.enabled:
            continue
        unlock_id = _row_value(row, "id")
        unlocked_at = _row_value(row, "unlocked_at")
        payload = {
            **definition.to_dict(),
            "achievement_id": definition.id,
            "enabled": True,
            "unlocked": True,
            "unlocked_at": unlocked_at.isoformat()
            if hasattr(unlocked_at, "isoformat")
            else str(unlocked_at),
        }
        timeline.append(
            MessageResponse(
                id=f"achievement-{unlock_id}",
                conversation_id=conversation_id,
                role="achievement",
                content=definition.name,
                metadata={"achievement": payload} if include_metadata else None,
                created_at=payload["unlocked_at"],
            )
        )
    return timeline


def _row_value(row, key: str):
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def _parse_timeline_at(value: str | None) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, UTC)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.fromtimestamp(0, UTC)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@router.get("/{conversation_id}/crisis-care")
async def get_conversation_crisis_care(conv=Depends(require_conversation_owner)):
    """获取当前会话的危机后续关怀状态。需 JWT 且会话属于本人或 admin."""
    return await get_crisis_care_status(
        conv.id,
        conv.userId,
        workspace_id=conv.workspaceId,
        agent_id=conv.agentId,
    )


@router.delete("/{conversation_id}")
async def delete_conversation(conv=Depends(require_conversation_owner)):
    await db.conversation.update(
        where={"id": conv.id},
        data={"isDeleted": True, "archivedAt": datetime.now(UTC)},
    )
    return {"status": "deleted"}
