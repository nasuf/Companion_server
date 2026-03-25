from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.db import db


def utc_now() -> datetime:
    return datetime.now(UTC)


async def get_active_workspace(
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
) -> Any | None:
    where: dict[str, Any] = {"status": "active"}
    if user_id:
        where["userId"] = user_id
    if agent_id:
        where["agentId"] = agent_id
    return await db.chatworkspace.find_first(
        where=where,
        order={"createdAt": "desc"},
    )


async def get_workspace_by_id(workspace_id: str) -> Any | None:
    return await db.chatworkspace.find_unique(where={"id": workspace_id})


async def resolve_workspace_id(
    *,
    workspace_id: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
) -> str | None:
    if workspace_id:
        return workspace_id
    workspace = await get_active_workspace(user_id=user_id, agent_id=agent_id)
    return workspace.id if workspace else None


async def create_workspace(user_id: str, agent_id: str) -> Any:
    return await db.chatworkspace.create(
        data={
            "id": uuid4().hex,
            "user": {"connect": {"id": user_id}},
            "agent": {"connect": {"id": agent_id}},
            "status": "active",
        }
    )


async def create_provisioning_workspace(user_id: str, agent_id: str) -> Any:
    return await db.chatworkspace.create(
        data={
            "id": uuid4().hex,
            "user": {"connect": {"id": user_id}},
            "agent": {"connect": {"id": agent_id}},
            "status": "provisioning",
        }
    )


async def activate_workspace(workspace_id: str) -> Any:
    return await db.chatworkspace.update(
        where={"id": workspace_id},
        data={"status": "active", "archivedAt": None},
    )


async def reactivate_workspace(workspace_id: str) -> Any:
    return await db.chatworkspace.update(
        where={"id": workspace_id},
        data={"status": "active", "archivedAt": None},
    )


async def ensure_workspace(user_id: str, agent_id: str) -> Any:
    existing = await get_active_workspace(user_id=user_id, agent_id=agent_id)
    if existing:
        return existing
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or agent.userId != user_id or getattr(agent, "status", "active") != "active":
        raise ValueError("Active agent not found for workspace creation")
    return await create_workspace(user_id, agent_id)


async def archive_workspace(
    workspace_id: str,
    *,
    clear_runtime: bool = True,
) -> dict[str, Any] | None:
    workspace = await db.chatworkspace.find_unique(
        where={"id": workspace_id},
        include={"agent": True, "conversations": True},
    )
    if not workspace:
        return None

    archived_at = utc_now()
    conversation_ids = [c.id for c in (workspace.conversations or [])]

    if workspace.status != "archived":
        await db.chatworkspace.update(
            where={"id": workspace_id},
            data={"status": "archived", "archivedAt": archived_at},
        )

    if workspace.agentId and workspace.agent and getattr(workspace.agent, "status", "active") != "archived":
        await db.aiagent.update(
            where={"id": workspace.agentId},
            data={"status": "archived", "archivedAt": archived_at},
        )

    await db.conversation.update_many(
        where={"workspaceId": workspace_id, "isDeleted": False},
        data={"isDeleted": True, "archivedAt": archived_at},
    )

    runtime_stats: dict[str, int] = {}
    if clear_runtime and workspace.agentId:
        from app.services.data_reset import clear_agent_runtime_state

        runtime_stats = await clear_agent_runtime_state(
            workspace.id,
            workspace.agentId,
            workspace.userId,
            conversation_ids,
        )

    return {
        "workspace_id": workspace.id,
        "agent_id": workspace.agentId,
        "user_id": workspace.userId,
        "conversation_ids": conversation_ids,
        "runtime": runtime_stats,
    }


async def archive_active_workspaces_for_user(user_id: str) -> list[dict[str, Any]]:
    workspaces = await db.chatworkspace.find_many(
        where={"userId": user_id, "status": "active"},
        order={"createdAt": "desc"},
    )
    results: list[dict[str, Any]] = []
    for workspace in workspaces:
        archived = await archive_workspace(workspace.id)
        if archived:
            results.append(archived)
    return results


async def stage_active_workspaces_for_user(user_id: str) -> list[dict[str, Any]]:
    workspaces = await db.chatworkspace.find_many(
        where={"userId": user_id, "status": "active"},
        include={"agent": True, "conversations": True},
        order={"createdAt": "desc"},
    )
    staged: list[dict[str, Any]] = []
    for workspace in workspaces:
        archived = await archive_workspace(workspace.id, clear_runtime=False)
        if archived:
            staged.append(archived)
    return staged


async def finalize_archived_workspaces(staged_workspaces: list[dict[str, Any]]) -> None:
    for workspace in staged_workspaces:
        agent_id = workspace.get("agent_id")
        if not agent_id:
            continue
        from app.services.data_reset import clear_agent_runtime_state

        await clear_agent_runtime_state(
            workspace["workspace_id"],
            agent_id,
            workspace["user_id"],
            workspace.get("conversation_ids") or [],
        )


async def restore_staged_workspaces(staged_workspaces: list[dict[str, Any]]) -> None:
    for workspace in staged_workspaces:
        await reactivate_workspace(workspace["workspace_id"])
        agent_id = workspace.get("agent_id")
        if agent_id:
            await db.aiagent.update(
                where={"id": agent_id},
                data={"status": "active", "archivedAt": None},
            )


async def archive_provisioning_workspace(workspace_id: str) -> Any:
    return await db.chatworkspace.update(
        where={"id": workspace_id},
        data={"status": "archived", "archivedAt": utc_now()},
    )
