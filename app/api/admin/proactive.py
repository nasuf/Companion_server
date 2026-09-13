"""Admin API: manually trigger proactive chat for QA.

POST /admin-api/proactive/trigger — fire one proactive message for the
admin's own workspace (or an explicit workspace_id), bypassing daily/fatigue
limits by default so repeated local testing is possible.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.services.proactive.admin_test import AdminProactiveSendOutcome, AdminProactiveTestOptions
from app.services.proactive.sender import send_manual_or_triggered_proactive
from app.services.proactive.special_dates import Occasion, send_special_date_proactive
from app.services.workspace.workspaces import resolve_workspace_id

router = APIRouter(
    prefix="/admin-api/proactive",
    tags=["admin", "proactive"],
    dependencies=[Depends(require_admin_jwt)],
)

ADMIN_TRIGGER_TYPES = frozenset({
    "silence_wakeup",
    "scheduled_scene",
    "special_date",
    "manual_trigger",
})

ADMIN_SPECIAL_DATE_PRESETS: dict[str, list[Occasion]] = {
    "holiday": [Occasion(type="holiday", name="测试节日", owner="user")],
    "birthday_user": [Occasion(type="birthday", name="生日", owner="user")],
    "birthday_ai": [Occasion(type="birthday", name="生日", owner="ai")],
    "combined": [
        Occasion(type="holiday", name="元旦", owner="user"),
        Occasion(type="birthday", name="生日", owner="user"),
    ],
}


class AdminProactiveTriggerRequest(BaseModel):
    workspace_id: str | None = None
    agent_id: str | None = None
    trigger_type: str = Field(default="silence_wakeup")
    skip_limits: bool = True
    special_date_preset: str = Field(default="holiday")
    use_web_search: bool = False
    use_link_card: bool = False


class AdminProactiveTriggerResponse(BaseModel):
    ok: bool
    trigger_type: str
    message: str | None = None
    reason: str | None = None
    web_search_used: bool = False
    link_card_used: bool = False


async def _resolve_workspace_and_agent(
    *,
    user_id: str,
    workspace_id: str | None,
    agent_id: str | None,
) -> tuple[str, str]:
    ws_id = (workspace_id or "").strip() or None
    ag_id = (agent_id or "").strip() or None

    if ws_id and not ag_id:
        row = await db.chatworkspace.find_unique(where={"id": ws_id})
        if not row:
            raise HTTPException(status_code=404, detail="workspace_not_found")
        ag_id = str(row.agentId)

    if not ws_id:
        if not ag_id:
            raise HTTPException(
                status_code=400,
                detail="workspace_id or agent_id is required",
            )
        ws_id = await resolve_workspace_id(user_id=user_id, agent_id=ag_id)
        if not ws_id:
            raise HTTPException(status_code=404, detail="workspace_not_found")

    if not ag_id:
        row = await db.chatworkspace.find_unique(where={"id": ws_id})
        if not row:
            raise HTTPException(status_code=404, detail="workspace_not_found")
        ag_id = str(row.agentId)

    return ws_id, ag_id


@router.post("/trigger", response_model=AdminProactiveTriggerResponse)
async def admin_trigger_proactive(
    payload: AdminProactiveTriggerRequest,
    admin: dict = Depends(require_admin_jwt),
) -> AdminProactiveTriggerResponse:
    trigger_type = (payload.trigger_type or "silence_wakeup").strip()
    if trigger_type not in ADMIN_TRIGGER_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"trigger_type must be one of: {', '.join(sorted(ADMIN_TRIGGER_TYPES))}",
        )

    user_id = str(admin.get("sub") or "")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    workspace_id, agent_id = await _resolve_workspace_and_agent(
        user_id=user_id,
        workspace_id=payload.workspace_id,
        agent_id=payload.agent_id,
    )

    admin_opts = AdminProactiveTestOptions(
        use_web_search=payload.use_web_search,
        use_link_card=payload.use_link_card,
    )

    if trigger_type == "special_date":
        preset = (payload.special_date_preset or "holiday").strip()
        occasions = ADMIN_SPECIAL_DATE_PRESETS.get(preset)
        if not occasions:
            raise HTTPException(
                status_code=400,
                detail=(
                    "special_date_preset must be one of: "
                    f"{', '.join(sorted(ADMIN_SPECIAL_DATE_PRESETS))}"
                ),
            )
        outcome = AdminProactiveSendOutcome()
        sent = await send_special_date_proactive(
            agent_id=agent_id,
            user_id=user_id,
            workspace_id=workspace_id,
            occasions=occasions,
            skip_limits=payload.skip_limits,
            admin_test_options=admin_opts,
            send_outcome=outcome,
        )
        if not sent:
            return AdminProactiveTriggerResponse(
                ok=False,
                trigger_type=trigger_type,
                message=None,
                reason="special_date_generation_blocked",
                web_search_used=outcome.web_search_used,
                link_card_used=outcome.link_card_used,
            )
        rows = await db.query_raw(
            """
            SELECT message
            FROM proactive_chat_logs
            WHERE workspace_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            workspace_id,
        )
        latest_message = str(rows[0]["message"]) if rows else None
        return AdminProactiveTriggerResponse(
            ok=True,
            trigger_type=trigger_type,
            message=latest_message,
            reason=None,
            web_search_used=outcome.web_search_used,
            link_card_used=outcome.link_card_used,
        )

    result = await send_manual_or_triggered_proactive(
        workspace_id=workspace_id,
        trigger_type=trigger_type,
        skip_limits=payload.skip_limits,
        admin_test_options=admin_opts,
    )
    return AdminProactiveTriggerResponse(
        ok=bool(result.get("ok")),
        trigger_type=trigger_type,
        message=result.get("message"),  # type: ignore[arg-type]
        reason=result.get("reason"),  # type: ignore[arg-type]
        web_search_used=bool(result.get("web_search_used")),
        link_card_used=bool(result.get("link_card_used")),
    )
