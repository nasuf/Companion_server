"""Admin API for career template management."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.models.career import CareerCreateRequest, CareerResponse, CareerUpdateRequest

router = APIRouter(prefix="/admin-api/careers", tags=["admin-careers"])


def _career_response(c, profile_count: int = 0) -> CareerResponse:
    return CareerResponse(
        id=c.id,
        title=c.title,
        duties=c.duties,
        outputs=c.outputs,
        social_value=c.socialValue,
        clients=c.clients,
        status=c.status,
        sort_order=c.sortOrder,
        profile_count=profile_count,
        created_at=str(c.createdAt),
        updated_at=str(c.updatedAt),
    )


@router.get("", response_model=list[CareerResponse])
async def list_careers(
    status: str | None = None,
    _: str = Depends(require_admin_jwt),
):
    where: dict = {}
    if status:
        where["status"] = status
    careers = await db.careertemplate.find_many(
        where=where,
        order={"sortOrder": "asc"},
    )
    # Batch count profiles per career (single query instead of N+1)
    if careers:
        ids = [c.id for c in careers]
        rows = await db.query_raw(
            'SELECT "career_id" AS fk, COUNT(*)::int AS cnt '
            'FROM "character_profiles" WHERE "career_id" = ANY($1) GROUP BY "career_id"',
            ids,
        )
        counts = {str(r["fk"]): int(r["cnt"]) for r in rows}
    else:
        counts = {}
    return [_career_response(c, profile_count=counts.get(c.id, 0)) for c in careers]


@router.post("", response_model=CareerResponse, status_code=201)
async def create_career(
    body: CareerCreateRequest,
    _: str = Depends(require_admin_jwt),
):
    c = await db.careertemplate.create(
        data={
            "title": body.title,
            "duties": body.duties,
            "outputs": body.outputs,
            "socialValue": body.social_value,
            "clients": body.clients,
            "sortOrder": body.sort_order,
        }
    )
    return _career_response(c)


@router.put("/{career_id}", response_model=CareerResponse)
async def update_career(
    career_id: str,
    body: CareerUpdateRequest,
    _: str = Depends(require_admin_jwt),
):
    c = await db.careertemplate.find_unique(where={"id": career_id})
    if not c:
        raise HTTPException(status_code=404, detail="Career not found")
    update_data: dict = {}
    if body.title is not None:
        update_data["title"] = body.title
    if body.duties is not None:
        update_data["duties"] = body.duties
    if body.outputs is not None:
        update_data["outputs"] = body.outputs
    if body.social_value is not None:
        update_data["socialValue"] = body.social_value
    if body.clients is not None:
        update_data["clients"] = body.clients
    if body.sort_order is not None:
        update_data["sortOrder"] = body.sort_order
    if not update_data:
        return _career_response(c)
    updated = await db.careertemplate.update(where={"id": career_id}, data=update_data)
    return _career_response(updated)


@router.delete("/{career_id}")
async def delete_career(
    career_id: str,
    force: bool = False,
    _: str = Depends(require_admin_jwt),
):
    c = await db.careertemplate.find_unique(where={"id": career_id})
    if not c:
        raise HTTPException(status_code=404, detail="Career not found")
    profile_count = await db.characterprofile.count(where={"careerId": career_id})
    if profile_count > 0 and not force:
        return {
            "ok": False,
            "action": "blocked",
            "profile_count": profile_count,
            "message": f"该职业关联了 {profile_count} 个背景，确认级联删除？",
        }
    if force and profile_count > 0:
        await db.characterprofile.delete_many(where={"careerId": career_id})
    await db.careertemplate.delete(where={"id": career_id})
    return {"ok": True, "action": "deleted"}
