"""Admin API for the agent name library."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from prisma.errors import UniqueViolationError

from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.models.name_template import (
    NameCreateRequest,
    NameListResponse,
    NameResponse,
    NameUpdateRequest,
)
from app.services.name_templates import name_row_to_dict, normalize_name_gender

router = APIRouter(
    prefix="/admin-api/names",
    tags=["admin", "name-templates"],
    dependencies=[Depends(require_admin_jwt)],
)

_MAX_LIST_LIMIT = 200


def _name_response(row) -> NameResponse:
    return NameResponse(**name_row_to_dict(row))


def _require_gender(value: str | None) -> str:
    gender = normalize_name_gender(value)
    if gender not in {"male", "female"}:
        raise HTTPException(status_code=400, detail="gender 只能是 male/female/男/女")
    return gender


@router.get("", response_model=NameListResponse)
async def list_names(
    gender: str | None = None,
    q: str | None = None,
    status: str | None = "active",
    limit: int = Query(80, ge=1, le=_MAX_LIST_LIMIT),
    offset: int = Query(0, ge=0),
):
    where: dict = {}
    if status:
        where["status"] = status
    if gender:
        where["gender"] = _require_gender(gender)
    query = (q or "").strip()
    if query:
        where["OR"] = [
            {"name": {"contains": query}},
            {"nickname": {"contains": query}},
        ]

    total = await db.nametemplate.count(where=where)
    rows = await db.nametemplate.find_many(
        where=where,
        order=[{"sortOrder": "asc"}, {"name": "asc"}],
        skip=offset,
        take=limit,
    )
    male_where = {"gender": "male"}
    female_where = {"gender": "female"}
    if status:
        male_where["status"] = status
        female_where["status"] = status
    male_count = await db.nametemplate.count(where=male_where)
    female_count = await db.nametemplate.count(where=female_where)
    return NameListResponse(
        items=[_name_response(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
        counts={"male": male_count, "female": female_count},
    )


@router.post("", response_model=NameResponse, status_code=201)
async def create_name(body: NameCreateRequest):
    gender = _require_gender(body.gender)
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="请填写姓名")
    nickname = (body.nickname or "").strip()
    sort_order = body.sort_order
    if sort_order <= 0:
        # Prepend so a newly added row shows up on page 1 instead of after
        # the 700 seeded names.
        min_row = await db.nametemplate.find_first(
            where={"gender": gender},
            order={"sortOrder": "asc"},
        )
        sort_order = (min_row.sortOrder if min_row else 1) - 1
    try:
        row = await db.nametemplate.create(
            data={
                "name": name,
                "nickname": nickname,
                "gender": gender,
                "sortOrder": sort_order,
            }
        )
    except UniqueViolationError as exc:
        raise HTTPException(status_code=409, detail="该性别下已有同名姓名") from exc
    return _name_response(row)


@router.put("/{name_id}", response_model=NameResponse)
async def update_name(name_id: str, body: NameUpdateRequest):
    row = await db.nametemplate.find_unique(where={"id": name_id})
    if not row:
        raise HTTPException(status_code=404, detail="姓名不存在")
    update_data: dict = {}
    if body.name is not None:
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="请填写姓名")
        update_data["name"] = name
    if body.nickname is not None:
        update_data["nickname"] = body.nickname.strip()
    if body.gender is not None:
        update_data["gender"] = _require_gender(body.gender)
    if body.status is not None:
        status = body.status.strip()
        if status not in {"active", "archived"}:
            raise HTTPException(status_code=400, detail="status 只能是 active/archived")
        update_data["status"] = status
    if body.sort_order is not None:
        update_data["sortOrder"] = body.sort_order
    if not update_data:
        return _name_response(row)
    try:
        updated = await db.nametemplate.update(where={"id": name_id}, data=update_data)
    except UniqueViolationError as exc:
        raise HTTPException(status_code=409, detail="该性别下已有同名姓名") from exc
    return _name_response(updated)


@router.delete("/{name_id}")
async def delete_name(name_id: str):
    row = await db.nametemplate.find_unique(where={"id": name_id})
    if not row:
        raise HTTPException(status_code=404, detail="姓名不存在")
    await db.nametemplate.delete(where={"id": name_id})
    return {"ok": True, "action": "deleted"}
