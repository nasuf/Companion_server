"""Admin API for character template & profile management."""

from __future__ import annotations

import asyncio
import logging
import random as _random
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from prisma import Json

from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.redis_client import get_redis
from app.models.character import (
    ProfileBatchStatusRequest,
    ProfileGenerateRequest,
    ProfileResponse,
    ProfileUpdateRequest,
    PromptDefaultsResponse,
    PromptPreviewRequest,
    PromptPreviewResponse,
    TemplateCopyRequest,
    TemplateCreateRequest,
    TemplateResponse,
    TemplateUpdateRequest,
)
from app.services.character import (
    build_generation_prompt,
    clamp_agent_age,
    generate_single_profile,
    get_default_prompts,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin-api/character", tags=["admin-character"])


# ── helpers ──

def _career_row_to_dict(c) -> dict:
    """Convert a CareerTemplate DB row to a plain dict for prompt injection."""
    return {
        "id": c.id,
        "title": c.title,
        "duties": c.duties,
        "socialValue": c.socialValue,
        "clients": c.clients,
    }


async def _count_profiles_grouped(field: str, ids: list[str]) -> dict[str, int]:
    """Batch-count profiles grouped by a foreign key field. Eliminates N+1."""
    if not ids:
        return {}
    rows = await db.query_raw(
        f'SELECT "{field}" AS fk, COUNT(*)::int AS cnt '
        f'FROM "character_profiles" WHERE "{field}" = ANY($1) GROUP BY "{field}"',
        ids,
    )
    return {str(r["fk"]): int(r["cnt"]) for r in rows}

def _template_response(tpl, profile_count: int = 0) -> TemplateResponse:
    return TemplateResponse(
        id=tpl.id,
        name=tpl.name,
        description=tpl.description,
        schema_body=tpl.schemaData,
        prompt_header=getattr(tpl, "promptHeader", None),
        defaults=tpl.defaults,
        prompt_requirements=getattr(tpl, "promptRequirements", None),
        status=tpl.status,
        profile_count=profile_count,
        created_at=str(tpl.createdAt),
        updated_at=str(tpl.updatedAt),
    )


def _profile_response(p, template_name: str | None = None) -> ProfileResponse:
    career_title = None
    if hasattr(p, "career") and p.career:
        career_title = p.career.title
    return ProfileResponse(
        id=p.id,
        template_id=p.templateId,
        template_name=template_name or (p.template.name if hasattr(p, "template") and p.template else None),
        career_id=getattr(p, "careerId", None),
        career_title=career_title,
        name=p.name,
        data=p.data if isinstance(p.data, dict) else {},
        status=p.status,
        agent_id=p.agentId,
        created_at=str(p.createdAt),
        updated_at=str(p.updatedAt),
    )


# ── Template CRUD ──

@router.get("/templates", response_model=list[TemplateResponse])
async def list_templates(
    status: str | None = None,
    _: str = Depends(require_admin_jwt),
):
    where: dict = {}
    if status:
        where["status"] = status
    templates = await db.charactertemplate.find_many(
        where=where,
        order={"createdAt": "desc"},
    )
    counts = await _count_profiles_grouped("template_id", [t.id for t in templates])
    return [_template_response(tpl, profile_count=counts.get(tpl.id, 0)) for tpl in templates]


@router.post("/templates", response_model=TemplateResponse, status_code=201)
async def create_template(
    body: TemplateCreateRequest,
    _: str = Depends(require_admin_jwt),
):
    tpl = await db.charactertemplate.create(
        data={
            "name": body.name,
            "description": body.description,
            "schemaData": Json(body.schema_body.model_dump()),
            "promptHeader": body.prompt_header,
            "defaults": body.defaults,
            "promptRequirements": body.prompt_requirements,
        }
    )
    return _template_response(tpl)


@router.get("/templates/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: str,
    _: str = Depends(require_admin_jwt),
):
    tpl = await db.charactertemplate.find_unique(where={"id": template_id})
    if not tpl:
        raise HTTPException(status_code=404, detail="Template not found")
    count = await db.characterprofile.count(where={"templateId": tpl.id})
    return _template_response(tpl, profile_count=count)


@router.put("/templates/{template_id}", response_model=TemplateResponse)
async def update_template(
    template_id: str,
    body: TemplateUpdateRequest,
    _: str = Depends(require_admin_jwt),
):
    tpl = await db.charactertemplate.find_unique(where={"id": template_id})
    if not tpl:
        raise HTTPException(status_code=404, detail="Template not found")
    update_data: dict = {}
    if body.name is not None:
        update_data["name"] = body.name
    if body.description is not None:
        update_data["description"] = body.description
    if body.schema_body is not None:
        update_data["schemaData"] = Json(body.schema_body.model_dump())
    if body.prompt_header is not None:
        update_data["promptHeader"] = body.prompt_header
    if body.defaults is not None:
        update_data["defaults"] = body.defaults
    if body.prompt_requirements is not None:
        update_data["promptRequirements"] = body.prompt_requirements
    if not update_data:
        count = await db.characterprofile.count(where={"templateId": tpl.id})
        return _template_response(tpl, profile_count=count)
    updated = await db.charactertemplate.update(where={"id": template_id}, data=update_data)
    count = await db.characterprofile.count(where={"templateId": updated.id})
    return _template_response(updated, profile_count=count)


@router.delete("/templates/{template_id}")
async def delete_template(
    template_id: str,
    force: bool = False,
    _: str = Depends(require_admin_jwt),
):
    tpl = await db.charactertemplate.find_unique(where={"id": template_id})
    if not tpl:
        raise HTTPException(status_code=404, detail="Template not found")
    profile_count = await db.characterprofile.count(where={"templateId": template_id})
    if profile_count > 0 and not force:
        return {
            "ok": False,
            "action": "blocked",
            "profile_count": profile_count,
            "message": f"该模板关联了 {profile_count} 个背景，确认级联删除？",
        }
    if force and profile_count > 0:
        await db.characterprofile.delete_many(where={"templateId": template_id})
    await db.charactertemplate.delete(where={"id": template_id})
    return {"ok": True, "action": "deleted"}


@router.post("/templates/{template_id}/copy", response_model=TemplateResponse, status_code=201)
async def copy_template(
    template_id: str,
    body: TemplateCopyRequest | None = None,
    _: str = Depends(require_admin_jwt),
):
    tpl = await db.charactertemplate.find_unique(where={"id": template_id})
    if not tpl:
        raise HTTPException(status_code=404, detail="Template not found")
    new_name = (body.name if body and body.name else None) or f"{tpl.name} (副本)"
    copied = await db.charactertemplate.create(
        data={
            "name": new_name,
            "description": tpl.description,
            "schemaData": Json(tpl.schemaData if isinstance(tpl.schemaData, dict) else {}),
            "promptHeader": getattr(tpl, "promptHeader", None),
            "defaults": tpl.defaults,
            "promptRequirements": getattr(tpl, "promptRequirements", None),
        }
    )
    return _template_response(copied)


# ── Prompt Preview / Defaults ──

@router.get("/prompt-defaults", response_model=PromptDefaultsResponse)
async def prompt_defaults(_: str = Depends(require_admin_jwt)):
    """Return system default prompt header and requirements.

    Used by frontend "重置默认" button to populate textareas with the original values.
    Reads from prompting registry (character.template_header /
    character.template_requirements) so admin edits via「提示词管理」UI take effect.
    """
    defaults = await get_default_prompts()
    return PromptDefaultsResponse(
        header=defaults["header"],
        requirements=defaults["requirements"],
    )


@router.post("/templates/preview-prompt", response_model=PromptPreviewResponse)
async def preview_prompt(
    body: PromptPreviewRequest,
    _: str = Depends(require_admin_jwt),
):
    """Build the assembled generation prompt without saving the template.

    Allows admin UI to show a live preview of what will be sent to the LLM.
    body 字段为空 → 用 registry default (与「重置默认」一致), 这样模板尚未保存
    时也能看到最终拼装结果。
    """
    career_data: dict | None = None
    if body.career_id:
        career_row = await db.careertemplate.find_unique(where={"id": body.career_id})
        if career_row:
            career_data = _career_row_to_dict(career_row)
    # truthy 判断: 兼容 admin 把 textarea 清空 → 字段送空字符串. 空覆盖一律
    # 回退 registry default, 与 generate_single_profile 行为一致。
    header = body.prompt_header
    requirements = body.prompt_requirements
    if not header or not requirements:
        defaults = await get_default_prompts()
        header = header or defaults["header"]
        requirements = requirements or defaults["requirements"]
    prompt = build_generation_prompt(
        body.schema_body.model_dump(),
        body.defaults,
        index=body.index,
        header=header,
        requirements=requirements,
        career=career_data,
    )
    return PromptPreviewResponse(prompt=prompt)


# ── Profile CRUD ──

@router.get("/profiles", response_model=list[ProfileResponse])
async def list_profiles(
    template_id: str | None = None,
    career_id: str | None = None,
    status: str | None = None,
    _: str = Depends(require_admin_jwt),
):
    where: dict = {}
    if template_id:
        where["templateId"] = template_id
    if career_id:
        where["careerId"] = career_id
    if status:
        where["status"] = status
    profiles = await db.characterprofile.find_many(
        where=where,
        order={"createdAt": "desc"},
        include={"template": True, "career": True},
    )
    return [_profile_response(p) for p in profiles]


_BATCH_LOCK_PREFIX = "lock:profile_gen:"


def _batch_lock_key(template_id: str) -> str:
    return f"{_BATCH_LOCK_PREFIX}{template_id}"


@router.get("/templates/{template_id}/batch-status")
async def batch_status(template_id: str, _: str = Depends(require_admin_jwt)):
    """前端 mount / refresh 后查询该模板是否有批次正在跑。

    返回 `{ "in_progress": bool }`. 前端用此值在 Redis 锁还在时把"开始生成"
    按钮置灰, 避免刷新后再次点击触发重复生成 (历史 bug: 用户连点 5 次,
    1 次 30 个 → 总产 150+ profile).
    """
    try:
        redis = await get_redis()
        held = await redis.exists(_batch_lock_key(template_id))
        return {"in_progress": bool(held)}
    except Exception as e:
        logger.warning(f"batch_status redis check failed for {template_id}: {e}")
        # Redis 挂时优雅降级 (后端会同步走 LLM, 不靠锁): 不阻止前端尝试。
        return {"in_progress": False}


@router.post("/profiles/generate", response_model=list[ProfileResponse])
async def generate_profiles(
    body: ProfileGenerateRequest,
    _: str = Depends(require_admin_jwt),
):
    from app.config import settings

    tpl = await db.charactertemplate.find_unique(where={"id": body.template_id})
    if not tpl:
        raise HTTPException(status_code=404, detail="Template not found")
    schema = tpl.schemaData if isinstance(tpl.schemaData, dict) else {}
    # 单请求上限默认 100; 实际 LLM 并发由下方 Semaphore 控制 (默认 10),
    # 防止 100 个同时打 DashScope 触发账号级 429. 100 × 30-60s / 10 ≈ 5-10 min.
    count = max(1, min(body.count, settings.character_profile_batch_max))
    header_override = getattr(tpl, "promptHeader", None)
    requirements_override = getattr(tpl, "promptRequirements", None)
    sem = asyncio.Semaphore(settings.character_profile_batch_concurrency)

    # 排他锁: 同一 template 同时只能跑一个批次. 防止用户在等待期间多次点击 /
    # 刷新后重复提交导致 N × count 个 profile 被生成 (生成速度 ≈ ceil(count /
    # concurrency) × 30-60s, 100/10 大约 5-10 min, 锁 TTL 给 15 min 留余量).
    lock_key = _batch_lock_key(body.template_id)
    lock_ttl = 15 * 60  # 15 min, 远超最长批次预期
    redis = None
    try:
        redis = await get_redis()
        acquired = await redis.set(lock_key, "1", nx=True, ex=lock_ttl)
    except Exception as e:
        logger.warning(f"batch lock acquire failed for {body.template_id}: {e}")
        acquired = True  # Redis 挂 → 不阻塞业务, fail-open
    if not acquired:
        raise HTTPException(
            status_code=409,
            detail="该模板已有批次正在生成, 请等待当前批次完成后再试",
        )

    # ── 职业选择 ──
    fixed_career: dict | None = None
    career_pool: list[dict] = []
    if body.career_id:
        # 指定职业
        career_row = await db.careertemplate.find_unique(where={"id": body.career_id})
        if not career_row:
            raise HTTPException(status_code=404, detail="Career not found")
        fixed_career = _career_row_to_dict(career_row)
    else:
        # 随机：加载所有活跃职业并打乱
        all_careers = await db.careertemplate.find_many(
            where={"status": "active"},
            order={"sortOrder": "asc"},
        )
        career_pool = [_career_row_to_dict(c) for c in all_careers]
        _random.shuffle(career_pool)

    # 决定每条的 (career, gender), 进 asyncio.gather 并行化
    plans: list[tuple[int, dict | None, str]] = []
    for i in range(count):
        if fixed_career:
            career = fixed_career
        elif career_pool:
            career = career_pool[i % len(career_pool)]
        else:
            career = None
        if body.gender in ("male", "female"):
            gender_for_run = body.gender
        elif body.gender is None:
            gender_for_run = _random.choice(["male", "female"])
        else:
            raise HTTPException(status_code=400, detail=f"gender 必须是 male/female/null, 收到 {body.gender!r}")
        plans.append((i, career, gender_for_run))

    async def _gen_one(i: int, career: dict | None, gender_for_run: str) -> ProfileResponse:
        # Semaphore 限并发: 100 个 task 全部进 gather, 实际只有 N 个同时 await
        # LLM, 其余排队. 防 DashScope 429.
        async with sem:
            data = await generate_single_profile(
                schema,
                tpl.defaults,
                index=i,
                header=header_override,
                requirements=requirements_override,
                career=career,
                gender=gender_for_run,
            )
        if not data:
            raise ValueError("LLM 返回为空")

        # 性别字段强制覆盖: profile.identity.gender 存中文 (与既有种子数据一致,
        # 读取侧 _profile_gender_en 会翻译回英文)。
        if not isinstance(data.get("identity"), dict):
            data["identity"] = {}
        data["identity"]["gender"] = "男" if gender_for_run == "male" else "女"

        # 职业 / 民族 / 姓名 等覆盖由 generate_single_profile 内部的
        # _apply_postprocess_overrides 统一处理 (含 income 默认值与 clients 拆 tags).

        # 后处理：用生日精确计算年龄（LLM 不擅长算术），再按 spec §1.3
        # 硬钳 20-29（prompt hint 不保证，LLM 偶尔生成超出区间的 birthday）
        identity = data.get("identity")
        if isinstance(identity, dict):
            birthday = identity.get("birthday")
            if birthday and isinstance(birthday, str):
                try:
                    bd = date.fromisoformat(birthday)
                    today = date.today()
                    age = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
                    identity["age"] = clamp_agent_age(age)
                except (ValueError, TypeError):
                    pass

        # 姓名由用户在 app 内手动输入，背景不含姓名 (_apply_postprocess_overrides 已清);
        # profile_name 用职业名做管理标识
        career_title = ""
        if career:
            career_title = career.get("title", "")
        elif isinstance(data.get("career"), dict):
            career_title = data["career"].get("title", "")
        profile_name = career_title or f"角色_{i + 1}"

        create_data: dict = {
            "templateId": tpl.id,
            "name": profile_name,
            "data": Json(data),
            "status": "draft",
        }
        if career and career.get("id"):
            create_data["careerId"] = career["id"]
        p = await db.characterprofile.create(data=create_data)
        logger.info(f"Generated profile {i + 1}/{count}: {profile_name}")
        return _profile_response(p, template_name=tpl.name)

    # 全 count 进 gather, 内部 Semaphore 控制实际并发 (默认 10, 可配),
    # 防 DashScope 429。无论成功 / 失败 / 异常都要释放锁防止后续请求被
    # 误认为"批次仍在跑"。
    try:
        raw = await asyncio.gather(
            *(_gen_one(i, career, gender) for i, career, gender in plans),
            return_exceptions=True,
        )
        results: list[ProfileResponse] = []
        failures: list[str] = []
        for i, item in enumerate(raw):
            if isinstance(item, Exception):
                err_msg = f"#{i + 1}: {type(item).__name__}: {str(item)[:200]}"
                failures.append(err_msg)
                logger.warning(f"Profile generation failed at index {i}: {item}", exc_info=item)
            else:
                results.append(item)

        # 全部失败 → 抛 500，让前端 toast 显示具体原因
        if not results and failures:
            raise HTTPException(
                status_code=500,
                detail=f"全部 {count} 个画像生成失败。首个错误: {failures[0]}",
            )

        return results
    finally:
        if redis is not None:
            try:
                await redis.delete(lock_key)
            except Exception as e:
                logger.warning(f"batch lock release failed for {body.template_id}: {e}")


@router.get("/profiles/{profile_id}", response_model=ProfileResponse)
async def get_profile(
    profile_id: str,
    _: str = Depends(require_admin_jwt),
):
    p = await db.characterprofile.find_unique(
        where={"id": profile_id},
        include={"template": True, "career": True},
    )
    if not p:
        raise HTTPException(status_code=404, detail="Profile not found")
    return _profile_response(p)


@router.put("/profiles/{profile_id}", response_model=ProfileResponse)
async def update_profile(
    profile_id: str,
    body: ProfileUpdateRequest,
    _: str = Depends(require_admin_jwt),
):
    p = await db.characterprofile.find_unique(where={"id": profile_id})
    if not p:
        raise HTTPException(status_code=404, detail="Profile not found")
    update_data: dict = {}
    if body.name is not None:
        update_data["name"] = body.name
    if body.data is not None:
        update_data["data"] = Json(body.data)
    if body.status is not None:
        update_data["status"] = body.status
    if not update_data:
        return _profile_response(p)
    updated = await db.characterprofile.update(
        where={"id": profile_id},
        data=update_data,
        include={"template": True, "career": True},
    )
    return _profile_response(updated)


@router.post("/profiles/batch-status")
async def batch_update_profile_status(
    body: ProfileBatchStatusRequest,
    _: str = Depends(require_admin_jwt),
):
    """Batch update status for multiple profiles. 仅支持 draft / published 二态。"""
    if body.status not in ("draft", "published"):
        raise HTTPException(status_code=400, detail="Invalid status")
    cnt = await db.execute_raw(
        'UPDATE "character_profiles" SET "status" = $1, "updated_at" = CURRENT_TIMESTAMP WHERE "id" = ANY($2::text[])',
        body.status,
        body.ids,
    )
    return {"ok": True, "updated": cnt or 0}


@router.delete("/profiles/{profile_id}")
async def delete_profile(
    profile_id: str,
    _: str = Depends(require_admin_jwt),
):
    """物理删除 profile. 历史曾支持 ?force=false 走"软归档", 现在 status 简化
    为 draft / published 二态, 不再有 archived → 永远走永久删除。"""
    p = await db.characterprofile.find_unique(where={"id": profile_id})
    if not p:
        raise HTTPException(status_code=404, detail="Profile not found")
    await db.characterprofile.delete(where={"id": profile_id})
    return {"ok": True, "action": "deleted"}
