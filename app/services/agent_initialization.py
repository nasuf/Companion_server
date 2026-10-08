"""Agent initialization pipeline, shared by local fallback and queue workers."""
import asyncio
import logging
from prisma import Json
from app.db import db
from app.services.interaction.boundary import init_patience
from app.services.mbti import build_mbti, seven_dim_to_mbti
from app.services.career import pick_random_active_career
from app.services.character import _apply_postprocess_overrides
from app.services.character_generation import generate_full_profile
from app.services.life_story import activate_agent, generate_l1_coverage, set_progress
from app.services.proactive.sender import dispatch_first_greeting_for_agent
from app.services.schedule_domain.schedule import generate_and_save_life_overview, generate_daily_schedule

logger = logging.getLogger(__name__)


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
    profile_override = payload.get("profile_override")
    career_template_override = payload.get("career_template_override")
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
        await _run_agent_initialization_inner(
            agent,
            user_id,
            workspace_id,
            personality_dict,
            profile_override=profile_override if isinstance(profile_override, dict) else None,
            career_template_override=(
                career_template_override if isinstance(career_template_override, dict) else None
            ),
        )


async def _run_agent_initialization_inner(
    agent,
    user_id: str,
    workspace_id: str | None,
    personality_dict: dict,
    *,
    profile_override: dict | None = None,
    career_template_override: dict | None = None,
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
        from app.services.speech_output.voices import ensure_agent_voice

        await ensure_agent_voice(agent)
    except Exception as e:
        logger.error(f"MBTI init failed for agent {agent.id}: {e}")
    await set_progress(agent.id, "mbti_done", message="MBTI 推导完成")

    await set_progress(agent.id, "prompt_building", message="正在构建生成提示...")
    profile: dict
    career = career_template_override
    if profile_override is not None:
        await set_progress(agent.id, "llm_generating", message="正在读取文档背景...")
        profile = dict(profile_override)
        profile.setdefault("identity", {})
        if getattr(agent, "gender", None) in {"male", "female"}:
            profile["identity"]["gender"] = "男" if agent.gender == "male" else "女"
        if not career:
            profile_career = profile.get("career")
            career = profile_career if isinstance(profile_career, dict) else None
        profile = _apply_postprocess_overrides(
            profile,
            agent_name=agent.name,
            career=career,
        )
    else:
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
    done_message = (
        "文档背景解析完成, 正在转换..."
        if profile_override is not None
        else "背景生成完成, 正在解析..."
    )
    await set_progress(agent.id, "llm_done", message=done_message)

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
