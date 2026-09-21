from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from app.services.memory.recording.pipeline import process_memory_pipeline
from app.services.memory.storage.persistence import store_memory
from app.services.runtime.tasks import fire_background

logger = logging.getLogger(__name__)


def remember_user_event(
    *,
    user_id: str,
    workspace_id: str | None,
    text: str,
    evidence_message_ids: list[str] | None = None,
) -> None:
    if not text.strip():
        return
    fire_background(
        process_memory_pipeline(
            user_id,
            text.strip(),
            side="user",
            workspace_id=workspace_id,
            statement_time=datetime.now(UTC),
            evidence_message_ids=evidence_message_ids,
        )
    )


def remember_ai_event(
    *,
    user_id: str,
    workspace_id: str | None,
    text: str,
    evidence_message_ids: list[str] | None = None,
) -> None:
    if not text.strip():
        return
    fire_background(
        process_memory_pipeline(
            user_id,
            text.strip(),
            side="ai",
            workspace_id=workspace_id,
            statement_time=datetime.now(UTC),
            evidence_message_ids=evidence_message_ids,
        )
    )


def remember_offline_fragment(
    *,
    user_id: str,
    workspace_id: str | None,
    text: str,
    location: str | None = None,
) -> None:
    """线下活动思绪碎片 → 保证写入 AI 记忆（memories_ai），不走"记/不记"预筛。

    产品决策（2026-09）：因活动而生成的思绪碎片必须落入 agent 记忆，不能游离于聊天
    系统之外。碎片是已产出的、结构化的既成事件，与共同游戏记忆同理——会话式记忆门控
    可能把"这一幕挺好的"判为不值得记，从而丢掉产品承诺要留住的东西。故走 store_memory
    直存（仍享 taxonomy 校验/embedding/去重/changelog/缓存失效），不经预筛。
    落 L3 + 低 importance，像真人回忆一样自然淡出；被反复提起再由 L2 动态升。
    """
    if not text.strip():
        return
    entities = [e for e in [location] if e and str(e).strip()]

    async def _store() -> None:
        try:
            await store_memory(
                user_id=user_id,
                content=text.strip(),
                level=3,
                importance=0.45,
                memory_type="life",
                main_category="生活",
                sub_category="交互",
                source="ai",
                statement_time=datetime.now(UTC),
                workspace_id=workspace_id,
                entities=entities,
                topics=["线下活动", "思绪碎片"],
            )
        except Exception as exc:  # 后台写入：失败只记日志，不影响碎片已交付的主流程
            logger.warning("[offline-fragment-memory] 写入失败 user=%s err=%s", user_id[:8], exc)

    fire_background(_store())


async def remember_shared_game_experience(
    *,
    user_id: str,
    workspace_id: str | None,
    user_text: str,
    ai_text: str,
    agent_name: str,
    game_title: str,
    sides: tuple[str, ...] = ("user", "ai"),
) -> dict[str, Any]:
    """Persist a completed game as a guaranteed two-sided shared memory.

    Game sessions are already structured, verified events. Running them through
    the conversational "remember / do not remember" gate can discard the very
    shared experience the product promises to keep, so this path starts at the
    existing storage layer instead. It still receives taxonomy validation,
    embeddings, semantic reconciliation, changelog, cache invalidation, and
    achievement hooks from ``store_memory``.
    """

    statement_time = datetime.now(UTC)
    topics = [game_title, "共同游戏"]
    entities = [agent_name, game_title]

    async def _store(side: str) -> str | None:
        is_ai = side == "ai"
        text = ai_text if is_ai else user_text
        return await store_memory(
            user_id=user_id,
            content=text,
            # L3 而不是 L2 —— 游戏记忆该像真人一样快速淡出。
            #
            # 之前是 level=2 / importance 0.74-0.80, 那个分数已经逼近 L1 阈值
            # (0.85), 而记的是"走了97步，4分钟"这类流水。真朋友一起下二十盘棋,
            # 隔天能想起的可能就一两盘 —— 而且想起的是"那次你连跳七格反超",
            # 不是统计量。
            #
            # 调用方 (games/native.py) 现在只在这一局客观稀有时才写, 所以进来的
            # 都是值得留一下的; 但"值得留一下"≠"该和用户父亲生病同等重要"。
            # 落 L3 让惰性衰减自然处理它, 真被反复提起的会通过 L2 动态升上去。
            level=3,
            importance=0.45 if is_ai else 0.42,
            memory_type="life",
            main_category="生活",
            sub_category="交互" if is_ai else "其他特殊事件",
            source=side,
            statement_time=statement_time,
            workspace_id=workspace_id,
            entities=entities,
            topics=topics,
        )

    requested_sides = tuple(side for side in ("user", "ai") if side in sides)
    raw_results = await asyncio.gather(
        *(_store(side) for side in requested_sides),
        return_exceptions=True,
    )
    ids: dict[str, str | None] = {"user": None, "ai": None}
    errors: list[str] = []
    for side, value in zip(requested_sides, raw_results, strict=True):
        if isinstance(value, BaseException):
            errors.append(side)
            logger.error(
                "Shared game memory failed side=%s user=%s game=%s: %s",
                side,
                user_id[:8],
                game_title,
                value,
            )
        else:
            ids[side] = value

    stored_count = sum(memory_id is not None for memory_id in ids.values())
    if errors and stored_count:
        status = "partial"
    elif errors:
        status = "failed"
    elif stored_count:
        status = "stored"
    else:
        status = "deduplicated"
    return {
        "status": status,
        "user_memory_id": ids["user"],
        "ai_memory_id": ids["ai"],
        "failed_sides": errors,
    }
