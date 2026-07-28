"""Async memory pipeline.

Orchestrates: extract -> score -> dedup -> store -> embed -> entity link.
Runs as FastAPI BackgroundTasks (non-blocking).

Spec 第二部分 §2.1 / §2.2：用户侧与 AI 侧各走一条独立管线，`side` 决定抽取 prompt 与
存储归属（B 库 / A 库）。owner 不再由 LLM 推断。

L1 冲突处理完全走 spec §4 交互矛盾机制（热路径询问用户确认）—
录入期不自动降级/修改 L1，避免绕过用户确认（spec §1.5.1）。
"""

import logging
from datetime import datetime, time
from typing import Literal

from app.observability.events import EVT_MEMORY_IDENTITY_REPAIR, EVT_MEMORY_STORED
from app.services.memory.storage.entity_repo import (
    record_entities_for_memory,
    record_preferences_for_memory,
    record_topics_for_memory,
)
from app.services.memory.recording.extraction import extract_memories
from app.services.memory.recording.filter import should_extract_memory
from app.services.memory.recording.pre_filter import (
    is_user_fact_acknowledgement,
    should_memorize,
)
from app.services.memory import provenance as provenance_mod
from app.services.memory.storage.persistence import store_memory, log_memory_changelog
from app.services.memory.lifecycle.quality import log_memory_evidence
from app.services.memory.config import level_for_importance, RECURRENCE_PERIODIC
from app.services.memory.recording.identity_repair import (
    repair_identity_classification,
)
from app.services.memory.taxonomy import SUBCATEGORY_ALIASES
from app.services.schedule_domain.time_parser import (
    has_explicit_time,
    parse_with_statement_time,
)
from app.services.schedule_domain.time_service import _now_corrected, ensure_aware
from app.services.workspace.workspaces import get_active_workspace, resolve_workspace_id

logger = logging.getLogger(__name__)

Side = Literal["user", "ai"]


class MemoryExtractionError(RuntimeError):
    """Raised when the extraction LLM hard-fails (timeout / provider error).

    Distinct from a legitimately empty extraction: the caller must NOT advance
    the per-side watermark on this error, otherwise the un-extracted messages
    are lost forever. A later batch will re-cover them.
    """

async def _create_reminder_timetrigger(
    *,
    user_id: str,
    memory_id: str,
    summary: str,
    occur_time: datetime,
    recurrence: str,
    side: Side,
) -> None:
    """统一提醒系统 (Phase 4.1): 提醒入库 → 同步建 timetrigger.

    spec §4.2 字面是"凌晨扫记忆 → 起床后 idle 触发", 但跟 §5.2 自相矛盾且不支持
    短期闹钟. 改为按精确 occur_time 直接调度 timetrigger; 若 parser 把"明天提醒
    我X"解析为全天范围 (00:00:00) → 退化到起床后第一个空闲段 (复用 special_dates
    的 find_first_idle_after_wakeup, 跟节日/生日触发对齐).

    agent_id 通过 user_id 反查当前 active workspace 取得. 无 active workspace →
    memory 仍入库, 但 trigger 不会创建; Phase 4.2 起 special_dates 路径已删除提醒
    扫描, 没有 fallback. 这里以 WARNING 记录方便 admin 排查丢失的提醒.
    """
    from app.db import db

    try:
        workspace = await get_active_workspace(user_id=user_id)
    except Exception as e:
        logger.warning(f"[MEM-{side}] reminder timetrigger: workspace lookup failed: {e}")
        return
    if not workspace:
        logger.warning(
            f"[MEM-{side}] reminder timetrigger SKIPPED: no active workspace for "
            f"user={user_id[:8]} memory={memory_id[:8]}; reminder will NOT fire"
        )
        return

    agent_id = getattr(workspace, "agentId", None) or getattr(workspace, "ai_agent_id", None)
    if not agent_id:
        logger.warning(
            f"[MEM-{side}] reminder timetrigger SKIPPED: workspace has no agentId "
            f"user={user_id[:8]} memory={memory_id[:8]}"
        )
        return

    trigger_time = occur_time
    # Heuristic: parser's _day_range emits start=00:00:00 with no time component.
    # Real "提醒我 0 点喝水" is rare enough we accept the edge case (still fires
    # at the next idle slot, off by ~30min — better than waking at midnight).
    if occur_time.time() == time(0, 0, 0):
        try:
            from app.services.proactive.special_dates import find_first_idle_after_wakeup
            from app.services.schedule_domain.schedule import get_cached_schedule
            schedule = await get_cached_schedule(agent_id)
            trigger_time = find_first_idle_after_wakeup(schedule, occur_time.date())
        except Exception as e:
            logger.warning(
                f"[MEM-{side}] reminder timetrigger: idle fallback failed ({e}); "
                f"using raw occur_time {occur_time}"
            )

    # upsert: 已有 (agent, memory_id) active trigger → update triggerTime + reset
    # lastFired (重设语义); 否则 create. 完整逻辑在 services/reminder/scheduling.py
    # 收口 — 三处历史调用 (pipeline / handler / cancel) 共享一致行为.
    from app.services.reminder.scheduling import upsert_reminder_trigger
    await upsert_reminder_trigger(
        user_id=user_id,
        agent_id=agent_id,
        memory_id=memory_id,
        summary=summary,
        trigger_time=trigger_time,
        recurrence=recurrence,
        side=side,  # type: ignore[arg-type]
    )


# Spec §1.5.2: keywords indicating user expressed that information is important.
# Detected once per pipeline call (on new_conversation), tagged on all extracted
# memories so L2→L1 promotion can verify the condition.
_IMPORTANCE_EXPRESSIONS = (
    "很重要", "一定要记住", "千万别忘", "记住了", "别忘了",
    "这很关键", "非常重要", "特别重要", "务必记住",
)
_AI_EPISTEMIC_UNCERTAINTY_MARKERS = (
    "记不清", "记不太清", "不太记得", "没印象", "没有印象",
    "不确定", "不太确定", "好像", "大概", "可能", "应该", "似乎",
    "印象里", "猜", "估计",
)

# Anti-drift: the AI's stable persona (preferences + identity) is seeded once
# from the creation profile. Chat-time extraction must NEVER mint a NEW 偏好/身份
# self-memory from the AI's own generated replies — that is exactly how a
# hallucinated "我喜欢浅紫色" / "我今年30岁" gets fossilized into a fake persona
# fact that then contradicts the real profile and compounds drift. Episodic
# self-memory (生活/情绪 experiences, e.g. the daily life summary) is unaffected.
_AI_SELF_AUTHORED_BLOCKED_CATEGORIES = frozenset({"偏好", "身份"})


def _compact_text(text: str | None) -> str:
    return "".join((text or "").split())


def _is_uncertain_ai_self_memory(
    *,
    side: Side,
    new_conversation: str,
    content: str,
) -> bool:
    """Guard against persisting epistemically weak AI self claims.

    The extraction model may turn a conversational hedge ("记不清/好像/可能")
    into a structured self-memory. Those claims are not stable facts; keeping
    them out of memories_ai is safer than later asking retrieval to distinguish
    truth from a previous guess.
    """
    if side != "ai":
        return False
    text = _compact_text(f"{new_conversation}\n{content}")
    return any(marker in text for marker in _AI_EPISTEMIC_UNCERTAINTY_MARKERS)


_DEFAULT_IMPORTANCE = 0.5


def _coerce_importance(raw: object) -> float:
    """LLM 给的 importance → 可比较的 float, 坏值退回默认.

    退回 0.5 而不是丢弃: 0.5 落 L2, 是"存下来但不当核心"的保守选择 —— 抽取
    已经判定这条值得记, 不该因为一个字段格式问题整条丢掉.
    """
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _DEFAULT_IMPORTANCE
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return _DEFAULT_IMPORTANCE
    return min(1.0, max(0.0, value))


def _last_ai_line(context_conversation: str) -> str:
    """从 `role: content` 格式的上下文里取最后一句 AI 的话 —— 仅作兜底.

    ⚠️ 记忆管线跑在 AI 回复**之后**, 所以用户侧的 context 末尾往往是 AI 对这句话
    的**回复**, 而不是引出这句话的**提问**:

        assistant: 你今天还好吗      ← 判断 "不好" 需要的是这句
        user: 不好                   (进 new_conversation)
        assistant: 怎么了跟我说说    ← 这一行才是 context 的末尾

    所以调用方应当显式传入正确的那一句 (post_process 知道真实顺序). 这个函数只在
    调用方没传时兜底 —— 后一句同样透露了上文语气, 比完全没有上下文强, 但召回不是
    按它测的.
    """
    for line in reversed((context_conversation or "").splitlines()):
        prefix, sep, content = line.partition(": ")
        if sep and prefix.strip() == "assistant" and content.strip():
            return content.strip()
    return ""


async def process_memory_pipeline(
    user_id: str,
    new_conversation: str,
    *,
    context_conversation: str = "",
    statement_time: datetime | None = None,
    side: Side = "user",
    workspace_id: str | None = None,
    evidence_message_ids: list[str] | None = None,
    prev_ai_line: str | None = None,
) -> list[str]:
    """Run the full memory extraction and storage pipeline for one side.

    Args:
        new_conversation: Dialogue to extract from (post-watermark messages).
            Pre-filter + heuristic filter run against this block only.
        context_conversation: Prior dialogue for LLM disambiguation only (no
            extraction). Empty string when no pre-watermark history.
        side: "user" → spec §2.1，抽取用户记忆，存入 memories_user；
              "ai"   → spec §2.2，抽取 AI 自我记忆，存入 memories_ai。
        statement_time: Part 5 §3.1 说出这句话的时间（消息接收时刻）。
            未提供时由 store_memory 取 now() 作为最佳估计。

    Returns list of stored memory IDs.
    """
    workspace_id = await resolve_workspace_id(workspace_id=workspace_id, user_id=user_id)

    # Step 0: Heuristic filter — skip purely noise messages (no LLM call)
    # 只对 new_conversation 判定, 历史上下文已经抽过了
    if not should_extract_memory(new_conversation):
        logger.debug(f"[MEM-{side}] filtered by heuristic filter")
        return []

    # Step 1 (spec §2.1.2 / §2.2.2): Small model pre-filter — "记" or "不记"
    # Uses registered prompt `memory.judgement_{side}` via should_memorize.
    from app.config import settings
    if settings.enable_memory_prefilter:
        try:
            if not await should_memorize(
                new_conversation, side=side,
                prev_ai=(
                    prev_ai_line
                    if prev_ai_line is not None
                    else _last_ai_line(context_conversation)
                ),
            ):
                logger.debug(f"[MEM-{side}] pre-filter: 不记")
                return []
        except Exception as e:
            logger.warning(f"[MEM-{side}] pre-filter failed ({e}), proceeding")

    # Step 2 (spec §2.1.3 / §2.2.3): Big model extraction
    extraction = await extract_memories(
        new_conversation,
        context_conversation=context_conversation,
        side=side,
    )
    # Hard LLM failure → propagate so the caller keeps the watermark put and a
    # later batch retries. An empty-but-successful extraction (no error flag)
    # falls through and legitimately advances the watermark.
    if extraction.get("_extraction_error"):
        kind = extraction.get("_extraction_error_kind", "llm_failure")
        raise MemoryExtractionError(
            f"extraction did not run for side={side} ({kind}); "
            "watermark must not advance"
        )
    memories = extraction.get("memories", [])

    if not memories:
        logger.info(f"[MEM-{side}] no memories extracted")
        return []

    # Spec Part 5 §3.1: 解析过程不调大模型。规则引擎作为权威源；只有当
    # 消息里唯一匹配到 1 个时间表述时才覆盖 LLM 的 occur_time — 否则
    # 无法把单个时间归因到多条记忆中的哪一条，回退到 LLM 抽取字段。
    rule_based_occur_time: datetime | None = None
    if has_explicit_time(new_conversation):
        parsed = parse_with_statement_time(new_conversation, now=statement_time)
        if len(parsed.event_times) == 1:
            rule_based_occur_time = ensure_aware(parsed.event_times[0].start)

    stored_ids: list[str] = []

    # Batch-level entity type hints (name → type) recovered from the extraction
    # aggregate; per-memory entities are bare names, so we resolve their type
    # here without linking the whole aggregate to every memory.
    entity_type_by_name: dict[str, str] = {}
    for ent in extraction.get("entities", []):
        if isinstance(ent, dict):
            name = str(ent.get("name") or "").strip()
            if name:
                entity_type_by_name[name] = str(ent.get("type") or "other")
    batch_preferences = [
        pref for pref in extraction.get("preferences", []) if isinstance(pref, dict)
    ]

    # Spec §1.5.2 user emphasis only drives user-side L2→L1 promotion.
    user_emphasized = side == "user" and any(
        kw in new_conversation for kw in _IMPORTANCE_EXPRESSIONS
    )

    # Step 3: Store each memory with dedup and conflict check
    for mem in memories:
        content = mem.get("content", "")
        if side == "ai" and is_user_fact_acknowledgement(content):
            logger.info(
                f"[MEM-{side}] skipped user-fact acknowledgement: {content[:40]}"
            )
            continue
        if _is_uncertain_ai_self_memory(
            side=side,
            new_conversation=new_conversation,
            content=content,
        ):
            logger.info(
                f"[MEM-{side}] skipped uncertain self-memory: {content[:40]}"
            )
            continue
        # LLM 偶尔把 importance 输出成字符串 ("0.8") 或 null. 下面的分层比较和
        # clamp 都按数字走, 类型不对会抛 TypeError —— 而这个异常会被上层当成抽取
        # 失败, 水位线卡住不动, 整批消息反复重试同一条坏数据. 在边界收口一次.
        importance = _coerce_importance(mem.get("importance"))
        memory_type = mem.get("type")
        main_category = mem.get("main_category")
        sub_category = mem.get("sub_category")
        # ③ Block self-authored persona claims: the AI must not "learn" new
        # preferences/identity about itself from its own chat output (that is the
        # fabrication→fossilization drift vector). The profile is the sole source
        # of stable persona facts; episodic 生活/情绪 self-memory still flows.
        if side == "ai" and main_category in _AI_SELF_AUTHORED_BLOCKED_CATEGORIES:
            logger.info(
                f"[MEM-{side}] skipped self-authored persona claim "
                f"({main_category}/{sub_category}): {content[:40]}"
            )
            continue
        # 提前解 alias: 否则 LLM 输出"备忘"/"提醒我"等别名时, 下游 sub_category=="提醒"
        # 比对会漏判 → recurrence 检测被跳过 → store_memory 把已 alias 解析的"提醒"
        # 行写入但 recurrence=NULL, 一次性提醒过去时间不被降级.
        if sub_category and sub_category in SUBCATEGORY_ALIASES:
            sub_category = SUBCATEGORY_ALIASES[sub_category]

        # 身份事实兜底: LLM 对「用户叫X」这类句式的分类不稳定, 分错时姓名会落进
        # 身份/其他 并被压到 L2, 同时绕过 L1 singleton 保护 (详见 identity_repair).
        if side == "user":
            main_category, sub_category, importance, repair = (
                repair_identity_classification(
                    summary=content,
                    main_category=main_category,
                    sub_category=sub_category,
                    importance=importance,
                )
            )
            if repair:
                logger.info(
                    f"[MEM-{side}] identity repair ({repair}): {content[:40]}",
                    extra={"event": EVT_MEMORY_IDENTITY_REPAIR, "repair": repair},
                )

        # Reminder importance clamp: extraction prompt asks 0.4-0.6, but LLM
        # drifts upward; without the clamp 1 reminder/week could ride the L2
        # frequency factor up to L1 over a year. Upper bound 0.49 (NOT 0.6) so
        # importance always lands in L3 (level derivation: ≥0.50 → L2). Triggering
        # is by timetrigger directly (key by sub_category="提醒"), so layer
        # demotion is harmless to the reminder feature itself.
        if sub_category == "提醒":
            importance = max(0.4, min(0.49, float(importance)))

        # Per spec《产品手册·背景信息》§2.3 — level is derived from importance
        # score (0-100), not whatever level the LLM may have guessed:
        #   ≥ 0.85 → L1   |   0.50-0.84 → L2   |   0.10-0.49 → L3   |   < 0.10 → drop
        if importance < 0.10:
            logger.debug(f"Memory dropped (importance={importance:.2f} < 0.10): {content[:40]}")
            continue
        else:
            level = level_for_importance(importance)

        # Rule engine wins when it's unambiguous; LLM occur_time is fallback.
        # 必须 ensure_aware: LLM 输出 ISO 经常没 tz, fromisoformat 解出来是
        # naive, 跟下面 ref_now (aware) 比较时崩 "can't compare offset-naive
        # and offset-aware datetimes". 在边界统一规范化, 避免每个比较点打补丁.
        occur_time: datetime | None = rule_based_occur_time
        if occur_time is None:
            raw_time = mem.get("occur_time")
            if raw_time and isinstance(raw_time, str):
                try:
                    occur_time = ensure_aware(datetime.fromisoformat(raw_time))
                except ValueError:
                    pass

        # spec Part 5 §4.2: 提醒重复规则 (once|yearly|monthly|weekly|daily).
        # 仅 sub_category="提醒" 有效, 其他子类强制 None (store_memory 也兜底).
        recurrence: str | None = None
        if sub_category == "提醒":
            raw_rec = mem.get("recurrence")
            recurrence = raw_rec if raw_rec in RECURRENCE_PERIODIC else "once"

        # spec Part 5 §4.2: 一次性提醒 (recurrence=once) 必须含未来 event_time.
        # 周期提醒 (yearly/monthly/weekly/daily) occur_time 是首次发生时间,
        # 后续按 recurrence 自动重复, 不要求未来 (e.g. yearly 的 1995-03-20 合法).
        # 用 _now_corrected (tz-aware + NTP), 否则 datetime.now() 是 naive,
        # 跟 parser 的 tz-aware occur_time 比较会 TypeError.
        reminder_demoted_reason: str | None = None
        if sub_category == "提醒" and recurrence == "once":
            ref_now = statement_time or _now_corrected()
            if occur_time is None or occur_time <= ref_now:
                reminder_demoted_reason = (
                    "missing_occur_time" if occur_time is None
                    else f"past_occur_time={occur_time.isoformat()}"
                )
                logger.warning(
                    f"[MEM-{side}] 提醒记忆 occur_time 缺失或非未来 "
                    f"({occur_time}); 改归生活/其他: {content[:40]}"
                )
                sub_category = "其他"
                # recurrence 不重置: store_memory 的 sub_category!="提醒" 闸门会丢弃

        memory_id = await store_memory(
            user_id=user_id,
            content=content,
            level=level,
            importance=importance,
            memory_type=memory_type,
            main_category=main_category,
            sub_category=sub_category,
            occur_time=occur_time,
            statement_time=statement_time,
            workspace_id=workspace_id,
            source=side,
            recurrence=recurrence,
            entities=[str(e) for e in mem.get("entities", []) if e],
            topics=[str(t) for t in mem.get("topics", []) if t],
            # Provenance is decided by the pipeline side, not the LLM: user
            # utterances → user_stated, the AI's own replies → ai_authored.
            provenance=(
                provenance_mod.USER_STATED if side == "user"
                else provenance_mod.AI_AUTHORED
            ),
        )

        if memory_id:
            stored_ids.append(memory_id)
            try:
                await log_memory_evidence(
                    user_id=user_id,
                    memory_id=memory_id,
                    message_ids=evidence_message_ids or [],
                    workspace_id=workspace_id,
                )
            except Exception as e:
                logger.debug(f"[MEM-{side}] evidence link failed for {memory_id}: {e}")

            # 提醒入库 → 同步建 timetrigger (统一提醒系统).
            # 限 side=="user": 用户主动设置的事项才该建实际的提醒计划.
            # AI-side 反思 (e.g. "我差点又忘记提醒用户喝水") extraction LLM 偶尔
            # 误归 sub_category=="提醒" + 给一个 occur_time, 不能据此真的发提醒
            # — AI 的内省不该变成产品行为. 严格限制 side="user" 避免这类误触发.
            #
            # once 已经在前面校验过必须是未来; 周期性 (yearly/monthly/weekly/daily)
            # 即使 occur_time 是历史(首次发生时间)也合法, trigger handler 会按
            # recurrence 续期到下次.
            if (
                side == "user"
                and sub_category == "提醒"
                and occur_time is not None
                and recurrence
            ):
                await _create_reminder_timetrigger(
                    user_id=user_id,
                    memory_id=memory_id,
                    summary=content,
                    occur_time=occur_time,
                    recurrence=recurrence,
                    side=side,
                )

            # Log user emphasis so L2→L1 promotion can verify the condition
            if user_emphasized:
                try:
                    await log_memory_changelog(
                        user_id, memory_id, "user_emphasized",
                        new_value="用户表达了该信息的重要性",
                        workspace_id=workspace_id,
                    )
                except Exception:
                    pass

            # spec Part 5 §4.2: 提醒降级审计 trail, 让 admin 能 grep 看到
            if reminder_demoted_reason:
                try:
                    await log_memory_changelog(
                        user_id, memory_id, "reminder_past_time_demoted",
                        new_value=reminder_demoted_reason,
                        workspace_id=workspace_id,
                    )
                except Exception:
                    pass

            # Step 3: Link entities / topics / preferences to THIS memory.
            # Must use the memory's own entities/topics (not the batch-level
            # aggregate) — otherwise a batch that extracts N memories links
            # every entity/topic/preference to all N rows, polluting the entity
            # graph and entity-recall (e.g. a "cat" memory tagged with "work").
            # Best-effort: failure here is advisory (retrieval still works from
            # the memory row + pgvector) so we log-and-continue.
            try:
                mem_entities = [
                    {"name": name, "type": entity_type_by_name.get(name, "other")}
                    for name in mem.get("entities", [])
                    if name
                ]
                await record_entities_for_memory(
                    memory_id=memory_id,
                    memory_source=side,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    entities=mem_entities,
                )
                await record_topics_for_memory(
                    memory_id=memory_id,
                    memory_source=side,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    topics=[str(t) for t in mem.get("topics", []) if t],
                )
                # Preferences have no per-memory field in the schema; attribute
                # each batch preference only to the memory whose text actually
                # mentions its value (no cross-memory pollution).
                mem_prefs = [
                    pref for pref in batch_preferences
                    if str(pref.get("value") or "").strip()
                    and str(pref.get("value")).strip() in content
                ]
                if mem_prefs:
                    await record_preferences_for_memory(
                        memory_id=memory_id,
                        memory_source=side,
                        user_id=user_id,
                        workspace_id=workspace_id,
                        preferences=mem_prefs,
                    )
            except Exception as e:
                logger.warning(f"Entity linking failed for memory {memory_id}: {e}")

    logger.info(
        f"Pipeline complete: {len(stored_ids)}/{len(memories)} memories stored",
        extra={
            "event": EVT_MEMORY_STORED,
            "n_extracted": len(memories),
            "n_stored": len(stored_ids),
            "side": side,
        },
    )
    return stored_ids
