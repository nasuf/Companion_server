"""意图路由修正 — LLM 意图识别的规则后处理层（orchestrator 拆分 R3）。

统一意图识别 LLM 对以下两类消息识别过度宽松，需要规则纠正：
- 记忆/身份追问被误标"询问当前状态"（downgrade → NONE，让记忆检索驱动回复）
- 社交寒暄 ("好几天没找你聊了") 被误标"计划查询 current"（downgrade / reroute）

这些函数是纯决策转换：IntentResult in → IntentResult out，副作用仅写
response_diagnostics（诊断可观测）。长期方向是优化 intent.unified prompt
让首次识别就精确，减少此层规则（见 CLAUDE.md 调试提示）。
"""

from __future__ import annotations

from typing import Any

from app.services.chat.intent_dispatcher import (
    IntentResult,
    IntentType,
    infer_schedule_query_type,
    is_explicit_current_state_query,
)


def _downgrade_non_explicit_current_state(
    detected_intent: IntentResult,
    user_message: str,
    response_diagnostics: dict[str, Any],
) -> IntentResult:
    """Keep CURRENT_STATE only for explicit AI state questions.

    LLM intent can over-label memory/identity recall questions as current-state.
    If that label survives, the reply path cannot use memory tier prompts even
    after retrieval. Demote those misses to normal chat so memory relevance can
    drive the final reply.
    """
    if detected_intent.intent != IntentType.CURRENT_STATE:
        return detected_intent
    if is_explicit_current_state_query(user_message):
        return detected_intent

    metadata = dict(detected_intent.metadata or {})
    metadata["downgraded_from"] = IntentType.CURRENT_STATE.value
    metadata["downgrade_reason"] = "not_explicit_current_state"
    response_diagnostics["intent_downgrade_reason"] = "not_explicit_current_state"
    return IntentResult(
        intent=IntentType.NONE,
        confidence=detected_intent.confidence,
        metadata=metadata,
    )


def _route_current_schedule_query_to_current_state(
    detected_intent: IntentResult,
    user_message: str,
    response_diagnostics: dict[str, Any],
) -> IntentResult:
    """Treat present-tense availability questions as current-state chat.

    The schedule-query path is for concrete schedule/routine/date lookups. For
    "现在忙吗/你现在有空吗", using that path exposes the full schedule table to
    the reply prompt and makes the assistant sound like it is reading a diary.
    """
    if detected_intent.intent != IntentType.SCHEDULE_QUERY:
        return detected_intent
    if (detected_intent.metadata or {}).get("query_type") != "current":
        return detected_intent
    if not is_explicit_current_state_query(user_message):
        return detected_intent

    metadata = dict(detected_intent.metadata or {})
    metadata["rerouted_from"] = IntentType.SCHEDULE_QUERY.value
    metadata["reroute_reason"] = "current_availability_as_current_state"
    response_diagnostics["intent_reroute_reason"] = "current_availability_as_current_state"
    return IntentResult(
        intent=IntentType.CURRENT_STATE,
        confidence=detected_intent.confidence,
        metadata=metadata,
    )


def _downgrade_non_explicit_current_schedule_query(
    detected_intent: IntentResult,
    user_message: str,
    response_diagnostics: dict[str, Any],
) -> IntentResult:
    """Keep current schedule-query only when the user actually asks.

    The unified classifier can over-label social openers such as "好几天没找你聊了"
    as "计划查询" with query_type=current. If that survives, a plain greeting
    takes the schedule short-circuit path and may spawn extra sub-intents.
    """
    if detected_intent.intent != IntentType.SCHEDULE_QUERY:
        return detected_intent
    if (detected_intent.metadata or {}).get("query_type") != "current":
        return detected_intent
    if infer_schedule_query_type(user_message, require_query_cue=True):
        return detected_intent
    if is_explicit_current_state_query(user_message):
        return detected_intent

    metadata = dict(detected_intent.metadata or {})
    metadata["downgraded_from"] = IntentType.SCHEDULE_QUERY.value
    metadata["downgrade_reason"] = "not_explicit_current_schedule_query"
    metadata.pop("fragments", None)
    response_diagnostics["intent_downgrade_reason"] = "not_explicit_current_schedule_query"
    return IntentResult(
        intent=IntentType.NONE,
        confidence=detected_intent.confidence,
        metadata=metadata,
    )


def _filter_non_explicit_sub_fragments(
    fragments: dict[str, str],
    response_diagnostics: dict[str, Any],
) -> dict[str, str]:
    """Drop over-eager current-state/schedule sub-fragments from casual text."""
    filtered: dict[str, str] = {}
    dropped: list[str] = []
    for label, text in fragments.items():
        fragment = str(text).strip()
        if not fragment:
            continue
        if label == "询问当前状态" and not is_explicit_current_state_query(fragment):
            dropped.append(label)
            continue
        if (
            label == "计划查询"
            and not infer_schedule_query_type(fragment, require_query_cue=True)
            and not is_explicit_current_state_query(fragment)
        ):
            dropped.append(label)
            continue
        filtered[label] = fragment
    if dropped:
        response_diagnostics["intent_sub_fragments_dropped"] = dropped
    return filtered
