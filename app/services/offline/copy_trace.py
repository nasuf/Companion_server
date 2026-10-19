"""Code-level decisions on the activity copy path, as Trace steps.

Without these, a draft rejected by ``copy_guards`` and its repair appear as two
identical writer steps, and nothing in the trace says why the page shows the
template (production 2026-10-18, trace b4f089ca).
"""
from __future__ import annotations

from typing import Any
import hashlib

from app.services.chat.local_tracer import record_manual_step

STAGES = {"card_copy": "卡片短文案", "detail": "详情文案", "note": "推荐寄语", "invite": "聊天邀请语"}
# What the user ended up seeing, in a few characters.
_OUTCOME = {
    "verified": "模型文案",
    "verified_no_evidence": "模型文案",
    "unchecked": "模型文案（未核验）",
    "deterministic": "固定文案",
    "fallback_no_proposal": "模板（选卡没拿到模型结果）",
    "fallback_rejected": "模板（被拒）",
    "fallback_invalid": "模板（被拒）",
    "fallback_unavailable": "模板（生成或核验失败）",
    "fallback_too_short": "模板（输出过短）",
    "fallback_empty": "模板（模型没给文案）",
    "fallback_not_published": "代码文案（提示词未发布）",
    "verified_partial": "模型文案（个别项改用默认）",
}


def _named(label: str, subject: str | None) -> str:
    # A preview builds five cards at once; their steps interleave in one trace.
    return f"{label} · {subject[:14]}" if subject else label


def record_guard_block(stage: str, attempt: int, issues: list[str], *, subject: str | None = None) -> None:
    """A draft stopped by code before any model review."""
    record_manual_step(
        name="offline_copy_guard",
        display_name=_named(f"代码核验 · {STAGES[stage]}第 {attempt + 1} 稿", subject),
        decision_label=("拦下：" + issues[0][:36]) if issues else "拦下",
        inputs={"stage": stage, "draft": attempt + 1, "activity": subject},
        outputs={"issues": issues},
    )


def record_selection_fallback(reason: str, *, subject: str | None = None) -> None:
    """Card selection produced nothing usable: the first candidate and template copy are used.

    The cancelled model calls themselves stay "pending" with no output.
    """
    record_manual_step(
        name="offline_selection_fallback",
        display_name=_named("推荐卡生成 · 未拿到模型结果", subject),
        decision_label=f"{reason}，改用默认候选和模板摘要",
        inputs={"activity": subject},
        outputs={"reason": reason},
    )


def record_evidence_withheld(reason: str | None) -> None:
    """Recent dialogue and emotion memories are not used as recommendation reasons."""
    record_manual_step(
        name="offline_evidence_withheld",
        display_name="用户依据 · 近期有危机照护" if reason == "crisis_care" else "用户依据 · 危机照护检查不可用",
        decision_label="不用最近对话和情绪类记忆，只用长期兴趣",
        outputs={"reason": reason},
    )


def record_copy_outcome(
    statuses: dict[str, Any], issues: dict[str, Any] | None = None, *, subject: str | None = None,
) -> None:
    """Which copy the user actually sees, per field, with the stored reasons."""
    shown = {stage: str(statuses.get(stage) or "not_recorded") for stage in STAGES if stage in statuses}
    record_manual_step(
        name="offline_copy_outcome",
        display_name=_named("活动文案结果", subject),
        decision_label=" · ".join(f"{STAGES[k]}：" + (
            "模型文案（已保留通过段落）" if k == "detail" and v == "verified_partial" else _OUTCOME.get(v, v))
            for k, v in shown.items()),
        inputs={"activity": subject},
        outputs={"statuses": shown, "issues": issues or {}},
    )


def record_final_copy(activity: dict[str, Any], invite: str) -> None:
    """Only persisted, client-visible copy, with hashes for exact comparisons.

    No private evidence or raw provider payload is included. The existing trace
    ownership checks still govern access to these user-visible messages.
    """
    metadata = activity.get("discovery_metadata") or {}
    fields = {"summary": str(activity.get("summary") or ""),
              "description": str(activity.get("description") or ""),
              "recommendation_message": str(activity.get("recommendation_message") or metadata.get("recommendation_message") or ""),
              "invite": invite}
    record_manual_step(
        name="offline_copy_final",
        display_name=_named("活动最终展示文案", activity.get("title")),
        decision_label="已入库并用于接口及消息展示；以此节点对照 App",
        inputs={"activity_id": activity.get("id")},
        outputs={"fields": {key: {"text": text[:2400], "length": len(text), "truncated": len(text) > 2400,
                                     "sha256": hashlib.sha256(text.encode()).hexdigest()} for key, text in fields.items()},
                 "statuses": {"detail": metadata.get("copy_status", "not_recorded"),
                              "card_copy": metadata.get("card_copy_status", "not_recorded"),
                              "note": metadata.get("recommendation_message_status", "not_recorded"),
                              "invite": metadata.get("invite_message_status", "not_recorded")},
                 "sections": {key: metadata.get(key) for key in ("copy_section_count", "copy_kept_sections", "copy_dropped_sections")},
                 "writing_inputs": {key: metadata.get(key) for key in (
                     "detail_input_summary_status", "card_copy_accepted_fields", "card_copy_repair_writer", "context_source_status", "context_source_count",
                     "context_search_count", "context_read_count", "relevance_screen", "recommendation_message_length",
                     "recommendation_message_length_bounds", "recommendation_message_writer", "recommendation_message_repair_route")},
                 "visit_timing": metadata.get("visit_timing")},
    )
