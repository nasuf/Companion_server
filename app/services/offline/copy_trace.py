"""Code-level decisions on the activity copy path, as Trace steps.

Without these, a draft rejected by ``copy_guards`` and its repair appear as two
identical writer steps, and nothing in the trace says why the page shows the
template (production 2026-10-18, trace b4f089ca).
"""
from __future__ import annotations

from typing import Any

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
        decision_label=" · ".join(f"{STAGES[k]}：{_OUTCOME.get(v, v)}" for k, v in shown.items()),
        inputs={"activity": subject},
        outputs={"statuses": shown, "issues": issues or {}},
    )
