"""MBTI personality model — derived from 7-dim via LLM at agent creation.

Spec verbatim (《产品手册·背景信息》§1.2):
  系统根据用户设定的 7 维性格分数，调用大模型推测并生成 AI 的
  MBTI 八个维度（E/I、N/S、T/F、J/P）的百分比分数（0–100）。
  此后 AI 所有行为中涉及性格描述的部分，统一使用 MBTI 分数表达，
  替代原始 7 维描述。

Implementation note — 存储和 prompt 输出按 4 轴偏向（而非 spec 字面的 8 独立数）:
  MBTI 本质是 4 条双极轴 (E-I / S-N / T-F / J-P)，每轴只能偏向一端，
  另一端由 100 减得出。spec "八个维度" 是 "4 轴 × 2 极" 的口语化表达。
  8 个独立百分比是 Big Five 模型的范畴，与 MBTI 双极假设不符。
  端到端只用 4 轴偏向（LLM prompt 输出 4 个数、存储 4 键）更干净。

Storage: agent.mbti JSON =
    { "EI": int 0-100,   # 100 = pure E, 0 = pure I
      "NS": int 0-100,   # 100 = pure N, 0 = pure S
      "TF": int 0-100,   # 100 = pure T, 0 = pure F
      "JP": int 0-100,   # 100 = pure J, 0 = pure P
      "type": "ENFP" 4-letter code,
      "summary": "1-2 sentence flavor text" }

Note: a value > 50 means the first letter (E/N/T/J), <= 50 means the
second (I/S/F/P).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.services.llm.models import get_utility_model, invoke_json

logger = logging.getLogger(__name__)


_MBTI_SUMMARY_PROMPT = """你是 MBTI 性格评估专家。给定一个 AI 角色的 MBTI 类型，
请用 80 字以内的自然中文写一段描述其整体性格画像的 summary。

MBTI 类型：{type}
四个维度强度（0-100，值越高越偏向首字母）：
- E/I: {ei}（>50 偏外向）
- N/S: {ns}（>50 偏直觉）
- T/F: {tf}（>50 偏思考）
- J/P: {jp}（>50 偏判断）

直接输出 JSON: {{"summary": "..."}} ，不要其他内容。"""


_VALID_TYPES = {
    f"{a}{b}{c}{d}"
    for a in "EI" for b in "NS" for c in "TF" for d in "JP"
}


def _derive_type(mbti: dict) -> str:
    """4-letter type from the 4 percentage scores (single source of truth)."""
    return (
        ("E" if mbti.get("EI", 50) > 50 else "I")
        + ("N" if mbti.get("NS", 50) > 50 else "S")
        + ("T" if mbti.get("TF", 50) > 50 else "F")
        + ("J" if mbti.get("JP", 50) > 50 else "P")
    )


MBTI_DIMS = ("EI", "NS", "TF", "JP")
_VALID_INPUT_KEYS = set(MBTI_DIMS)


def _clamp_pct(v) -> int:
    """强制 0-100 整数。LLM / 用户输入的非数值会抛 ValueError/TypeError。"""
    return max(0, min(100, int(v)))


async def seven_dim_to_mbti(p: dict) -> dict[str, int]:
    """Spec §1.2 + 指令模版 P25-26「AI性格打分」：7 维 → MBTI 4 轴偏向百分比。

    在 agent 创建的后台异步任务中调用（`api/public/agents.py:_init_mbti_then_emotion`），
    不阻塞 API 响应。若 LLM 调用失败或输出非法，抛异常由上层捕获。

    关于 4 轴 vs spec 字面 "8 维" 的语义差异见模块顶部 Implementation note。

    输入字段命名兼容 spec（liveliness/rationality/sensitivity/planning/
    spontaneity/imagination/humor）与代码内命名（lively/rational/emotional/
    planned/spontaneous/creative/humor）两种。
    """
    from app.services.prompting.store import get_prompt_text

    def _pick(*keys: str, default: int = 50) -> int:
        for k in keys:
            if k in p:
                try:
                    return _clamp_pct(p[k])
                except (TypeError, ValueError):
                    pass
        return default

    template = await get_prompt_text("agent.personality_scoring")
    prompt = template.format(
        liveliness=_pick("liveliness", "lively"),
        rationality=_pick("rationality", "rational"),
        sensitivity=_pick("sensitivity", "emotional"),
        planning=_pick("planning", "planned"),
        spontaneity=_pick("spontaneity", "spontaneous"),
        imagination=_pick("imagination", "creative"),
        humor=_pick("humor"),
    )
    result = await invoke_json(get_utility_model(), prompt)
    if not isinstance(result, dict):
        raise ValueError(f"personality_scoring LLM returned non-dict: {type(result).__name__}")

    return {dim: _clamp_pct(result.get(dim, 50)) for dim in MBTI_DIMS}


def _validate_input(percentages: dict) -> dict[str, int]:
    """Validate user-supplied 4 MBTI percentages. Strict: keys must be
    exactly EI/NS/TF/JP, values must be int-coercible and clamped to
    [0, 100]. Raises ValueError on bad input — there is no fallback,
    the API layer should reject the request.
    """
    missing = _VALID_INPUT_KEYS - set(percentages.keys())
    if missing:
        raise ValueError(f"MBTI input missing keys: {sorted(missing)}")
    unknown = set(percentages.keys()) - _VALID_INPUT_KEYS
    if unknown:
        raise ValueError(f"MBTI input has unknown keys: {sorted(unknown)}")
    out: dict[str, int] = {}
    for k in _VALID_INPUT_KEYS:
        try:
            out[k] = _clamp_pct(percentages[k])
        except (TypeError, ValueError):
            raise ValueError(f"MBTI dimension '{k}' must be int 0-100, got {percentages[k]!r}")
    return out


async def build_mbti(percentages: dict) -> dict:
    """Build the canonical agent.mbti JSON shape from user-supplied 4
    percentages. Calls a small LLM only to flesh out the `summary` text;
    the type and percentages come straight from the input.

    Per spec §1.2 起 MBTI 直接由用户填写，不再有 7 维中转层。
    """
    pct = _validate_input(percentages)
    type_str = _derive_type(pct)
    summary = await _generate_summary(pct, type_str)
    return {**pct, "type": type_str, "summary": summary}


async def _generate_summary(pct: dict[str, int], type_str: str) -> str:
    """LLM-generate an 80-字 summary. Falls back to "" on LLM failure;
    callers must tolerate empty summary (it's flavor text, not core data)."""
    prompt = _MBTI_SUMMARY_PROMPT.format(
        type=type_str,
        ei=pct["EI"], ns=pct["NS"], tf=pct["TF"], jp=pct["JP"],
    )
    try:
        result = await invoke_json(get_utility_model(), prompt)
    except Exception as e:
        logger.warning(f"MBTI summary LLM call failed: {e}")
        return ""
    if isinstance(result, dict):
        return str(result.get("summary", "")).strip()[:200]
    return ""


def _coerce(raw: Any) -> dict | None:
    if isinstance(raw, dict) and raw.get("type") in _VALID_TYPES:
        return raw
    if isinstance(raw, str):  # defensive: Prisma serialized
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("type") in _VALID_TYPES:
                return parsed
        except Exception:
            pass
    return None


def get_mbti(agent: Any) -> dict | None:
    """Returns the current effective MBTI: currentMbti (post-adjustment)
    if present, else the initial mbti, else None."""
    return (
        _coerce(getattr(agent, "currentMbti", None))
        or _coerce(getattr(agent, "mbti", None))
    )


def get_initial_mbti(agent: Any) -> dict | None:
    """Returns the as-created MBTI snapshot (ignores trait adjustments).
    Used by trait_adjustment to enforce the ±10 drift cap."""
    return _coerce(getattr(agent, "mbti", None))


# ── Derived signals ──
# Many downstream modules (style / schedule / emotion / topic) need a
# 0-1 single-letter knob rather than 4 raw
# percentages. These helpers map MBTI dimensions to common axes so call
# sites don't have to know MBTI internals.

# Letter → (raw_field, is_positive_side)
# 约定 EI 字段值 >50 偏 E；NS >50 偏 N；TF >50 偏 T；JP >50 偏 J。
# 因此 E/N/T/J 是各字段的"正向字母"，I/S/F/P 是其互补。
_LETTER_MAP: dict[str, tuple[str, bool]] = {
    "E": ("EI", True),  "I": ("EI", False),
    "N": ("NS", True),  "S": ("NS", False),
    "T": ("TF", True),  "F": ("TF", False),
    "J": ("JP", True),  "P": ("JP", False),
}


def signal(mbti: dict | None, letter: str) -> float:
    """Return the 0-1 strength of a single MBTI letter.

    letter ∈ {E, I, N, S, T, F, J, P}. Each maps directly to one of the
    4 stored percentages (or its complement). No invented derived names
    — composite signals like "humor" must be assembled at the call site
    so the formula is visible, e.g.:

        humor = (signal(m, "E") + signal(m, "N")) / 2

    Returns 0.5 when mbti is None (neutral fallback for legacy agents).
    """
    if letter not in _LETTER_MAP:
        raise ValueError(
            f"Unknown MBTI letter '{letter}'; expected one of "
            f"{sorted(_LETTER_MAP.keys())}"
        )
    if not mbti:
        return 0.5
    field, is_positive = _LETTER_MAP[letter]
    raw = mbti.get(field, 50) / 100
    return raw if is_positive else 1 - raw


def format_mbti_for_prompt(mbti: dict | None) -> str:
    """Compact one-line description for prompt injection.

    Returns an empty string when no MBTI is available — caller can then
    fall back to whatever 7-dim text it was using.
    """
    if not mbti:
        return ""
    parts = [
        f"E={mbti['EI']}/I={100 - mbti['EI']}",
        f"N={mbti['NS']}/S={100 - mbti['NS']}",
        f"T={mbti['TF']}/F={100 - mbti['TF']}",
        f"J={mbti['JP']}/P={100 - mbti['JP']}",
    ]
    summary = mbti.get("summary") or ""
    head = f"MBTI {mbti.get('type', '')}（{' · '.join(parts)}）"
    return f"{head} {summary}".strip()
