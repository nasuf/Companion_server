"""MBTI personality model — derived from 7-dim via LLM at agent creation.

Per spec《产品手册·背景信息》§1.2:
  系统根据用户设定的 7 维性格分数，调用大模型推测并生成 AI 的
  MBTI 八个维度（E/I、N/S、T/F、J/P）的百分比分数（0–100）。
  此后 AI 所有行为中涉及性格描述的部分，统一使用 MBTI 分数表达，
  替代原始 7 维描述。

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


_MBTI_PROMPT = """你是 MBTI 性格评估专家。基于以下 AI 角色的 7 维性格分数（0-100），
推测其 MBTI 八维度百分比分数（0-100）。

7 维分数（值越高，特质越强）：
{seven_dim}

请输出 JSON：
{{
  "EI": 65,    // 0-100，> 50 偏外向(E)，≤ 50 偏内向(I)
  "NS": 70,    // > 50 偏直觉(N)，≤ 50 偏感觉(S)
  "TF": 40,    // > 50 偏思考(T)，≤ 50 偏情感(F)
  "JP": 35,    // > 50 偏判断(J)，≤ 50 偏知觉(P)
  "type": "ENFP",   // 4 字母代号，按上面 4 个分数取 > 50 的字母
  "summary": "活泼开朗、富有想象力、感性细腻，倾向自由灵活的生活节奏"
}}

注意：
- 4 个数字必须 0-100 整数
- type 字段必须严格按 EI/NS/TF/JP 的分数 > 50 取首字母（E/N/T/J），≤ 50 取后字母（I/S/F/P）
- summary 80 字以内，自然中文，描述 AI 整体性格画像"""


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


def _validate_and_normalize(raw: Any) -> dict:
    """Coerce LLM output into the storage shape; clamp percentages to [0,100]
    and trust derived `type` over whatever the LLM emitted."""
    if not isinstance(raw, dict):
        return _default_mbti()
    out: dict = {}
    for k in ("EI", "NS", "TF", "JP"):
        v = raw.get(k, 50)
        try:
            iv = max(0, min(100, int(v)))
        except (TypeError, ValueError):
            iv = 50
        out[k] = iv
    out["type"] = _derive_type(out)  # always recompute (LLM 可能写错)
    summary = raw.get("summary") or ""
    out["summary"] = str(summary).strip()[:200]
    return out


def _default_mbti() -> dict:
    """Neutral fallback when LLM is unavailable. All midpoint."""
    return {"EI": 50, "NS": 50, "TF": 50, "JP": 50, "type": "ISFP", "summary": ""}


_BIG_FIVE_KEYS = {"extraversion", "openness", "conscientiousness", "agreeableness", "neuroticism"}


def _format_personality_input(scores: dict) -> str:
    """Render whatever shape the user supplied into the prompt-friendly bullet
    list. Accepts either:
      - 7-dim Chinese keys (活泼度...) with 0-100 ints
      - Big Five English keys (extraversion...) with 0-1 floats (multiplied
        to 0-100 for prompt clarity)
    Other shapes are rendered as-is.
    """
    if not scores:
        return "(无)"
    if _BIG_FIVE_KEYS.intersection(scores.keys()):
        # Big Five 0-1 → 0-100 for the prompt
        return "\n".join(
            f"- {k}: {round(v * 100)}" for k, v in scores.items()
        )
    return "\n".join(f"- {k}: {v}" for k, v in scores.items())


async def compute_mbti(personality_scores: dict) -> dict:
    """Generate MBTI from any 7-dim or Big Five personality dict via LLM.
    Falls back to neutral on failure."""
    if not personality_scores:
        return _default_mbti()
    prompt = _MBTI_PROMPT.format(seven_dim=_format_personality_input(personality_scores))
    try:
        result = await invoke_json(get_utility_model(), prompt)
    except Exception as e:
        logger.warning(f"MBTI LLM call failed: {e}; using neutral fallback")
        return _default_mbti()
    return _validate_and_normalize(result)


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
# 0-1 "lively" or "planned" or "extraversion" knob rather than 4 raw
# percentages. These helpers map MBTI dimensions to common axes so call
# sites don't have to know MBTI internals.

def signal(mbti: dict | None, name: str) -> float:
    """Derived 0-1 signal from MBTI. Returns 0.5 when mbti is None.

    Recognized signals (designed to mirror the old 7-dim semantics so
    consumers can swap in place):
      - 'extraversion' / 'lively'   ← EI / 100
      - 'rational'                  ← TF / 100   (T 偏理性)
      - 'emotional'                 ← (100 - TF) / 100   (F 偏感性)
      - 'planned'                   ← JP / 100   (J 偏计划)
      - 'spontaneous'               ← (100 - JP) / 100   (P 偏知觉)
      - 'creative' / 'imaginative'  ← NS / 100   (N 偏直觉)
      - 'agreeableness'             ← (100 - TF) / 100  + bias
      - 'openness'                  ← NS / 100
      - 'conscientiousness'         ← JP / 100
      - 'neuroticism'               ← clamp(0.5 + (50 - TF)/200 + (50 - JP)/200, 0, 1)
        (近似: 越偏 F + 越偏 P，神经质略高)
    """
    if not mbti:
        return 0.5
    ei = mbti.get("EI", 50)
    ns = mbti.get("NS", 50)
    tf = mbti.get("TF", 50)
    jp = mbti.get("JP", 50)
    mapping = {
        "extraversion": ei / 100,
        "lively": ei / 100,
        "rational": tf / 100,
        "emotional": (100 - tf) / 100,
        "planned": jp / 100,
        "spontaneous": (100 - jp) / 100,
        "creative": ns / 100,
        "imaginative": ns / 100,
        "openness": ns / 100,
        "conscientiousness": jp / 100,
        # F 偏宜人 (intuition + empathy bias)
        "agreeableness": ((100 - tf) + ns) / 200,
        # 简化近似：F + P 略升高神经质感
        "neuroticism": max(0.0, min(1.0, 0.5 + ((50 - tf) + (50 - jp)) / 200)),
        # 幽默 = 外向 + 直觉
        "humor": (ei + ns) / 200,
    }
    return mapping.get(name, 0.5)


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
