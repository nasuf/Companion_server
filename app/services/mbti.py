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


async def compute_mbti(seven_dim: dict) -> dict:
    """Generate MBTI from 7-dim via LLM. Falls back to neutral on failure."""
    if not seven_dim:
        return _default_mbti()
    seven_text = "\n".join(f"- {k}: {v}" for k, v in seven_dim.items())
    prompt = _MBTI_PROMPT.format(seven_dim=seven_text)
    try:
        result = await invoke_json(get_utility_model(), prompt)
    except Exception as e:
        logger.warning(f"MBTI LLM call failed: {e}; using neutral fallback")
        return _default_mbti()
    return _validate_and_normalize(result)


def get_mbti(agent: Any) -> dict | None:
    """Returns the stored MBTI dict, or None if not yet computed."""
    raw = getattr(agent, "mbti", None)
    if isinstance(raw, dict) and raw.get("type") in _VALID_TYPES:
        return raw
    if isinstance(raw, str):  # defensive: in case Prisma serialized it
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("type") in _VALID_TYPES:
                return parsed
        except Exception:
            pass
    return None


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
