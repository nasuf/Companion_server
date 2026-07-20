"""Small-model memory pre-filter (spec §2.1.2 / §2.2.2).

Runs BEFORE the expensive big-model extraction step, saving cost on messages
that are clearly not memorable.

Uses registered prompts (managed via admin UI):
- `memory.judgement_user` — Spec §2.1.2 「记忆判断」
- `memory.judgement_ai`   — Spec §2.2.2 「AI信息记忆判断」

Output per spec: plain text "记" or "不记" (not JSON).
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from app.services.llm.models import get_utility_model, invoke_text
from app.services.prompting.store import get_prompt_text
from app.services.prompting.utils import SafeDict

logger = logging.getLogger(__name__)

Side = Literal["user", "ai"]

_PROMPT_KEY_BY_SIDE: dict[Side, str] = {
    "user": "memory.judgement_user",
    "ai": "memory.judgement_ai",
}

# NOTE: 纯"当前偏好"模式 (我喜欢X / 现在仍然喜欢X) 已从 fast-track 移除 —
# pipeline 现在拦截 AI 侧新增 偏好/身份 自我记忆 (自述人设漂移防护), 快速放行
# 这类句子只会白烧一次大模型抽取然后被丢弃. 长期经历 (生活) / 观点 (思维)
# 仍是合法的 AI 自我记忆, 保留 fast-track.
_STABLE_AI_SELF_MEMORY_PATTERNS = [
    re.compile(r"(承包|陪伴).{0,12}我.{0,12}(青春|童年|学生时代|小时候|成长)"),
    re.compile(r"我(以前|小时候|从前|过去|当年|青春|学生时代).{0,40}(喜欢|爱|经常|常常|总是|听|看|玩|去|吃|喝)"),
    re.compile(r"我(一直|始终|总觉得|总认为|觉得|认为).{1,40}(重要|好|不好|有意思|有感觉|值得|适合|不适合)"),
]

_AI_ACK_VERBS_RE = re.compile(
    r"(我.{0,4}(记住|记下|知道|明白|了解)了|我不会忘|我会记得)"
)
_USER_FACT_ACK_TERMS = (
    "用户", "你", "你的", "你叫", "名字", "姓名", "这个名字",
    "下次", "以后", "提醒你", "帮你",
)


def _has_stable_ai_self_memory(message: str) -> bool:
    """Detect AI-side stable preferences, long-running experiences, or views."""
    return any(pattern.search(message) for pattern in _STABLE_AI_SELF_MEMORY_PATTERNS)


def is_user_fact_acknowledgement(message: str) -> bool:
    """Detect assistant acknowledgements of user facts, not AI self-memory.

    Examples: "我记住了用户的名字叫馒头", "这个名字很可爱，我记住了",
    "我记下了，下次去试试". These are conversational commitments about the
    user or future action, not stable facts about the AI.
    """
    if not _AI_ACK_VERBS_RE.search(message or ""):
        return False
    return any(term in message for term in _USER_FACT_ACK_TERMS)


async def should_memorize(message: str, side: Side = "user") -> bool:
    """Return True if the message is worth extracting memories from.

    Args:
        message: The text to judge.
        side: "user" → spec §2.1.2；"ai" → spec §2.2.2.

    Uses the smallest available model for speed. Expected latency: <500ms.
    On LLM failure we fail open (return True) so the big model decides.
    """
    if side == "ai":
        if is_user_fact_acknowledgement(message):
            return False
        if _has_stable_ai_self_memory(message):
            return True

    try:
        template = await get_prompt_text(_PROMPT_KEY_BY_SIDE[side])
        prompt = template.format_map(SafeDict({"message": message}))
        raw = await invoke_text(get_utility_model(), prompt)
        decision = (raw or "").strip()
        # Spec output: plain "记" or "不记". "不记" must come first in check
        # because it contains "记".
        if "不记" in decision:
            return False
        if "记" in decision:
            return True
        # Unrecognized output → fail open
        logger.debug(f"Memory pre-filter unrecognized output ({side}): {decision[:40]!r}")
        return True
    except Exception as e:
        logger.warning(f"Memory pre-filter LLM failed ({side}): {e}")
        return True
