"""Shared helpers for prompt template rendering.

- `SafeDict`: format_map 兜底，未知占位符原样保留或返回 "(无)"。
- `pad_params`: [DEPRECATED Phase 2.3] 历史: 把 PAD 情绪 dict 转成 prompt 模板参数.
  当前所有 prompt 模板已删除 raw PAD 占位符 — LLM 看不懂抽象数值是 token 浪费.
  保留 helper 仅供兼容外部调用 / 未来 step 2 自然语言版可能复用; 但生产路径
  已无 caller. 详见 prompt_builder._build_emotion_section 的 Phase 2.3 注释.
- `render_prompt`: 取模板 → format_map → 调用 LLM → 裁剪，单一入口。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable

from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)

# 句末标点 — 中英都要 (。?!？！). 留空格/换行允许吃掉句末空白.
_SENTENCE_END_PAT = re.compile(r"[。?!？！]+[\s]*")


def _truncate_at_sentence_boundary(text: str, max_len: int) -> str:
    """裁到 max_len 内最后一个句末符号. 找不到合理边界 → 退回硬切但优先在
    标点处. 防 raw[:max_len] 切到中文字符中段 (生产 bug 复现 2026-05-03
    trace 019dec46: schedule_query_reply max_chars=120 切到 '明天是你生|日'
    正中字).

    跟 chat/orchestrator.truncate_at_sentence 同语义但独立实现 — utils 不应
    依赖 chat 层. 句末符更宽松 (中英 ?! 都算) 因为短回复多用问号结尾.
    """
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_match = None
    for m in _SENTENCE_END_PAT.finditer(truncated):
        last_match = m
    # 找到的句末必须在后半段 (>max_len/2), 否则只切几个字毫无意义, 干脆硬切
    if last_match and last_match.end() > max_len // 2:
        return truncated[:last_match.end()].rstrip()
    return truncated


# 所有 prompt 里 {recent_context} 为空时的统一文本.
# 全角括号: 跟 format_recent_context 的输出对齐, 让 LLM 始终看到同一 token.
EMPTY_RECENT_CONTEXT = "（无）"


class SafeDict(dict):
    """format_map 兜底：未填充占位符返回 "(无)"。"""

    def __missing__(self, key: str) -> str:
        return "(无)"


def pad_params(emotion: dict[str, Any] | None) -> dict[str, str]:
    """把 PAD 情绪 dict 转成模板参数。"""
    e = emotion or {}
    return {
        "pleasure": f"{float(e.get('pleasure', 0.0)):.2f}",
        "arousal": f"{float(e.get('arousal', 0.0)):.2f}",
        "dominance": f"{float(e.get('dominance', 0.5)):.2f}",
    }


async def render_prompt(
    prompt_key: str,
    params: dict[str, Any],
    invoke_fn: Callable[[str], Awaitable[Any]],
    *,
    max_chars: int | None = None,
    strip_split: bool = True,
) -> Any:
    """取 prompt → format_map → 调 invoke_fn。

    - invoke_fn: `invoke_text` 返回 str，`invoke_json` 返回 dict/list。
    - max_chars / strip_split 仅对字符串结果生效：按 "||" 只取首段并裁剪。
    失败或空结果返回 None（或 {} 视 invoke_fn 而定，由调用方判断）。
    """
    try:
        tmpl = await get_prompt_text(prompt_key)
        prompt = tmpl.format_map(SafeDict(params))
        raw = await invoke_fn(prompt)
        if isinstance(raw, str):
            if strip_split:
                raw = raw.strip().split("||")[0]
            if max_chars and len(raw) > max_chars:
                return _truncate_at_sentence_boundary(raw, max_chars)
            return raw
        return raw
    except Exception as e:
        logger.warning(f"render_prompt failed ({prompt_key}): {e}")
        return None
