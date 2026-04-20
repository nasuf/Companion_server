"""Shared helpers for prompt template rendering.

- `SafeDict`: format_map 兜底，未知占位符原样保留或返回 "(无)"。
- `pad_params`: 把 PAD 情绪 dict 按统一精度转成字符串参数。
- `render_prompt`: 取模板 → format_map → 调用 LLM → 裁剪，单一入口。
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)


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
            return raw[:max_chars] if max_chars else raw
        return raw
    except Exception as e:
        logger.warning(f"render_prompt failed ({prompt_key}): {e}")
        return None
