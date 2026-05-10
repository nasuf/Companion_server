"""Shared helpers for prompt template rendering.

- `SafeDict`: format_map 兜底，未知占位符原样保留或返回 "(无)"。
- `render_prompt`: 取模板 → format_map → 调用 LLM → 裁剪，单一入口。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable, Iterable

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

_FIELD_PAT = re.compile(r"(?<!{){([A-Za-z_][A-Za-z0-9_]*)(?:![^}:]+)?(?::[^}]+)?}(?!})")
_EMPTY_REFERENCE_VALUES = {
    "",
    "。",
    "．",
    ".",
    "(无)",
    "（无）",
    "(未知)",
    "（未知）",
    "(暂无)",
    "（暂无）",
    "无",
    "未知",
    "暂无",
}


def _is_empty_reference_value(value: Any) -> bool:
    """Return True for admin/runtime placeholders that should not occupy prompt rows."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in _EMPTY_REFERENCE_VALUES
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _remove_empty_reference_headers(lines: list[str]) -> list[str]:
    """Drop orphan reference headers left after all optional rows were removed.

    This is intentionally narrow: only `【参考信息...】` blocks are removed, and
    only when the next non-empty line is another bracket section or EOF.
    """
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("【参考信息"):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines) or lines[j].lstrip().startswith("【"):
                i = j
                continue
        result.append(line)
        i += 1
    return result


def compact_optional_reference_rows(
    template: str,
    params: dict[str, Any],
    optional_keys: Iterable[str] | None = None,
) -> str:
    """Remove optional reference rows whose values are empty placeholders.

    The caller decides which placeholders are optional. Required fields such as
    user message, output schema, reminder content, or schedule state are never
    removed unless explicitly listed.
    """
    optional = set(optional_keys or ())
    if not optional:
        return template

    kept: list[str] = []
    for line in template.splitlines():
        fields = _FIELD_PAT.findall(line)
        optional_fields = [name for name in fields if name in optional]
        if (
            optional_fields
            and len(optional_fields) == len(fields)
            and all(_is_empty_reference_value(params.get(name)) for name in optional_fields)
        ):
            continue
        kept.append(line)

    return "\n".join(_remove_empty_reference_headers(kept))


def render_template(
    template: str,
    params: dict[str, Any],
    *,
    optional_keys: Iterable[str] | None = None,
    safe: bool = True,
) -> str:
    """Format a prompt template after dropping optional empty reference rows."""
    compacted = compact_optional_reference_rows(template, params, optional_keys)
    if safe:
        return compacted.format_map(SafeDict(params))
    return compacted.format(**params)


class SafeDict(dict):
    """format_map 兜底：未填充占位符返回 "(无)"。"""

    def __missing__(self, key: str) -> str:
        return "(无)"


async def render_prompt(
    prompt_key: str,
    params: dict[str, Any],
    invoke_fn: Callable[[str], Awaitable[Any]],
    *,
    max_chars: int | None = None,
    strip_split: bool = True,
    optional_keys: Iterable[str] | None = None,
) -> Any:
    """取 prompt → format_map → 调 invoke_fn。

    - invoke_fn: `invoke_text` 返回 str，`invoke_json` 返回 dict/list。
    - max_chars / strip_split 仅对字符串结果生效：按 "||" 只取首段并裁剪。
    失败或空结果返回 None（或 {} 视 invoke_fn 而定，由调用方判断）。
    """
    try:
        tmpl = await get_prompt_text(prompt_key)
        prompt = render_template(tmpl, params, optional_keys=optional_keys)
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
