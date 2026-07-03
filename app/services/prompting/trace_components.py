"""Runtime prompt render tracing for Trace modal editing.

This module records prompt template provenance while a chat/proactive request is
building LLM inputs. It intentionally stores only hashes and component spans,
not full prompt text; the final prompt is already visible in LangSmith, while
message metadata should stay compact.
"""

from __future__ import annotations

import hashlib
import logging
from contextvars import ContextVar, Token
from typing import Any

logger = logging.getLogger(__name__)

_prompt_render_traces: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "prompt_render_traces",
    default=None,
)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def start_prompt_render_trace() -> Token[list[dict[str, Any]] | None]:
    return _prompt_render_traces.set([])


def reset_prompt_render_trace(token: Token[list[dict[str, Any]] | None]) -> None:
    _prompt_render_traces.reset(token)


def snapshot_prompt_render_traces() -> list[dict[str, Any]]:
    traces = _prompt_render_traces.get()
    if not traces:
        return []
    return [dict(item) for item in traces]


def record_prompt_render(
    rendered_prompt: str,
    *,
    prompt_key: str | None = None,
    prompt_variant: str | None = None,
    components: list[dict[str, Any]] | None = None,
    source: str = "managed",
) -> None:
    traces = _prompt_render_traces.get()
    if traces is None or not rendered_prompt:
        return
    payload: dict[str, Any] = {
        "prompt_hash": prompt_hash(rendered_prompt),
        "source": source,
    }
    if prompt_key:
        payload["prompt_key"] = prompt_key
    if prompt_variant:
        payload["prompt_variant"] = prompt_variant
    if components:
        payload["components"] = [dict(item) for item in components]
    elif prompt_key:
        payload["components"] = [{
            "prompt_key": prompt_key,
            "start": 0,
            "end": len(rendered_prompt),
            "editable": True,
        }]
    traces.append(payload)


class _SafeRenderDict(dict):
    """format_map 兜底：未知占位符渲染为 "(无)"（与 prompting.utils.SafeDict
    同语义；定义在此处避免 utils ↔ trace_components 循环 import）。"""

    def __missing__(self, key: str) -> str:
        return "(无)"


class ManagedPromptText(str):
    """str subclass that records direct `.format*()` prompt render calls.

    W5 防炸渲染：admin 可在后台编辑这些模板，两类人为失误不该炸掉调用
    链路——误删/写错占位符（KeyError/IndexError → SafeDict 补 "(无)" 重试）、
    写入字面大括号如 JSON 示例（ValueError 无效 format spec → 返回未渲染
    原文并告警）。在 str 子类层修一次，所有 get_prompt_text().format(...)
    调用站点（15+ 文件）自动获得防护，无需逐站点迁移。
    """

    def __new__(cls, value: str, prompt_key: str, prompt_variant: str = "active"):
        obj = str.__new__(cls, value)
        obj.prompt_key = prompt_key
        obj.prompt_variant = prompt_variant
        return obj

    def _render_safely(self, render_fn, fallback_params: dict) -> str:
        try:
            return render_fn()
        except (KeyError, IndexError) as e:
            logger.warning(
                f"prompt '{self.prompt_key}' strict render failed "
                f"({type(e).__name__}: {e}); retrying with SafeDict fallback",
            )
            try:
                return str(self).format_map(_SafeRenderDict(fallback_params))
            except (ValueError, KeyError, IndexError):
                return str(self)
        except ValueError as e:
            # 模板含字面大括号 (如 JSON 示例没写成 {{...}}) — 返回原文,
            # LLM 看到 {placeholder} 原文通常也能理解任务, 好过链路炸掉.
            logger.warning(
                f"prompt '{self.prompt_key}' has invalid format spec ({e}); "
                f"returning unrendered template. 检查模板字面大括号是否写成 {{{{...}}}}",
            )
            return str(self)

    def format(self, *args: Any, **kwargs: Any) -> str:  # type: ignore[override]
        rendered = self._render_safely(
            lambda: str(self).format(*args, **kwargs), kwargs,
        )
        record_prompt_render(
            rendered,
            prompt_key=self.prompt_key,
            prompt_variant=self.prompt_variant,
        )
        return rendered

    def format_map(self, mapping: Any) -> str:  # type: ignore[override]
        rendered = self._render_safely(
            lambda: str(self).format_map(mapping),
            dict(mapping) if isinstance(mapping, dict) else {},
        )
        record_prompt_render(
            rendered,
            prompt_key=self.prompt_key,
            prompt_variant=self.prompt_variant,
        )
        return rendered
