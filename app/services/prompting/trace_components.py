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


class _RenderTraceLog(list):
    """本请求的渲染记录; served 登记本请求里各 key 实际取到的修订号 (O01.01)。

    None = 本请求里这个 key 回落过默认文案或取到过两个不同修订号, 不再写修订号。
    """

    def __init__(self):
        super().__init__()
        self.served: dict[str, int | None] = {}


def start_prompt_render_trace() -> Token[list[dict[str, Any]] | None]:
    return _prompt_render_traces.set(_RenderTraceLog())


def note_served_prompt(key: str, revision: int | None) -> None:
    """get_prompt_text 成功返回快照/数据库里的内容时登记修订号; 拿不准传 None。"""
    served = getattr(_prompt_render_traces.get(), "served", None)
    if served is None or not key:
        return
    if not isinstance(revision, int) or (key in served and served[key] != revision):
        served[key] = None
    else:
        served.setdefault(key, revision)


def note_prompt_fallback(key: str) -> None:
    """回落到代码默认文案: 本请求里这个 key 的修订号都不可信。"""
    served = getattr(_prompt_render_traces.get(), "served", None)
    if served is not None and key:
        served[key] = None


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
    # O01.01: 每段用的是哪个修订号, 让回复能对到提示词版本历史; 查不到就不写 (= 未记录).
    # 组件可能复制自更早的记录, 查不到时要删掉带过来的旧值
    for item in (payload, *payload.get("components", ())):
        revision = _served_revision(item.get("prompt_key"))
        if revision is None:
            item.pop("prompt_revision", None)
        else:
            item["prompt_revision"] = revision
    traces.append(payload)


def _served_revision(key: str | None) -> int | None:
    served = getattr(_prompt_render_traces.get(), "served", None)
    return served.get(key) if served and key else None


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

    @classmethod
    def compose(cls, parts: list[tuple[str, str]], prompt_key: str):
        obj = cls("\n\n".join(text for _, text in parts), prompt_key)
        obj.parts = tuple(parts)
        return obj

    def render_components(self, render_fn) -> str:
        rendered_parts, components = [], []
        offset = 0
        for key, template in self.parts:
            text = render_fn(ManagedPromptText(template, key, self.prompt_variant))
            rendered_parts.append(text)
            components.append({"prompt_key": key, "start": offset,
                               "end": offset + len(text), "editable": True})
            offset += len(text) + 2
        rendered = "\n\n".join(rendered_parts)
        record_prompt_render(rendered, prompt_key=self.prompt_key,
                             prompt_variant=self.prompt_variant, components=components)
        return rendered

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
        if getattr(self, "parts", None):
            return self.render_components(lambda part: part._render_safely(
                lambda: str(part).format(*args, **kwargs), kwargs))
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
        if getattr(self, "parts", None):
            return self.render_components(lambda part: part._render_safely(
                lambda: str(part).format_map(mapping),
                dict(mapping) if isinstance(mapping, dict) else {}))
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
