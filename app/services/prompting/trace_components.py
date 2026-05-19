"""Runtime prompt render tracing for Trace modal editing.

This module records prompt template provenance while a chat/proactive request is
building LLM inputs. It intentionally stores only hashes and component spans,
not full prompt text; the final prompt is already visible in LangSmith, while
message metadata should stay compact.
"""

from __future__ import annotations

import hashlib
from contextvars import ContextVar, Token
from typing import Any

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


class ManagedPromptText(str):
    """str subclass that records direct `.format*()` prompt render calls."""

    def __new__(cls, value: str, prompt_key: str, prompt_variant: str = "active"):
        obj = str.__new__(cls, value)
        obj.prompt_key = prompt_key
        obj.prompt_variant = prompt_variant
        return obj

    def format(self, *args: Any, **kwargs: Any) -> str:  # type: ignore[override]
        rendered = str(self).format(*args, **kwargs)
        record_prompt_render(
            rendered,
            prompt_key=self.prompt_key,
            prompt_variant=self.prompt_variant,
        )
        return rendered

    def format_map(self, mapping: Any) -> str:  # type: ignore[override]
        rendered = str(self).format_map(mapping)
        record_prompt_render(
            rendered,
            prompt_key=self.prompt_key,
            prompt_variant=self.prompt_variant,
        )
        return rendered
