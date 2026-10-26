"""Render registered dialogue replies with one explicit, final current message.

Registry contents are never rewritten. The wrapper returned by the prompt store
substitutes reference placeholders and appends literal dialogue after all task
and temporal components. Main native messages and judgement prompts opt out.
"""
from __future__ import annotations

from string import Formatter

from app.services.prompting.trace_components import ManagedPromptText, record_prompt_render

HISTORY_KEY = "chat.dialogue_history_header"
CURRENT_KEY = "chat.dialogue_current_message_header"
CONTINUATION_KEY = "chat.dialogue_continuation_hint"
# The JSON-producing adjustment reply is explicitly approved as an exception;
# its original output schema stays in the task component and is tested separately.
REPLY_KEYS = frozenset({
    "memory.weak_reply", "memory.medium_reply", "memory.strong_reply", "memory.l3_reply",
    "intent.current_state_reply", "intent.schedule_query_reply",
    "intent.schedule_adjust_reply", "intent.end_reply",
})
# The closing hint ("接着……往下说") goes only where the paired replay measured it:
# the tier replies and the current-state reply. The goodbye reply must not
# extend the topic, and the adjustment reply must end on its JSON schema.
CONTINUATION_HINT_KEYS = frozenset({
    "memory.weak_reply", "memory.medium_reply", "memory.strong_reply", "memory.l3_reply",
    "intent.current_state_reply",
})
_EMPTY = frozenset({"", "(无)", "（无）", "无"})


def supports_layout(template: str) -> bool:
    """Only exact named placeholders qualify; malformed/custom forms stay intact."""
    try:
        fields = {name for _, name, _, _ in Formatter().parse(template) if name}
    except ValueError:
        return False
    return {"context", "message"} <= fields


class DialoguePromptText(ManagedPromptText):
    def __new__(cls, template: ManagedPromptText, *, history: str, current: str, hint: str = ""):
        obj = super().__new__(cls, str(template), template.prompt_key, template.prompt_variant)
        obj.parts = getattr(template, "parts", ((template.prompt_key, str(template)),))
        obj.history_header, obj.current_header, obj.continuation_hint = history, current, hint
        return obj

    def render_dialogue(self, params, *, optional_keys=None, safe=True, context_parts=()):
        from app.services.prompting.utils import render_template
        from app.services.chat.context_policy import observation
        from app.services.chat.dialogue_context import _timestamp

        history = str(params.get("context") or "")
        message = str(params.get("message") or "")
        # No last-message guessing in admin previews or malformed callers.
        if not message.strip():
            return render_template(ManagedPromptText.compose(list(self.parts), self.prompt_key),
                                   params, optional_keys=optional_keys, safe=safe,
                                   context_parts=context_parts)
        has_history = history.strip() not in _EMPTY
        references = {**params, "message": "见文末【用户这次发来的消息】",
                      "context": "见文末【最近对话】" if has_history else "（无）"}
        # Preserve the explicit empty-history placeholder instead of dropping it
        # via optional-reference row compaction.
        optional_keys = set(optional_keys or ()) - {"context", "message"}
        parts = []
        for key, template in self.parts:
            text = render_template(ManagedPromptText(template, key), references,
                                   optional_keys=optional_keys, safe=safe)
            parts.append((key, text, True))
        parts.extend((key, str(text), True) for key, text in context_parts if text)
        if has_history:
            parts.extend([(HISTORY_KEY, self.history_header, True), (None, history, False)])
        clock = params.get("received_at") or (observation() or {}).get("message_received_at")
        parts.extend([(CURRENT_KEY, self.current_header, True),
                      (None, _timestamp(clock) + "用户: " + message, False)])
        if self.continuation_hint:
            parts.append((CONTINUATION_KEY, self.continuation_hint, True))
        rendered, components, offset = [], [], 0
        for key, text, editable in parts:
            rendered.append(text)
            component = {"start": offset, "end": offset + len(text), "editable": editable}
            if key:
                component["prompt_key"] = key
            components.append(component)
            offset += len(text) + 2
        prompt = "\n\n".join(rendered)
        record_prompt_render(prompt, prompt_key=self.prompt_key,
                             prompt_variant=self.prompt_variant, components=components)
        return prompt

    def format(self, *args, **kwargs):
        if args:
            return super().format(*args, **kwargs)
        return self.render_dialogue(kwargs)

    def format_map(self, mapping):
        return self.render_dialogue(dict(mapping))


async def attach_layout(template: ManagedPromptText, get_prompt):
    if template.prompt_key not in REPLY_KEYS or not supports_layout(str(template)):
        return template
    from app.services.prompting.store import PromptDisabledError
    try:
        history, current = await get_prompt(HISTORY_KEY), await get_prompt(CURRENT_KEY)
    except PromptDisabledError:
        # Disabling either structural header restores the original placement;
        # never replace the slots while hiding their only actual values.
        return template
    if not str(history).strip() or not str(current).strip():
        return template
    hint = ""
    if template.prompt_key in CONTINUATION_HINT_KEYS:
        try:
            hint = await get_prompt(CONTINUATION_KEY)
        except PromptDisabledError:
            hint = ""
    return DialoguePromptText(template, history=str(history), current=str(current), hint=str(hint))
