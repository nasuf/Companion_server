"""Bounded template dialogue rendering. Main native history and memory IO are separate."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from app.services.chat import context_policy
from app.services.chat.message_utils import (
    is_offering_received_notice, last_user_turns, message_field, group_dialogue_turns,
)
from app.services.memory.retrieval.context_selector import estimate_tokens


@dataclass(frozen=True)
class Dialogue:
    text: str
    user_text: str
    lines: tuple[str, ...]
    metrics: dict


def _timestamp(value) -> str:
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
        if not isinstance(dt, datetime):
            return ''
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(ZoneInfo('Asia/Shanghai')).strftime('[%m-%d %H:%M] ')
    except (ValueError, TypeError, OverflowError):
        return ''


def render_dialogue(messages, *, turns=None, line_chars=context_policy.LINE_CHARS,
                    max_chars=None, exclude_ids=None, purpose='checks') -> Dialogue:
    policy = context_policy.current()
    rounds = policy.rounds if turns is None else max(0, min(20, turns))
    limit = policy.max_chars if max_chars is None else max(0, max_chars)
    # Bound processing to the same 200-row database history ceiling.
    excluded = set(exclude_ids or ())
    rows = [m for m in (messages or [])[-200:] if message_field(m, 'role') in {'user', 'assistant'}
            and message_field(m, 'id') not in excluded
            and not is_offering_received_notice(message_field(m, 'metadata'))
            and str(message_field(m, 'content') or '').strip()]
    # Preserve CHAT01.09.2's continuous user-fragment policy. Memory uses
    # the same grouping helper with a finite gap for extraction context.
    merged = []
    for group in group_dialogue_turns(rows):
        text = "\n".join(str(message_field(row, 'content') or '') for row in group)
        merged.append({'role': message_field(group[-1], 'role'),
                       'content': ' '.join(text.split()),
                       'createdAt': message_field(group[-1], 'createdAt')})
    window = last_user_turns(merged, rounds, lambda row: row['role'])
    truncated = len(window) < len(merged) or len(messages or []) > 200
    lines = []
    for row in window:
        text = row['content']
        if len(text) > line_chars:
            truncated = True
            text = ('…' + text[-line_chars:]) if row['role'] == 'assistant' else (text[:line_chars] + '…')
        line = _timestamp(row['createdAt']) + ('AI' if row['role'] == 'assistant' else '用户') + ': ' + text
        lines.append((row['role'], line, text))
    selected = []
    used = 0
    for item in reversed(lines):
        extra = len(item[1]) + (1 if selected else 0)
        if used + extra > limit:
            truncated = True
            # Keep a bounded latest line, never exceed the advertised limit.
            if not selected and limit:
                role, line, text = item
                prefix = line[:-len(text)] if text else line
                room = max(0, limit - len(prefix) - 1)
                clipped = ('…' + text[-room:]) if role == 'assistant' and room else (text[:room] + '…')
                selected.append((role, (prefix + clipped)[:limit], clipped))
            break
        selected.append(item)
        used += extra
    selected.reverse()
    text = '\n'.join(x[1] for x in selected) or '（无）'
    metrics = {'effective_rounds': rounds, 'injected_rounds': sum(x[0] == 'user' for x in selected),
               'injected_lines': len(selected), 'injected_tokens_estimate': estimate_tokens(text) if selected else 0,
               'token_basis': 'context_selector heuristic; not provider billable tokens',
               'truncated': truncated, 'max_chars': limit, 'line_chars': line_chars,
               'timestamp_timezone': 'Asia/Shanghai', 'policy_revision': policy.revision}
    context_policy.note_window(purpose, metrics)
    return Dialogue(text, '\n'.join(x[2] for x in selected if x[0] == 'user'),
                    tuple(x[1] for x in selected), metrics)
