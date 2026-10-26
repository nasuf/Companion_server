"""One registered hot-path dialogue policy, resolved from the bound turn snapshot."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

DEFAULT_ROUNDS = 5
MIN_ROUNDS, MAX_ROUNDS = 1, 20
LINE_CHARS = 800
# Judgement prompts run on almost every turn: increasing N must not let their
# character budget grow to the reply templates' 6000-character ceiling.
JUDGEMENT_LINE_CHARS = 150
JUDGEMENT_MAX_CHARS = 1500
FORMAT_VERSION = 'dialogue-turns-v1'
PARAMS = ({'key': 'rounds', 'label': '对话上下文轮数', 'default': DEFAULT_ROUNDS,
           'minimum': MIN_ROUNDS, 'maximum': MAX_ROUNDS,
           'description': '用户连续发言与随后 AI 多个气泡算一轮；用于聊天热路径模板。'},)
EXCLUSIONS = (
    {'key': 'main', 'label': '主回复历史', 'value': '保持现状：4000 token 预算和缓存切点'},
    {'key': 'followup', 'label': '话题完结判定与 B 追问', 'value': '20 轮'},
    {'key': 'proactive', 'label': '主动消息', 'value': '近 48 小时及各自窗口'},
    {'key': 'reminder', 'label': '提醒预检', 'value': '近 30 分钟'},
    {'key': 'memory', 'label': '记忆抽取', 'value': '按抽取批次'},
)
_observation: ContextVar[dict | None] = ContextVar('chat_context_observation', default=None)


def stored(raw: Any) -> dict:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


def revision(raw: Any) -> str:
    return sha256(json.dumps(stored(raw), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def apply_patch(raw: Any, values: dict) -> dict:
    if not values or set(values) - {'rounds'}:
        raise ValueError('仅可修改对话轮数 rounds；主回复历史模式尚未开放')
    result = stored(raw)
    value = values['rounds']
    if value is None or type(value) is int and value == DEFAULT_ROUNDS:
        result.pop('rounds', None)
    elif type(value) is not int or not MIN_ROUNDS <= value <= MAX_ROUNDS:
        raise ValueError('对话轮数必须是 1–20 的整数')
    else:
        result['rounds'] = value
    return result


REPLY_CHARS_PER_ROUND, REPLY_MAX_CHARS = 600, 6000


def reply_max_chars(rounds: int) -> int:
    return min(REPLY_MAX_CHARS, rounds * REPLY_CHARS_PER_ROUND)


def window_caps() -> dict:
    """What each consumer may receive; the admin modal renders this, never a copy."""
    return {
        "reply_templates": {"line_chars": LINE_CHARS, "chars_per_round": REPLY_CHARS_PER_ROUND,
                            "max_chars_ceiling": REPLY_MAX_CHARS, "grows_with_rounds": True},
        "judgements": {"line_chars": JUDGEMENT_LINE_CHARS, "max_chars": JUDGEMENT_MAX_CHARS,
                       "grows_with_rounds": False},
    }


@dataclass(frozen=True)
class Policy:
    rounds: int
    revision: str
    valid: bool = True

    @property
    def max_chars(self) -> int:
        return reply_max_chars(self.rounds)

    def public(self) -> dict:
        return {'effective_rounds': self.rounds, 'revision': self.revision,
                'main_history_mode': 'budget', 'main_history_token_budget': 4000,
                'format_version': FORMAT_VERSION, 'stored_policy_valid': self.valid}


def resolve(raw: Any) -> Policy:
    data = stored(raw)
    n = data.get('rounds', DEFAULT_ROUNDS)
    valid = type(n) is int and MIN_ROUNDS <= n <= MAX_ROUNDS
    return Policy(n if valid else DEFAULT_ROUNDS, revision(raw), valid)


def current() -> Policy:
    from app.services.runtime_config import chat_context_policy_overrides
    return resolve(chat_context_policy_overrides())


def start_observation():
    return _observation.set({**current().public(), 'windows': {}})


def reset_observation(token):
    if token is not None:
        _observation.reset(token)


def observation() -> dict | None:
    return _observation.get()


def note_window(name: str, metrics: dict) -> None:
    target = observation()
    if target is not None and name in {'checks', 'intent', 'relevance', 'crisis', 'tier', 'continuation'}:
        target['windows'][name] = metrics
