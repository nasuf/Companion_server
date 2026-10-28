"""One registered hot-path dialogue policy, resolved from the bound turn snapshot."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any

from app.services.prompts.system_prompts import CHAT_HISTORY_TOKEN_BUDGET

DEFAULT_ROUNDS = 5
MIN_ROUNDS, MAX_ROUNDS = 1, 20
# Main reply history is a token budget filled with whole messages, newest first
# (prompt_builder.build_chat_messages); the estimate is the context selector's.
DEFAULT_MAIN_HISTORY_TOKENS = CHAT_HISTORY_TOKEN_BUDGET
MIN_MAIN_HISTORY_TOKENS, MAX_MAIN_HISTORY_TOKENS = 500, 8000
# Search requests already reached ~30.6k input tokens at the default budget and
# the provider bills above 32k at a higher tier, so search turns never grow.
SEARCH_MAIN_HISTORY_TOKENS = DEFAULT_MAIN_HISTORY_TOKENS
# Production averaged ~22 estimated tokens per stored message (200 rows ≈ 4400),
# a thin margin once bubbles coalesce. The default keeps its existing 200 rows;
# a raised budget fetches at 10 tokens per row so the budget, not the row limit,
# decides the history. Trace reports when the row limit was the actual bound.
HISTORY_FETCH_FLOOR = 200
_TOKENS_PER_FETCHED_ROW = 10
LINE_CHARS = 800
# Judgement prompts run on almost every turn: increasing N must not let their
# character budget grow to the reply templates' 6000-character ceiling.
JUDGEMENT_LINE_CHARS = 150
JUDGEMENT_MAX_CHARS = 1500
FORMAT_VERSION = 'dialogue-turns-v1'
PARAMS = ({'key': 'rounds', 'label': '对话上下文轮数', 'default': DEFAULT_ROUNDS,
           'minimum': MIN_ROUNDS, 'maximum': MAX_ROUNDS, 'unit': '轮',
           'description': '用户连续发言与随后 AI 多个气泡算一轮；用于聊天热路径模板。'},
          {'key': 'main_history_tokens', 'label': '主回复历史长度', 'default': DEFAULT_MAIN_HISTORY_TOKENS,
           'minimum': MIN_MAIN_HISTORY_TOKENS, 'maximum': MAX_MAIN_HISTORY_TOKENS, 'unit': '估算 token',
           'description': '主回复从最新一条往前整条放入历史，放满为止；隔了 3 小时以上的旧对话不放原文。'
                          f'联网搜索回合最多 {SEARCH_MAIN_HISTORY_TOKENS} token，避免超出 32k 输入计价档。'})
_LIMITS = {param['key']: (param['minimum'], param['maximum'], param['default'], param['label'])
           for param in PARAMS}
EXCLUSIONS = (
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
    if not values or set(values) - set(_LIMITS):
        raise ValueError('仅可修改对话轮数 rounds 与主回复历史长度 main_history_tokens')
    result = stored(raw)
    for key, value in values.items():
        low, high, default, label = _LIMITS[key]
        if value is None or type(value) is int and value == default:
            result.pop(key, None)
        elif type(value) is not int or not low <= value <= high:
            raise ValueError(f'{label}必须是 {low}–{high} 的整数')
        else:
            result[key] = value
    return result


def _resolved(data: dict, key: str) -> tuple[int, bool]:
    low, high, default, _ = _LIMITS[key]
    value = data.get(key, default)
    valid = type(value) is int and low <= value <= high
    return (value if valid else default), valid


def main_history_budget(policy: 'Policy', *, web_search: bool) -> int:
    return min(policy.main_history_tokens, SEARCH_MAIN_HISTORY_TOKENS) if web_search else policy.main_history_tokens


def history_fetch_limit(main_history_tokens: int) -> int:
    if main_history_tokens <= DEFAULT_MAIN_HISTORY_TOKENS:
        return HISTORY_FETCH_FLOOR
    return max(HISTORY_FETCH_FLOOR, math.ceil(main_history_tokens / _TOKENS_PER_FETCHED_ROW))


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
    main_history_tokens: int = DEFAULT_MAIN_HISTORY_TOKENS

    @property
    def max_chars(self) -> int:
        return reply_max_chars(self.rounds)

    def public(self) -> dict:
        return {'effective_rounds': self.rounds, 'revision': self.revision,
                'main_history_mode': 'budget', 'main_history_token_budget': self.main_history_tokens,
                'format_version': FORMAT_VERSION, 'stored_policy_valid': self.valid}


def resolve(raw: Any) -> Policy:
    data = stored(raw)
    rounds, rounds_valid = _resolved(data, 'rounds')
    tokens, tokens_valid = _resolved(data, 'main_history_tokens')
    return Policy(rounds, revision(raw), rounds_valid and tokens_valid, tokens)


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
    if target is not None and name in {'checks', 'intent', 'relevance', 'crisis', 'tier', 'continuation', 'main'}:
        target['windows'][name] = metrics
