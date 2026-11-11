"""Admin-tunable switches and thresholds for the reply repetition check.

Stored as overrides in system_config.reply_repetition_policy (only values that
differ from the defaults) and resolved from the turn's bound config snapshot,
like chat/context_policy. Defaults are the behaviour shipped before the policy
existed, so an empty row changes nothing. The prompt sections themselves are
switched in prompt management, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

PARAMS = (
    {'key': 'rewrite_enabled', 'label': '复述时重写一次', 'kind': 'bool', 'default': True,
     'description': '对方只是附和、或上一轮回复晚于这条消息时，生成的每一条都是上一轮的复述，就带着提示重写一次再发。'},
    {'key': 'rewrite_overlap', 'label': '重写判定重合度', 'kind': 'ratio', 'default': 0.6,
     'minimum': 0.4, 'maximum': 0.95, 'unit': '相邻两字重合比例',
     'description': '整段回复和上一轮的重合度达到这个比例就算复述。调低会多重写（多花钱、多 1–2 秒）。'},
    {'key': 'backstop_enabled', 'label': '删除照抄的单条气泡', 'kind': 'bool', 'default': True,
     'description': '发出前去掉和上一轮基本是同一句话的气泡；不会删光，对方在问或让你再说一遍时不删。'},
    {'key': 'backstop_overlap', 'label': '附和/交叠轮次的删除界值', 'kind': 'ratio', 'default': 0.6,
     'minimum': 0.4, 'maximum': 0.95, 'unit': '相邻两字重合比例',
     'description': '对方只是附和、或上一轮回复晚于这条消息时，单条气泡重合度到这个比例就删。'},
    {'key': 'backstop_verbatim_overlap', 'label': '普通轮次的删除界值', 'kind': 'ratio', 'default': 0.8,
     'minimum': 0.6, 'maximum': 1.0, 'unit': '相邻两字重合比例',
     'description': '对方说了新内容时，只删几乎原样照抄的气泡。'},
    {'key': 'semantic_enabled', 'label': '语义复读检查', 'kind': 'bool', 'default': False,
     'description': '用本地向量比对这一轮和前三轮 AI 的气泡，意思几乎相同（换了说法）的那条发出前删掉。'
                    '保护规则同上：不删光、对方在问或让你再说一遍时不删。'},
    {'key': 'semantic_similarity', 'label': '语义复读判定界值', 'kind': 'ratio', 'default': 0.9,
     'minimum': 0.8, 'maximum': 0.99, 'unit': '向量余弦相似度',
     'description': '2026-11-04~10 生产 938 轮校准：0.90 时判对 91%、覆盖 15% 的复读、触发 3.4% 轮次；'
                    '0.85 时判对 83%、覆盖 25%。调低会多删，也会多误删。'},
    {'key': 'semantic_rollout_pct', 'label': '语义复读检查灰度比例', 'kind': 'int', 'default': 0,
     'minimum': 0, 'maximum': 100, 'unit': '% 用户',
     'description': '打开开关后，按用户固定分桶只对这部分用户生效；0 表示谁都不生效。'},
    {'key': 'breaker_rate', 'label': '重写熔断比例', 'kind': 'ratio', 'default': 0.05,
     'minimum': 0.01, 'maximum': 0.5, 'unit': '最近 1 小时重写轮次 / 主回复轮次',
     'description': '超过这个比例就自动停止重写（提示段和删除照旧），并发一次告警。'},
    {'key': 'breaker_min_turns', 'label': '熔断最少样本', 'kind': 'int', 'default': 40,
     'minimum': 10, 'maximum': 1000, 'unit': '主回复轮次',
     'description': '最近 1 小时主回复不到这么多轮时不判熔断，避免夜里两三轮就触发。'},
)
_BY_KEY = {param['key']: param for param in PARAMS}


def stored(raw: Any) -> dict:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


def revision(raw: Any) -> str:
    return sha256(json.dumps(stored(raw), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _valid(param: dict, value: Any) -> bool:
    if param['kind'] == 'bool':
        return type(value) is bool
    if param['kind'] == 'int':
        return type(value) is int and param['minimum'] <= value <= param['maximum']
    return (type(value) in (int, float) and not isinstance(value, bool)
            and param['minimum'] <= value <= param['maximum'])


def _error(param: dict) -> str:
    if param['kind'] == 'bool':
        return f"{param['label']}必须是开或关"
    return f"{param['label']}必须在 {param['minimum']}–{param['maximum']} 之间"


def apply_patch(raw: Any, values: dict) -> dict:
    """Merge {key: value | None}; None or the default removes the override."""
    unknown = set(values or {}) - set(_BY_KEY)
    if not values or unknown:
        raise ValueError('未知或为空的配置项: ' + ', '.join(sorted(unknown)) if unknown else '没有要修改的配置项')
    result = stored(raw)
    for key, value in values.items():
        param = _BY_KEY[key]
        if value is None or value == param['default'] and type(value) is type(param['default']):
            result.pop(key, None)
        elif not _valid(param, value):
            raise ValueError(_error(param))
        else:
            result[key] = float(value) if param['kind'] == 'ratio' else value
    return result


@dataclass(frozen=True)
class Policy:
    rewrite_enabled: bool
    rewrite_overlap: float
    backstop_enabled: bool
    backstop_overlap: float
    backstop_verbatim_overlap: float
    semantic_enabled: bool
    semantic_similarity: float
    semantic_rollout_pct: int
    breaker_rate: float
    breaker_min_turns: int
    revision: str
    valid: bool = True

    def public(self) -> dict:
        return {**{p['key']: getattr(self, p['key']) for p in PARAMS},
                'revision': self.revision, 'stored_policy_valid': self.valid}


def resolve(raw: Any) -> Policy:
    """Invalid stored values fall back to their default one by one and are reported."""
    data = stored(raw)
    values, valid = {}, True
    for param in PARAMS:
        value = data.get(param['key'], param['default'])
        if not _valid(param, value):
            value, valid = param['default'], False
        values[param['key']] = value
    return Policy(**values, revision=revision(raw), valid=valid)


def current() -> Policy:
    from app.services.runtime_config import reply_repetition_policy_overrides
    return resolve(reply_repetition_policy_overrides())
