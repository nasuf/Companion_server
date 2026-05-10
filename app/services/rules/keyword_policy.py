"""Shared primitives for deterministic keyword rules.

The agent stack intentionally uses keywords for different purposes: safety
gates, low-latency fast paths, LLM-failure fallback, write-action guards, and
retrieval boosts. A common metadata shape makes those roles explicit instead
of leaving anonymous tuples scattered through business modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class RulePurpose(str, Enum):
    SAFETY_GATE = "safety_gate"
    FAST_PATH = "fast_path"
    LLM_FALLBACK = "llm_fallback"
    WRITE_ACTION = "write_action"
    CONFIRMATION = "confirmation"
    RETRIEVAL_GATE = "retrieval_gate"
    RETRIEVAL_BOOST = "retrieval_boost"
    NORMALIZATION = "normalization"
    STYLE_HINT = "style_hint"


@dataclass(frozen=True)
class KeywordRuleSet:
    """A named, auditable keyword set with a declared control-flow purpose."""

    name: str
    purpose: RulePurpose
    terms: tuple[str, ...]
    description: str = ""

    def contains_any(self, text: str) -> bool:
        return contains_any(text, self.terms)

    def strip_all(self, text: str) -> str:
        cleaned = text or ""
        for term in self.terms:
            cleaned = cleaned.replace(term, "")
        return cleaned


def contains_any(text: str, terms: Iterable[str]) -> bool:
    if not text:
        return False
    return any(term in text for term in terms)


_COMPACT_CHAT_RE = re.compile(r"[\s，。！？!?~～…,.、]+")


def compact_chat_text(text: str) -> str:
    """Normalize chat text for exact phrase gates without changing semantics."""

    return _COMPACT_CHAT_RE.sub("", (text or "").strip())
