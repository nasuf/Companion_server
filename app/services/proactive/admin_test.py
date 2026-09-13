"""Admin-only knobs for manual proactive QA triggers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AdminProactiveTestOptions:
    use_web_search: bool = False
    use_link_card: bool = False


@dataclass
class AdminProactiveSendOutcome:
    web_search_used: bool = False
    link_card_used: bool = False
    skip_reason: str | None = None
    extra: dict = field(default_factory=dict)
