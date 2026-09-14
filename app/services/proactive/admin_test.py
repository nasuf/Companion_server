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
    # 卡片被 skip 时具体原因 (2026-09-14 task#12 可观测性):
    #   no_preselected_no_force / gate_rejected / preselected_no_url /
    #   candidate_search_empty / metadata_unusable_{status} / exception_{exc}
    # link_card_used=True 时保持 None (成功挂卡, 无 skip_reason).
    link_card_skip_reason: str | None = None
    extra: dict = field(default_factory=dict)
