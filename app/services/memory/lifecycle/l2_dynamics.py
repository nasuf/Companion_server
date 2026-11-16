"""Compatibility API for bounded cumulative maintenance.

Legacy pure factor functions remain for historical offline comparisons. The
production entrypoints delegate exclusively to lazy_update's pure decay.
"""

from __future__ import annotations


def _time_factor(days_since_access: int) -> float:
    if days_since_access < 30:
        return 1.0
    if days_since_access < 90:
        return 0.9
    if days_since_access < 180:
        return 0.8
    if days_since_access < 365:
        return 0.7
    if days_since_access < 730:
        return 0.6
    return 0.5


def _frequency_factor(mentions_1y: int) -> float:
    if mentions_1y <= 2:
        return 1.0
    if mentions_1y <= 5:
        return 1.1
    if mentions_1y <= 10:
        return 1.2
    return 1.3


def _quality_factor(corrections_1y: int, evidence_links: int) -> float:
    """Bounded quality modifier for P1 memory governance.

    Confirmed/user corrections reduce confidence in a memory's current form;
    source evidence links give a small stability boost. The factor stays close
    to 1.0 so spec time/frequency dynamics remain the dominant signal.
    """
    penalty = min(0.30, max(0, corrections_1y) * 0.10)
    boost = min(0.10, max(0, evidence_links) * 0.03)
    return round(max(0.70, min(1.10, 1.0 - penalty + boost)), 3)


async def _adjust_side(side: str, user_id: str | None) -> dict:
    """Compatibility entry; maintenance uses the same reward-free lifecycle."""
    from app.services.memory.lifecycle.lazy_update import sweep_stale_values

    result = await sweep_stale_values(user_id=user_id, sources=(side,))
    return result[side]


async def run_l2_adjustment(user_id: str | None = None) -> dict:
    """Bounded pure decay; never overwrite cumulative scores from importance."""
    from app.services.memory.lifecycle.lazy_update import sweep_stale_values

    result = await sweep_stale_values(user_id=user_id)
    return {
        "user": result["user"], "ai": result["ai"],
        "total": result["scanned"], "promoted": result["promoted"],
        "demoted": result["demoted"], "adjusted": result["adjusted"],
        "engine": "cumulative_decay_v2", "batch_limit_per_side": 1000,
    }
