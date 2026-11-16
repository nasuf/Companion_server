"""L2 dynamic score must never compound into `importance`.

Legacy formula tests must not require rewards during pure decay.
The old implementation wrote current_score back into importance, so the next
nightly run used last night's product as the base: frequently-accessed rows
inflated ×ff nightly, idle rows decayed ×tf nightly. Now `importance` is the
immutable initial score and the computed score lands in `current_score`.
Singleton L1 promotion is enforced by lazy_update (test_memory_lifecycle_postgres).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["user", "ai"])
async def test_maintenance_compatibility_entry_delegates_to_pure_decay(side):
    from app.services.memory.lifecycle.l2_dynamics import _adjust_side
    stats = {"scanned": 3, "adjusted": 2, "promoted": 0, "demoted": 1}
    with patch("app.services.memory.lifecycle.lazy_update.sweep_stale_values", AsyncMock(return_value={side: stats})) as sweep:
        assert await _adjust_side(side, "user-1") == stats
    sweep.assert_awaited_once_with(user_id="user-1", sources=(side,))


def test_decay_never_rewards_idle_memory_or_exceeds_initial_value():
    from app.services.memory.lifecycle.value import decayed_value
    assert decayed_value(0.6, 0) == pytest.approx(0.6)
    assert 0 < decayed_value(0.6, 180) < 0.6
    assert decayed_value(0.6, 240) == pytest.approx(0.3)
    # Real score/importance immutability, concurrency and clocks are also
    # asserted by test_memory_lifecycle_postgres.py on migrated PostgreSQL.


