from __future__ import annotations

import math
import random
from collections.abc import Callable


MIN_GIFT_AMOUNT_CENTS = 500
GIFT_LOGNORMAL_SIGMA = 1.2


def sample_gift_amount_cents(
    available_cents: int,
    *,
    normal_sample: Callable[[float, float], float] | None = None,
) -> int | None:
    """Sample a gift amount from the product spec's log-normal budget formula."""

    if available_cents < MIN_GIFT_AMOUNT_CENTS:
        return None
    sample = normal_sample or random.gauss
    median = 0.2 * available_cents
    z_value = sample(0, 1)
    amount = min(available_cents, round(median * math.exp(GIFT_LOGNORMAL_SIGMA * z_value)))
    if amount < MIN_GIFT_AMOUNT_CENTS:
        return None
    return int(amount)
