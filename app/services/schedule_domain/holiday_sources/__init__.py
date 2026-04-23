"""External holiday data sources used by admin preview + refresh cron.

See `merger.collect_candidates` for the unified entry point.
"""

from app.services.schedule_domain.holiday_sources.merger import (
    SourceStatus,
    collect_candidates,
)

__all__ = ["SourceStatus", "collect_candidates"]
