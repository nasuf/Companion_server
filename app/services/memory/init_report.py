"""Per-agent L1 initialization report.

捕获每次 `generate_l1_coverage` 的 phase 耗时 / LLM token / 每大类产出 /
dedupe 条数, 存 Redis 7 天供运维查证。
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import AsyncIterator

from app.redis_client import get_redis
from app.services.memory.redis_keys import report_key

logger = logging.getLogger(__name__)

_REPORT_TTL = 604800  # 7 days


@dataclass
class MainStats:
    """Per-main LLM call stats."""
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: int = 0
    produced: int = 0
    failed: bool = False


@dataclass
class InitReport:
    agent_id: str
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    profile_id: str | None = None
    phase_ms: dict[str, int] = field(default_factory=dict)
    main_stats: dict[str, MainStats] = field(default_factory=dict)
    retry_stats: dict[str, MainStats] = field(default_factory=dict)
    direct_count: int = 0
    llm_count: int = 0
    dedupe_removed: int = 0
    total_stored: int = 0
    distinct_subs: int = 0
    conditional_included: list[str] = field(default_factory=list)
    gaps_after_phase1: int = 0
    gaps_after_llm: int = 0

    def total_duration_ms(self) -> int:
        return int(((self.ended_at or time.time()) - self.started_at) * 1000)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["main_stats"] = {k: asdict(v) for k, v in self.main_stats.items()}
        d["retry_stats"] = {k: asdict(v) for k, v in self.retry_stats.items()}
        d["total_duration_ms"] = self.total_duration_ms()
        return d


@contextlib.asynccontextmanager
async def init_report(agent_id: str, profile_id: str | None = None) -> AsyncIterator[InitReport]:
    """Collect a report while generation runs; persist to Redis on exit.

    On exception the report is still written (partial data aids debugging)
    before re-raising.
    """
    report = InitReport(agent_id=agent_id, profile_id=profile_id)
    try:
        yield report
    finally:
        report.ended_at = time.time()
        try:
            redis = await get_redis()
            await redis.set(
                report_key(agent_id),
                json.dumps(report.to_dict(), ensure_ascii=False),
                ex=_REPORT_TTL,
            )
        except Exception as e:
            logger.debug(f"init_report persist failed for {agent_id}: {e}")
        logger.info(
            f"[InitReport] agent={agent_id} total={report.total_duration_ms()}ms "
            f"direct={report.direct_count} llm={report.llm_count} "
            f"dedupe={report.dedupe_removed} stored={report.total_stored} "
            f"subs={report.distinct_subs}"
        )


@contextlib.contextmanager
def phase_timer(report: InitReport, name: str):
    start = int(time.time() * 1000)
    try:
        yield
    finally:
        report.phase_ms[name] = int(time.time() * 1000) - start


async def get_init_report(agent_id: str) -> dict | None:
    """Read-back helper for debugging / admin UI."""
    try:
        redis = await get_redis()
        raw = await redis.get(report_key(agent_id))
    except Exception:
        return None
    if not raw:
        return None
    return json.loads(raw)
