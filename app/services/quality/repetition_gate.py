"""CHAT01.10.2 repetition gate: manifest, block budget and case import.

The manual GitHub workflow replays the human-reviewed, sanitized cases on two
source trees. Its trusted controller meters every provider call through a local
proxy that reserves budget here in blocks (one block covers many calls), so the
hours-long replay costs a few hundred control requests, not tens of thousands.
Cases live only in this database; the public repository never holds them.
"""

from decimal import Decimal

from app.config import settings
from app.db import db
from app.services.prompting.store import history_requires_web
from app.services.quality import snapshots, transport
from app.services.quality.policy import GATE_KIND, money

EMBEDDING_MODEL = "dashscope/text-embedding-v4"
JUDGE_MODELS = ("dashscope/qwen3.5-flash", "dashscope/qwen3.5-plus")
MAX_BLOCK_CNY = Decimal("2")
MAX_CASES = 1000
MODES = ("gate", "calibrate")


async def protected_prompts() -> list[str]:
    """Keys whose production text a deploy keeps (Web-managed); others follow code."""
    rows = await db.prompttemplate.find_many()
    versions = await db.prompttemplateversion.find_many(
        where={"promptKey": {"in": [r.key for r in rows]}}
    )
    by_key: dict[str, list] = {}
    for version in versions:
        by_key.setdefault(version.promptKey, []).append(version)
    return sorted(r.key for r in rows if history_requires_web(r, by_key.get(r.key, [])))


async def snapshot() -> dict:
    value: dict = await snapshots.capture()
    price = settings.quality_embedding_cny_per_million
    prices: dict = value["prices"]
    prices[EMBEDDING_MODEL] = dict(input=price, output=price, cached_input=price)
    value["protected_prompts"] = await protected_prompts()
    return value


def models(job: dict) -> set[str]:
    return {"/".join(job["snapshot"]["route"]), *JUDGE_MODELS, EMBEDDING_MODEL}


async def cases(case_set: str) -> tuple[list[dict], dict]:
    rows = await db.query_raw(
        "SELECT id,case_data,context FROM quality_repetition_cases WHERE case_set=$1 ORDER BY id LIMIT $2",
        case_set,
        MAX_CASES + 1,
    )
    if len(rows) > MAX_CASES:
        raise ValueError("repetition_cases_exceed_bound")
    return [r["case_data"] for r in rows], {r["case_data"]["id"]: r["context"] for r in rows}


def block_amount(value) -> Decimal:
    amount = money(value)
    if not 0 < amount <= MAX_BLOCK_CNY:
        raise ValueError("invalid_block_amount")
    return amount


def block_cost(job: dict, model: str, usage: dict) -> float:
    """Server-side price of one settled block, from the job's pinned prices.

    `unknown_cny` is the reserved bound of calls whose outcome is ambiguous
    (timeout, missing usage): conservatively charged, never assumed free.
    """
    counts = ("input", "output", "cached_input", "calls")
    if any(type(usage.get(k)) is not int or usage[k] < 0 for k in counts):
        raise ValueError("invalid_block_usage")
    if usage["cached_input"] > usage["input"]:
        raise ValueError("invalid_block_usage")
    unknown = usage.get("unknown_cny", 0)
    if type(unknown) not in (int, float) or not 0 <= unknown <= float(MAX_BLOCK_CNY):
        raise ValueError("invalid_block_usage")
    price = transport.pinned_price(model, job["snapshot"]["prices"])
    return transport.price_cost(
        price, usage["input"], usage["output"], usage["cached_input"]
    ) + float(unknown)


async def import_cases(tx, case_set: str, items: list[dict]) -> int:
    """One-off load of a reviewed set. A set is immutable once loaded."""
    if not items or len(items) > MAX_CASES:
        raise ValueError("invalid_case_count")
    existing = await tx.query_raw(
        "SELECT count(*)::int AS n FROM quality_repetition_cases WHERE case_set=$1", case_set
    )
    if existing[0]["n"]:
        raise ValueError("case_set_already_loaded")
    for item in items:
        case, context = item["case"], item["context"]
        await tx.execute_raw(
            "INSERT INTO quality_repetition_cases(id,case_set,case_data,context) VALUES($1,$2,$3::jsonb,$4::jsonb)",
            f"{case_set}:{case['id']}",
            case_set,
            _json(case),
            _json(context),
        )
    return len(items)


def _json(value) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
