"""Immutable bank, scope-aware identities, and a stratified family-level split."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

BANK = Path(__file__).with_name("cases.jsonl")
GROUPS = {
    "persona",
    "preference",
    "alias",
    "negation",
    "correction",
    "multi_agent",
    "reminder",
    "temporal",
    "l3",
    "long_memory",
    "deletion",
    "concurrency",
}
SEED = 20261010


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def load_cases(path: Path = BANK) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids, families = set(), set()
    for case in cases:
        cid, family = case["id"], case["family"]
        if cid in ids or family in families or case["group"] not in GROUPS:
            raise ValueError("Duplicate case/family or unsupported group")
        ids.add(cid)
        families.add(family)
        if not case["query"].strip() or case.get("data_class") != "synthetic":
            raise ValueError("Only labeled synthetic scenarios are permitted")
        keys = {m["key"] for m in case["memories"]}
        if len(keys) != len(case["memories"]):
            raise ValueError("Duplicate side-qualified memory identity")
        for m in case["memories"]:
            if (
                m["source"] not in {"user", "ai"}
                or m["scope"] not in {"target", "other_agent", "other_user"}
                or not m["text"]
                or m["level"] not in {1, 2, 3}
                or m["key"].split(":", 1)[0] != m["source"]
            ):
                raise ValueError("Invalid memory fixture")
        expected, forbidden = set(case["expected"]), set(case["forbidden"])
        if not expected <= keys or not forbidden <= keys or expected & forbidden:
            raise ValueError("Invalid expected/forbidden identities")
        if any(
            m["scope"] != "target" or m.get("archived")
            for m in case["memories"]
            if m["key"] in expected
        ):
            raise ValueError("Expected evidence cannot be archived or cross-scope")
        if (
            type(case["critical_model"]) is not bool
            or type(case["hard_invariant"]) is not bool
        ):
            raise ValueError("Explicit model/hard-invariant qualification required")
    if len(cases) < 240 or set(Counter(c["group"] for c in cases)) != GROUPS:
        raise ValueError(
            "At least 240 scenarios covering every required group are required"
        )
    return cases


def split_cases(cases: list[dict], seed: int = SEED) -> dict[str, str]:
    """Partition independent families; this bank has one scenario per family.

    Exactly 30% per stratum; this bank requires stratum sizes divisible by ten.
    The seed and resulting membership are included in every run manifest.
    """
    out = {}
    for group in sorted(GROUPS):
        members = [c for c in cases if c["group"] == group]
        if len(members) % 10:
            raise ValueError("Stratum size must support an exact 30% holdout")
        ordered = sorted(members, key=lambda c: fingerprint([seed, c["family"]]))
        held = {c["family"] for c in ordered[: len(ordered) * 3 // 10]}
        out.update(
            {
                c["id"]: "holdout" if c["family"] in held else "development"
                for c in members
            }
        )
    return out
