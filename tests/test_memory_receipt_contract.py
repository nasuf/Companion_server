"""Untrusted plans and public time metadata are bounded and fail closed."""

from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.services.memory.recording import receipts, receipt_read, temporal_sources


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"memories": {}},
        {"memories": [None]},
        {"memories": [{}] * 101},
        {"memories": [{"content": "x" * 262145}]},
    ],
)
def test_invalid_plan_is_rejected(payload):
    with pytest.raises(receipts.MemoryReceiptError):
        receipts._sanitize(payload, 1)


@pytest.mark.parametrize("index", [-1, 100])
def test_split_identity_has_a_finite_range(index):
    token = receipts.WriteToken("batch", 0, 0, None, "user")
    with pytest.raises(receipts.MemoryReceiptError):
        token.piece(index)


def test_model_cannot_supply_internal_clock_or_erasure_controls():
    raw = {
        "memories": [
            {
                "_erased": True,
                "_clock": {"occur_time": "2099"},
                "source_indices": [True],
                "entities": {},
                "topics": None,
            }
        ],
        "entities": {},
        "topics": None,
        "preferences": 3,
    }
    clean = receipts._sanitize(raw, 1)
    assert clean["memories"] == [{"source_indices": [], "entities": [], "topics": []}]
    assert clean["entities"] == clean["topics"] == clean["preferences"] == []
    assert raw["memories"][0]["_erased"] is True


@pytest.mark.parametrize(
    "fact", [{}, {"_clock": None}, {"_clock": {"statement_time": "not-time"}}]
)
def test_corrupt_frozen_clock_is_not_reparsed_using_today(fact):
    with pytest.raises(receipts.MemoryReceiptError):
        receipts.saved_clock(fact)


@pytest.mark.parametrize(
    "field,value",
    [
        ("precision", {}),
        ("precision", []),
        ("precision", "bad"),
        ("timezone", None),
        ("timezone", "x" * 65),
        ("timezone", "Missing/Zone"),
    ],
)
def test_public_clock_metadata_does_not_crash_or_echo_bad_values(field, value):
    basis = {
        "version": 1,
        "status": "time_not_stated",
        "source_role": "fact",
        field: value,
    }
    assert temporal_sources.public_temporal_basis(basis) == {
        "status": "invalid_metadata"
    }


def test_public_clock_keeps_explicit_timezone_and_precision():
    basis = {
        "version": 1,
        "status": "time_not_stated",
        "source_role": "fact",
        "timezone": "Asia/Shanghai",
        "precision": "minute",
    }
    assert temporal_sources.public_temporal_basis(basis) == basis


@pytest.mark.parametrize("limit", [0, 51])
async def test_receipt_query_rejects_unbounded_page(limit):
    with pytest.raises(ValueError):
        await receipt_read.recent_receipts(limit=limit)


async def test_legacy_storage_without_token_does_not_invent_receipts():
    assert await receipts.claim(None, None, "hash") is None
    assert await receipts.lookup(None, "hash") is None
    assert await receipts.commit(None, None, "hash", None) is None
    assert await receipts.record_skip(None, "hash") is None


def test_invalid_fact_types_cannot_poison_a_durable_retry():
    with pytest.raises(receipts.MemoryReceiptError):
        receipts._sanitize({"memories": [{"content": {"invented": "object"}}]}, 1)
    clean = receipts._sanitize(
        {
            "memories": [
                {
                    "content": "合成事实",
                    "main_category": [],
                    "sub_category": {},
                    "type": False,
                    "recurrence": {"daily": True},
                }
            ]
        },
        1,
    )
    assert all(
        clean["memories"][0][key] is None
        for key in ("main_category", "sub_category", "type", "recurrence")
    )
