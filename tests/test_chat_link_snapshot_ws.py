from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.realtime import ws
from app.services.chat_links import repo, cards
from tests.test_chat_link_snapshots_postgres import links, origins


async def test_preview_version_survives_sanitization_and_changed_cache(
    links, monkeypatch
):
    _, uid, _, convs, prepare, _ = links
    old = await prepare("saved preview")
    await prepare("latest page")
    monkeypatch.setattr(ws, "find_link_card", repo.find_link_card)
    fetch = AsyncMock(
        side_effect=AssertionError("a pinned preview must not be fetched again")
    )
    monkeypatch.setattr(ws, "extract_link_metadata", fetch)
    incoming = cards.component_card_for_link(old)
    incoming["payload"]["content_text"] = "forged"
    sanitized = ws._sanitize_component_card(incoming)
    assert sanitized["payload"]["snapshot_id"] == old.snapshot_id
    card, metadata = await ws._ensure_link_card_for_turn(
        user_id=uid, conversation_id=convs[0], text="看看这个", component_card=sanitized
    )
    assert (
        metadata["title"] == "saved preview"
        and metadata["content_text"] == "saved preview"
    )
    assert card["payload"]["snapshot_id"] == old.snapshot_id
    fetch.assert_not_called()
    # Old clients omit the added key and still resolve the current cache.
    incoming["payload"].pop("snapshot_id")
    _, legacy = await ws._ensure_link_card_for_turn(
        user_id=uid,
        conversation_id=convs[0],
        text="",
        component_card=ws._sanitize_component_card(incoming),
    )
    assert legacy["title"] == "latest page"
    incoming["payload"]["snapshot_id"] = "missing"
    with pytest.raises(ValueError, match="unavailable"):
        await ws._ensure_link_card_for_turn(
            user_id=uid,
            conversation_id=convs[0],
            text="",
            component_card=ws._sanitize_component_card(incoming),
        )


async def test_unavailable_snapshot_reports_correlated_error_without_persist(
    monkeypatch,
):
    from app.services.admin_takeover import service

    monkeypatch.setattr(
        service, "is_conversation_taken_over", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(ws.wallet, "is_vip", AsyncMock(return_value=False))
    monkeypatch.setattr(
        ws.chat_quota, "consume_one", AsyncMock(return_value={"allowed": True})
    )
    monkeypatch.setattr(
        ws.chat_media_repo, "get_message_attachments", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(ws, "ensure_vision_summaries", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        ws,
        "_ensure_link_card_for_turn",
        AsyncMock(side_effect=ValueError("unavailable")),
    )
    persist = AsyncMock()
    monkeypatch.setattr(ws, "_persist_user_message", persist)
    socket = SimpleNamespace(send_json=AsyncMock())
    await ws._handle_message(
        socket, "conv", "owner", SimpleNamespace(id="agent"), "", client_id="retry-id"
    )
    result = socket.send_json.await_args.args[0]
    assert result["type"] == "error" and result["data"]["client_id"] == "retry-id"
    persist.assert_not_called()


async def test_version_without_cache_identity_is_rejected():
    with pytest.raises(ValueError, match="cache identity"):
        await ws._ensure_link_card_for_turn(
            user_id="u",
            conversation_id="c",
            text="",
            component_card={
                "type": "external_link",
                "payload": {"snapshot_id": "v", "source_url": "https://example.com"},
            },
        )
