"""No provider IO: bind the authoritative immutable version inside admission."""

from typing import Any

from app.services.chat_links.cards import component_card_for_link, metadata_for_link_card
from app.services.chat_links.extraction import extract_first_url
from app.services.chat_links.repo import find_link_card
from app.services.runtime.execution_scope import ExecutionScope


async def bind_ingress_link(tx: Any, scope: ExecutionScope, original: dict,
                            message_id: str, metadata: dict) -> str | None:
    # Local import keeps the effects adapter and its resource error one contract.
    from app.services.runtime.chat_ingress_effects import ChatIngressResourceInvalid

    card = original["component_card"]
    card_type = card.get("type") if card else None
    projection = metadata.get("link_card")
    expected = card_type == "external_link" or (card is None and extract_first_url(original["text"]))
    if not expected:
        if projection is not None:
            raise ChatIngressResourceInvalid()
        return None
    if type(projection) is not dict:
        raise ChatIngressResourceInvalid()
    link_id, version = projection.get("id"), projection.get("snapshot_id")
    if any(type(value) is not str or not value or len(value) > 256 for value in (link_id, version)):
        raise ChatIngressResourceInvalid()
    rows = await tx.query_raw(
        "SELECT s.id FROM chat_link_snapshots s JOIN chat_link_cards l ON l.id=s.link_card_id "
        "WHERE s.id=$1 AND l.id=$2 AND l.user_id=$3 AND l.conversation_id=$4 AND l.role IN ('user','assistant') "
        "FOR SHARE OF s,l", version, link_id, scope.owner_user_id, scope.conversation_id)
    if len(rows) != 1:
        raise ChatIngressResourceInvalid()
    link = await find_link_card(link_id=link_id, snapshot_id=version,
        user_id=scope.owner_user_id, conversation_id=scope.conversation_id, database=tx)
    if (link is None or projection != metadata_for_link_card(link)
            or metadata.get("component_card") != component_card_for_link(link)):
        raise ChatIngressResourceInvalid()
    if card:
        payload = card.get("payload", {})
        if type(payload) is not dict:
            raise ChatIngressResourceInvalid()
        if "snapshot_id" in payload and payload["snapshot_id"] != version:
            raise ChatIngressResourceInvalid()
        origin = payload.get("link_id")
        if origin and origin != link_id:
            # Redirects or resharing an assistant recommendation can create a
            # different cache. Only a server-recorded, same-conversation refresh
            # may bridge that identity, never a supplied pinned version.
            if (payload.get("snapshot_id") or
                    (link.metadata or {}).get("refreshed_from_link_id") != origin or
                    await find_link_card(link_id=origin, user_id=scope.owner_user_id,
                        conversation_id=scope.conversation_id, database=tx) is None):
                raise ChatIngressResourceInvalid()
    await tx.execute_raw(
        "INSERT INTO chat_link_message_bindings(message_id,snapshot_id) VALUES ($1,$2)",
        message_id, version)
    return version
