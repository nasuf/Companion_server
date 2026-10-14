"""Link preparation outside admission locks, with a generation-fenced SQL save.

The request receipt is the retry identity; the returned snapshot is the exact
version selected by this preparation. Competing preparations may leave unused
previews, but only the winning admission can charge and bind a message.
"""

import json
from uuid import uuid4

from app.services.chat_links import repo
from app.services.chat_links.extraction import extract_first_url, extract_link_metadata
from app.services.runtime.chat_ingress_contracts import ChatRequestInput
from app.services.runtime.chat_ingress_effects import ChatIngressResourceInvalid
from app.services.runtime.execution_scope import ExecutionScope, revalidate_scope, scoped_transaction


async def prepare_ingress_link(scope: ExecutionScope, request: ChatRequestInput) -> repo.ChatLinkCard | None:
    original = json.loads(request.input_json)
    card = original["component_card"]
    text = original["text"].strip()
    if card is not None and card.get("type") != "external_link":
        return None
    if card is None and not extract_first_url(text):
        return None
    await revalidate_scope(scope)
    payload = card.get("payload", {}) if card else {}
    if type(payload) is not dict:
        raise ChatIngressResourceInvalid()
    for key in ("link_id", "snapshot_id"):
        if key in payload and (type(payload[key]) is not str or not payload[key]
                               or len(payload[key]) > 256 or payload[key] != payload[key].strip()):
            raise ChatIngressResourceInvalid()
    link_id, version = payload.get("link_id"), payload.get("snapshot_id")
    if version and not link_id:
        raise ChatIngressResourceInvalid()
    if link_id:
        link = await repo.find_link_card(link_id=link_id, snapshot_id=version,
            user_id=scope.owner_user_id, conversation_id=scope.conversation_id)
        # Never turn a missing/foreign identity into a new URL fetch.
        if link is None or link.role != "user":
            raise ChatIngressResourceInvalid()
        if not link.snapshot_id:
            # Old caches remain compatible. Capture the selected cache under the
            # same lifecycle fence, without pretending it was a historical version.
            async with scoped_transaction(scope) as tx:
                rows = await tx.query_raw(
                    "SELECT id FROM chat_link_cards WHERE id=$1 AND user_id=$2 "
                    "AND conversation_id=$3 AND role='user' FOR SHARE", link_id,
                    scope.owner_user_id, scope.conversation_id)
                if not rows:
                    raise ChatIngressResourceInvalid()
                version = str(uuid4())
                await tx.execute_raw(
                    "INSERT INTO chat_link_snapshots(id,link_card_id,payload) "
                    "SELECT $1,id,to_jsonb(l)-'message_id'-'created_at'-'updated_at' "
                    "FROM chat_link_cards l WHERE id=$2", version, link_id)
                link = await repo.find_link_card(link_id=link_id, snapshot_id=version,
                    user_id=scope.owner_user_id, conversation_id=scope.conversation_id, database=tx)
                if link is None:
                    raise ChatIngressResourceInvalid()
        return link
    url = payload.get("source_url") or payload.get("final_url")
    if url is not None and (type(url) is not str or len(url) > 2000):
        raise ChatIngressResourceInvalid()
    if not (url or extract_first_url(text)):
        raise ChatIngressResourceInvalid()
    try:
        metadata = await extract_link_metadata(url=url, shared_text=text)
    except ValueError as error:
        raise ChatIngressResourceInvalid() from error
    return await repo.create_or_update_link_card(
        user_id=scope.owner_user_id, conversation_id=scope.conversation_id,
        metadata=metadata, role="user", source_app="sql_chat",
        execution_scope=scope,
    )
