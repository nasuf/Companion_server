"""Chat history search — spec: Flutter 加号面板「查找」功能.

Three independent match kinds, each scoped to one conversation:
  - text:  Message.content substring match (same pattern as memories.py search)
  - card:  Message.metadata->'component_card' substring match on title/subtitle/
           body/footer — no JSON index exists, so we scan the conversation's
           most recent card-bearing rows via raw SQL and filter in Python.
  - image: ChatMessageAttachment.vision_summary substring match — every image
           already gets a Doubao vision summary at send time (chat_media/
           vision.py), so this reuses that text for free instead of adding OCR.

`scope="all"` returns a small preview per kind (+ has_more flags) with no
cross-kind pagination — merging three independent cursors into one stable
page is not worth the complexity here. A single `scope` does real
limit/offset pagination within that one kind.
"""

from __future__ import annotations

from app.db import db
from app.models.message import MessageSearchHit, MessageSearchResponse

# Safety valve, not a realistic ceiling — a single conversation's card
# messages are expected to be in the dozens/hundreds, never near this.
_CARD_SCAN_LIMIT = 5000
_PREVIEW_COUNT = 5

# Quick-filter categories the Flutter search landing page offers instead of
# one generic "卡片" tile. Keyed by the semantic category the client sends;
# each maps to the underlying component_card `type` string(s) — "gift" is
# deliberately two types (in-chat wallet gift vs. AI-initiated real-world
# gift delivery, see offerings.py / offline/chat_emit.py) that read as one
# feature to a user searching for "礼物".
_CARD_CATEGORY_TYPES: dict[str, frozenset[str]] = {
    "music": frozenset({"music_track"}),
    "checkin": frozenset({"checkin_reminder", "checkin_habit"}),
    "capsule": frozenset({"time_capsule"}),
    "gift": frozenset({"gift", "offline_gift"}),
    "red_packet": frozenset({"red_packet"}),
    # AI-initiated (or user-requested) offline activity recommendations —
    # same card type either way, see offline/activity_service.py.
    "activity": frozenset({"offline_activity"}),
}


async def search_messages(
    *,
    conversation_id: str,
    q: str | None,
    scope: str,
    limit: int,
    offset: int,
    card_category: str | None = None,
) -> MessageSearchResponse:
    query = (q or "").strip() or None

    if scope == "all":
        text_rows, text_more = await _search_text(conversation_id, query, _PREVIEW_COUNT, 0)
        card_rows, card_more = await _search_cards(
            conversation_id, query, None, _PREVIEW_COUNT, 0
        )
        image_rows, image_more = await _search_images(conversation_id, query, _PREVIEW_COUNT, 0)
    elif scope == "text":
        text_rows, text_more = await _search_text(conversation_id, query, limit, offset)
        card_rows, card_more = [], False
        image_rows, image_more = [], False
    elif scope == "card":
        text_rows, text_more = [], False
        card_rows, card_more = await _search_cards(
            conversation_id, query, card_category, limit, offset
        )
        image_rows, image_more = [], False
    elif scope == "image":
        text_rows, text_more = [], False
        card_rows, card_more = [], False
        image_rows, image_more = await _search_images(conversation_id, query, limit, offset)
    else:
        raise ValueError(f"unknown search scope: {scope!r}")

    all_ids = [row["id"] for row in (*text_rows, *card_rows, *image_rows)]
    ranks = await _ranks_for(conversation_id, all_ids)

    return MessageSearchResponse(
        text=[_to_hit(row, "text", ranks) for row in text_rows],
        cards=[_to_hit(row, "card", ranks) for row in card_rows],
        images=[_to_hit(row, "image", ranks) for row in image_rows],
        has_more_text=text_more,
        has_more_cards=card_more,
        has_more_images=image_more,
    )


def _to_hit(row: dict, match_type: str, ranks: dict[str, int]) -> MessageSearchHit:
    return MessageSearchHit(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        metadata=row.get("metadata"),
        created_at=str(row["created_at"]) if row.get("created_at") else None,
        match_type=match_type,
        rank=ranks.get(row["id"], 0),
        matched_attachment_id=row.get("matched_attachment_id"),
    )


async def _search_text(
    conversation_id: str, query: str | None, limit: int, offset: int
) -> tuple[list[dict], bool]:
    where: dict = {
        "conversationId": conversation_id,
        "role": {"in": ["user", "assistant"]},
    }
    if query:
        # Same contains/insensitive pattern as memories.py list_memories search.
        where["content"] = {"contains": query, "mode": "insensitive"}
    rows = await db.message.find_many(
        where=where,
        order={"createdAt": "desc"},
        take=limit + 1,
        skip=offset,
    )
    has_more = len(rows) > limit
    rows = rows[:limit]
    return [
        {
            "id": m.id,
            "conversation_id": m.conversationId,
            "role": m.role,
            "content": m.content,
            "metadata": m.metadata,
            "created_at": m.createdAt,
        }
        for m in rows
    ], has_more


async def _search_cards(
    conversation_id: str,
    query: str | None,
    card_category: str | None,
    limit: int,
    offset: int,
) -> tuple[list[dict], bool]:
    # Red packet / in-chat gift cards render fixed boilerplate text ("红包" /
    # "给你的一点心意" for every single one — offerings.py:build_red_packet_card)
    # — the user's actual blessing message and the amount live in
    # user_offerings (linked 1:1 via message_id, see offerings.py's
    # bind-to-message step), not in the card's own title/subtitle/body/footer.
    # Left-joining it here is what makes a red packet findable by its
    # blessing/amount instead of every red packet looking identical to search.
    rows = await db.query_raw(
        """
        SELECT m.id, m.conversation_id, m.role, m.content, m.metadata, m.created_at,
               uo.blessing AS offering_blessing,
               uo.ticket_amount AS offering_ticket_amount
        FROM messages m
        LEFT JOIN user_offerings uo ON uo.message_id = m.id
        WHERE m.conversation_id = $1
          AND m.metadata -> 'component_card' IS NOT NULL
        ORDER BY m.created_at DESC
        LIMIT $2
        """,
        conversation_id,
        _CARD_SCAN_LIMIT,
    )
    allowed_types = _CARD_CATEGORY_TYPES.get(card_category) if card_category else None
    if allowed_types:
        rows = [row for row in rows if _card_type(row) in allowed_types]
    if query:
        needle = query.lower()
        rows = [row for row in rows if _card_text_matches(row, needle)]
    has_more = len(rows) > offset + limit
    page = rows[offset : offset + limit]
    return list(page), has_more


def _card(row: dict) -> dict | None:
    metadata = row.get("metadata")
    card = metadata.get("component_card") if isinstance(metadata, dict) else None
    return card if isinstance(card, dict) else None


def _card_type(row: dict) -> str | None:
    card = _card(row)
    card_type = card.get("type") if card else None
    return str(card_type) if card_type else None


def _card_text_matches(row: dict, needle: str) -> bool:
    card = _card(row)
    if card is None:
        return False
    haystack_parts = [
        str(card.get(field) or "") for field in ("title", "subtitle", "body", "footer")
    ]
    blessing = row.get("offering_blessing")
    if blessing:
        haystack_parts.append(str(blessing))
    ticket_amount = row.get("offering_ticket_amount")
    if ticket_amount is not None:
        haystack_parts.append(str(ticket_amount))
    haystack = " ".join(haystack_parts).lower()
    return needle in haystack


def _escape_like_pattern(value: str) -> str:
    """Escape LIKE/ILIKE wildcards so `query` matches as a literal substring.

    Prisma's `contains` filter (used for text/card matching) does this
    automatically; this hand-written raw query needs it spelled out, or a
    user searching for e.g. "50%" would have '%' act as a wildcard instead
    of a literal character.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def _search_images(
    conversation_id: str, query: str | None, limit: int, offset: int
) -> tuple[list[dict], bool]:
    # A single JOIN, paginated at the SQL level, rather than paginating
    # attachment rows and then looking up messages in a second query: the
    # two-step version let an attachment whose message failed to resolve
    # (should be impossible — Message->ChatMessageAttachment is an onDelete:
    # Cascade FK — but was silently dropped post-slice) desync the page's
    # returned-item count from the offset the client advances by, which
    # would skip/duplicate a row on the next page. A JOIN can't have that
    # problem: a row only exists in the result if the message resolves.
    if query:
        rows = await db.query_raw(
            """
            SELECT m.id, m.conversation_id, m.role, m.content, m.metadata,
                   m.created_at, a.id AS matched_attachment_id
            FROM chat_message_attachments a
            JOIN messages m ON m.id = a.message_id
            WHERE a.conversation_id = $1
              AND a.kind = 'image'
              AND a.vision_summary ILIKE '%' || $2 || '%' ESCAPE '\\'
            ORDER BY m.created_at DESC
            LIMIT $3 OFFSET $4
            """,
            conversation_id,
            _escape_like_pattern(query),
            limit + 1,
            offset,
        )
    else:
        rows = await db.query_raw(
            """
            SELECT m.id, m.conversation_id, m.role, m.content, m.metadata,
                   m.created_at, a.id AS matched_attachment_id
            FROM chat_message_attachments a
            JOIN messages m ON m.id = a.message_id
            WHERE a.conversation_id = $1
              AND a.kind = 'image'
            ORDER BY m.created_at DESC
            LIMIT $2 OFFSET $3
            """,
            conversation_id,
            limit + 1,
            offset,
        )
    has_more = len(rows) > limit
    return list(rows[:limit]), has_more


async def _ranks_for(conversation_id: str, message_ids: list[str]) -> dict[str, int]:
    if not message_ids:
        return {}
    rows = await db.query_raw(
        """
        SELECT m.id, (
            SELECT COUNT(*) FROM messages m2
            WHERE m2.conversation_id = m.conversation_id AND m2.created_at > m.created_at
        ) AS rank
        FROM messages m
        WHERE m.conversation_id = $1 AND m.id = ANY($2::text[])
        """,
        conversation_id,
        message_ids,
    )
    return {row["id"]: int(row["rank"]) for row in rows}
