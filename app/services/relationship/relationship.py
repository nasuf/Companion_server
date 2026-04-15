"""Relationship evolution tracking.

User↔AI relationship state (trust / interaction count / growth stage)
lives in the Postgres `intimacies` table — see schema.prisma::Intimacy and
the dedicated intimacy service. This module is a thin view helper kept
for back-compat with any caller that expects a flat trust/count shape.

If you want the rich relationship model (topic intimacy + growth levels
+ updated_at), call `app.services.relationship.intimacy` directly.
"""

import logging
import uuid

from app.db import db

logger = logging.getLogger(__name__)


async def update_relationship(
    user_id: str,
    agent_id: str,
    workspace_id: str | None = None,
    interaction_quality: float = 0.5,
) -> None:
    """EMA-update the pair's growth intimacy using the passed quality.

    Retained for back-compat; new code should use the richer
    `relationship.intimacy` API which also handles topic-level signals.
    """
    _ = workspace_id  # Intimacy is (agent, user) unique; ws not needed here
    # Client-side UUID (avoids depending on pgcrypto's gen_random_uuid,
    # which isn't enabled by our baseline migration and only ships with
    # PostgreSQL ≥ 13). EMA on existing value: new = 0.9*old + 0.1*q,
    # clamped to [0, 1000].
    quality_scaled = max(0, min(1000, int(interaction_quality * 1000)))
    await db.execute_raw(
        """
        INSERT INTO intimacies
            (id, agent_id, user_id, growth_intimacy, growth_updated_at,
             created_at, updated_at)
        VALUES ($1, $2, $3, $4, NOW(), NOW(), NOW())
        ON CONFLICT (agent_id, user_id) DO UPDATE SET
            growth_intimacy = LEAST(1000, GREATEST(0,
                (intimacies.growth_intimacy * 9 + EXCLUDED.growth_intimacy) / 10
            )),
            growth_updated_at = NOW(),
            updated_at = NOW()
        """,
        str(uuid.uuid4()), agent_id, user_id, quality_scaled,
    )


async def get_relationship(
    user_id: str,
    agent_id: str,
    workspace_id: str | None = None,
) -> dict | None:
    """Return a flat {trust, interaction_count} view. Interaction count is
    approximated by number of conversations (cheap JOIN); topic intimacy is
    not included — call the intimacy service for that."""
    _ = workspace_id
    rows = await db.query_raw(
        """
        SELECT i.growth_intimacy::float / 1000 AS trust,
               COALESCE(c.conversation_count, 0) AS interaction_count
        FROM intimacies i
        LEFT JOIN (
            SELECT agent_id, user_id, COUNT(*) AS conversation_count
            FROM conversations
            WHERE agent_id = $1 AND user_id = $2
            GROUP BY agent_id, user_id
        ) c ON c.agent_id = i.agent_id AND c.user_id = i.user_id
        WHERE i.agent_id = $1 AND i.user_id = $2
        """,
        agent_id, user_id,
    )
    return dict(rows[0]) if rows else None
