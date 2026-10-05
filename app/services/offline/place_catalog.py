"""Shared public place records. Never cache personal recommendation copy or media."""
from __future__ import annotations

import json
import logging
from app.db import db
from app.services.offline.geocode import make_place_key

logger = logging.getLogger(__name__)


async def load_place(card: dict) -> dict | None:
    key = make_place_key(card.get('location_name'), card.get('address'), card.get('city'))
    try:
        rows = await db.query_raw(
            "SELECT data FROM offline_place_catalog WHERE id=$1 AND expires_at > CURRENT_TIMESTAMP", key)
        if rows:
            value = rows[0]['data']
            return json.loads(value) if isinstance(value, str) else value
    except Exception:
        logger.warning('[offline-place] catalog read unavailable')
    return None


async def save_place(card: dict, images: list[dict]) -> None:
    key = make_place_key(card.get('location_name'), card.get('address'), card.get('city'))
    if not key:
        return
    data = {k: card.get(k) for k in ('location_name', 'address', 'city', 'official_url')}
    data['images'] = images
    try:
        await db.execute_raw('''
            INSERT INTO offline_place_catalog(id,data,expires_at)
            VALUES ($1,$2::jsonb,CURRENT_TIMESTAMP + INTERVAL '7 days')
            ON CONFLICT(id) DO UPDATE SET data=EXCLUDED.data,
                expires_at=EXCLUDED.expires_at,updated_at=CURRENT_TIMESTAMP
        ''', key, json.dumps(data, ensure_ascii=False))
    except Exception:
        logger.warning('[offline-place] catalog write unavailable')
