"""Clear all VIP activation codes and redemption records, then recompute VIP.

Usage:
    cd Companion_server
    PYTHONPATH=. python scripts/clear_vip_activation_codes.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main() -> None:
    from app.db import db
    from app.services.vip.entitlements import recompute_vip_entitlements

    await db.connect()
    try:
        code_rows = await db.query_raw("SELECT COUNT(*) AS cnt FROM vip_activation_codes")
        redeem_rows = await db.query_raw("SELECT COUNT(*) AS cnt FROM vip_code_redemptions")
        code_cnt = int(code_rows[0]["cnt"] if isinstance(code_rows[0], dict) else code_rows[0].cnt)
        redeem_cnt = int(
            redeem_rows[0]["cnt"] if isinstance(redeem_rows[0], dict) else redeem_rows[0].cnt
        )
        print(f"Before: codes={code_cnt}, redemptions={redeem_cnt}")

        affected = await db.query_raw(
            """
            SELECT DISTINCT user_id
            FROM vip_code_redemptions
            WHERE status = 'granted'
            """
        )
        user_ids = [
            str(row["user_id"] if isinstance(row, dict) else row.user_id) for row in affected
        ]

        deleted_redeem = await db.execute_raw("DELETE FROM vip_code_redemptions")
        deleted_codes = await db.execute_raw("DELETE FROM vip_activation_codes")
        print(f"Deleted: codes={deleted_codes}, redemptions={deleted_redeem}")

        changed = 0
        for user_id in user_ids:
            if await recompute_vip_entitlements(user_id, clear_lapse=True):
                changed += 1
        print(f"Recomputed VIP for {len(user_ids)} users ({changed} wallet rows updated)")
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
