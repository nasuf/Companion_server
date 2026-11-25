"""服务号 openid ↔ users.id 绑定：保证 H5 付款到账落在用户原本的账号上。

## 账号为什么不会分裂

两条链路写的是同一套 `users` / `auth_identities`：

* Flutter App 微信登录 → `exchange_wechat_code`（移动应用 appid）
* 服务号 H5 微信登录 → `exchange_wechat_h5_code`（服务号 appid）

两个 appid 已绑定到同一个微信开放平台账号，所以同一个微信用户在两边拿到的
`unionid` 相同；`_find_identity_preferring_union` 以 unionid 优先查身份，于是
两次登录命中同一行 `auth_identities`，映射到同一个 `users.id`。

openid 则不同：它是 **per-appid** 的，App 与服务号各有一个，且
`auth_identities.openid` 会被每次登录覆盖。所以本模块把服务号 openid 单独存进
`wechat_pay_openids`，不去改身份表，既不影响 App 登录，也不会用错 openid 下单。

## 到账依据

下单时就把 `user_id` 写进订单（来自 JWT，不是来自 openid）。回调只按
`out_trade_no` 找订单，再给订单上记着的 `user_id` 发权益。因此即使同一个微信号
在不同时间绑到过不同账号，历史订单也只会结算给当时下单的那个账号，不会错乱。
"""

from __future__ import annotations

import logging
from typing import Any

from app.db import db
from app.services.payments.wechat.errors import WechatPayOpenidMissingError

logger = logging.getLogger(__name__)


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


async def bind_openid(
    *, user_id: str, openid: str, app_id: str, unionid: str | None = None
) -> None:
    """登记（或改绑）一个服务号 openid 到当前账号。

    同一个微信号换绑到另一个账号时覆盖 user_id——以最后一次网页授权为准。已经
    产生的订单不受影响：它们记着下单当时的 user_id。
    """
    await db.execute_raw(
        """
        INSERT INTO wechat_pay_openids (openid, app_id, user_id, unionid)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (openid) DO UPDATE
        SET user_id = EXCLUDED.user_id,
            app_id = EXCLUDED.app_id,
            unionid = COALESCE(EXCLUDED.unionid, wechat_pay_openids.unionid),
            updated_at = CURRENT_TIMESTAMP
        """,
        openid,
        app_id,
        user_id,
        unionid,
    )
    logger.info(
        "wechat pay openid bound user=%s",
        user_id[:8],
        extra={"event": "wechat_pay_openid_bound", "user_id": user_id},
    )


async def conflicting_owner(
    *, user_id: str, openid: str, unionid: str | None
) -> str | None:
    """这个微信号是否已属于另一个账号；是则返回那个账号的 user_id。

    用于付款授权（`POST /pay/wechat/authorize`）：已登录的账号要把服务号 openid
    记到自己名下时，若这个微信本身就是另一个账号的登录方式，说明用户登错了账号，
    必须拦下，否则会出现"用 A 的微信给 B 账号付款"的错位。

    判据：unionid 命中的微信身份（最可靠，跨 App/服务号一致）；身份表里最近一次
    服务号登录留下的 openid；已登记过的付款 openid。任一命中别的账号即视为冲突。
    """
    if unionid:
        rows = await db.query_raw(
            """
            SELECT user_id FROM auth_identities
            WHERE provider = 'wechat' AND unionid = $1 AND user_id <> $2
            LIMIT 1
            """,
            unionid,
            user_id,
        )
        if rows:
            return str(_field(rows[0], "user_id"))
    rows = await db.query_raw(
        """
        SELECT user_id FROM auth_identities
        WHERE provider = 'wechat' AND openid = $1 AND user_id <> $2
        UNION ALL
        SELECT user_id FROM wechat_pay_openids
        WHERE openid = $1 AND user_id <> $2
        LIMIT 1
        """,
        openid,
        user_id,
    )
    return str(_field(rows[0], "user_id")) if rows else None


async def openid_for_user(user_id: str, *, app_id: str) -> str:
    """取当前账号在该服务号下最近一次授权的 openid；没有就要求先走网页授权。"""
    openid, _ = await payer_for_user(user_id, app_id=app_id)
    return openid


async def payer_for_user(user_id: str, *, app_id: str) -> tuple[str, str | None]:
    """同 `openid_for_user`，另带登记时的 unionid（下单前复核归属用）。"""
    rows = await db.query_raw(
        """
        SELECT openid, unionid FROM wechat_pay_openids
        WHERE user_id = $1 AND app_id = $2
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        user_id,
        app_id,
    )
    openid = _field(rows[0], "openid") if rows else None
    if not isinstance(openid, str) or not openid:
        raise WechatPayOpenidMissingError("当前账号尚未完成服务号授权")
    unionid = _field(rows[0], "unionid")
    return openid, unionid if isinstance(unionid, str) and unionid else None
