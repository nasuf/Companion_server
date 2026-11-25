"""微信支付端到端：真实路由 + 真实 PostgreSQL/Redis，只把微信服务器换成假的。

假微信服务器不是 mock 掉签名：它用商户公钥校验每个请求的 Authorization，
用"微信支付公钥"对应的私钥给应答和回调签名，回调 resource 用 APIv3 key 做
AES-256-GCM 加密——与线上的密码学路径逐字节一致。微信 OAuth（code→openid/
unionid）也换成固定映射，其余（建号、绑定、下单、结算、退款、对账、VIP 重算）
全部走生产代码和真实 SQL。

需要隔离库：PROACTIVE_E2E_DATABASE_URL（/companion_proactive_e2e，已跑迁移）与
PROACTIVE_E2E_REDIS_URL（本机）。
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import FastAPI
from prisma import Prisma

from app.config import settings

MCH_ID = "1900000109"
APP_ID = "wx-oa-e2e"
PUB_KEY_ID = "PUB_KEY_ID_E2E0000000000000000000000000001"
API_V3_KEY = secrets.token_hex(16)  # 32 个 ASCII 字符
NOTIFY_URL = "https://e2e.test/api/pay/wechat/notifications"
CST = timezone(timedelta(hours=8))


def _pem_private(key: rsa.RSAPrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _pem_public(key: rsa.RSAPrivateKey) -> bytes:
    return key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


@dataclass
class FakeWechatPay:
    """api.mch.weixin.qq.com 的替身：验商户签名、签应答、造回调。"""

    merchant_key: rsa.RSAPrivateKey
    platform_key: rsa.RSAPrivateKey
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    refunds: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)

    # ── 签名 ────────────────────────────────────────────────
    def _verify_merchant(self, request: httpx.Request, body: str) -> None:
        auth = request.headers["Authorization"]
        scheme, _, params = auth.partition(" ")
        assert scheme == "WECHATPAY2-SHA256-RSA2048"
        fields = dict(
            part.split("=", 1) for part in params.split(",")
        )
        fields = {k: v.strip('"') for k, v in fields.items()}
        assert fields["mchid"] == MCH_ID
        path = request.url.raw_path.decode()
        message = (
            f"{request.method}\n{path}\n{fields['timestamp']}\n"
            f"{fields['nonce_str']}\n{body}\n"
        ).encode()
        # 抛 InvalidSignature 即测试失败：商户签名必须能被微信验过。
        self.merchant_key.public_key().verify(
            base64.b64decode(fields["signature"]), message, padding.PKCS1v15(), hashes.SHA256()
        )

    def sign(self, body: str) -> dict[str, str]:
        timestamp = str(int(time.time()))
        nonce = secrets.token_hex(16)
        signature = self.platform_key.sign(
            f"{timestamp}\n{nonce}\n{body}\n".encode(), padding.PKCS1v15(), hashes.SHA256()
        )
        return {
            "Wechatpay-Timestamp": timestamp,
            "Wechatpay-Nonce": nonce,
            "Wechatpay-Signature": base64.b64encode(signature).decode(),
            "Wechatpay-Serial": PUB_KEY_ID,
        }

    def _json(self, status: int, payload: dict[str, Any]) -> httpx.Response:
        body = json.dumps(payload)
        return httpx.Response(
            status, content=body.encode(), headers={**self.sign(body), "Content-Type": "application/json"}
        )

    # ── API ─────────────────────────────────────────────────
    fail_next_refund_submit: bool = False

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if self.fail_next_refund_submit and request.url.path == "/v3/refund/domestic/refunds":
            self.fail_next_refund_submit = False
            raise httpx.ReadTimeout("simulated", request=request)
        body = request.content.decode() if request.content else ""
        self._verify_merchant(request, body)
        path = request.url.path
        self.calls.append((request.method, path))
        data = json.loads(body) if body else {}

        if request.method == "POST" and path == "/v3/pay/transactions/jsapi":
            assert data["appid"] == APP_ID and data["mchid"] == MCH_ID
            assert data["notify_url"] == NOTIFY_URL
            datetime.fromisoformat(data["time_expire"])  # 必须是合法 RFC3339
            prepay_id = "wx" + secrets.token_hex(12)
            self.orders[data["out_trade_no"]] = {
                "request": data, "trade_state": "NOTPAY", "prepay_id": prepay_id,
            }
            return self._json(200, {"prepay_id": prepay_id})

        if path.startswith("/v3/pay/transactions/out-trade-no/"):
            out_trade_no = path.split("/")[5]
            order = self.orders.get(out_trade_no)
            if path.endswith("/close"):
                if order is None:
                    return self._json(404, {"code": "ORDER_NOT_EXIST", "message": "x"})
                if order["trade_state"] == "SUCCESS":
                    return self._json(400, {"code": "ORDERPAID", "message": "x"})
                order["trade_state"] = "CLOSED"
                return httpx.Response(204)
            assert parse_qs(urlsplit(str(request.url)).query)["mchid"] == [MCH_ID]
            if order is None:
                return self._json(404, {"code": "ORDER_NOT_EXIST", "message": "x"})
            return self._json(200, self.transaction(out_trade_no))

        if request.method == "POST" and path == "/v3/refund/domestic/refunds":
            order = self.orders[data["out_trade_no"]]
            assert order["trade_state"] == "SUCCESS", "微信不会给未支付订单退款"
            assert data["amount"]["total"] == order["request"]["amount"]["total"]
            refund = self.refunds.setdefault(
                data["out_refund_no"],
                {"refund_id": "50" + secrets.token_hex(8), "status": "PROCESSING", "request": data},
            )
            return self._json(200, self.refund_view(data["out_refund_no"], refund))

        if request.method == "GET" and path.startswith("/v3/refund/domestic/refunds/"):
            out_refund_no = path.rsplit("/", 1)[1]
            return self._json(200, self.refund_view(out_refund_no, self.refunds[out_refund_no]))

        raise AssertionError(f"unexpected wechat call {request.method} {path}")

    def platform_refund(self, out_trade_no: str) -> str:
        """运营在商户平台直接退款：微信侧有退款单，我方没有。"""
        order = self.orders[out_trade_no]
        order["trade_state"] = "REFUND"
        out_refund_no = "PLATFORM" + secrets.token_hex(6).upper()
        total = order["request"]["amount"]["total"]
        self.refunds[out_refund_no] = {
            "refund_id": "50" + secrets.token_hex(8),
            "status": "PROCESSING",
            "request": {"out_trade_no": out_trade_no, "amount": {"refund": total, "total": total}},
        }
        return out_refund_no

    def transaction(self, out_trade_no: str, *, total: int | None = None) -> dict[str, Any]:
        order = self.orders[out_trade_no]
        req = order["request"]
        result = {
            "appid": APP_ID,
            "mchid": MCH_ID,
            "out_trade_no": out_trade_no,
            "trade_state": order["trade_state"],
            "payer": {"openid": req["payer"]["openid"]},
        }
        if order["trade_state"] in {"SUCCESS", "REFUND"}:
            result.update({
                "transaction_id": order["transaction_id"],
                "trade_type": "JSAPI",
                "success_time": order["success_time"],
                "amount": {
                    "total": req["amount"]["total"] if total is None else total,
                    "payer_total": req["amount"]["total"],
                    "currency": "CNY",
                    "payer_currency": "CNY",
                },
            })
        return result

    def refund_view(self, out_refund_no: str, refund: dict[str, Any]) -> dict[str, Any]:
        req = refund["request"]
        view = {
            "refund_id": refund["refund_id"],
            "out_refund_no": out_refund_no,
            "out_trade_no": req["out_trade_no"],
            "status": refund["status"],
            "amount": {"refund": req["amount"]["refund"], "total": req["amount"]["total"]},
        }
        if refund["status"] == "SUCCESS":
            view["success_time"] = datetime.now(CST).replace(microsecond=0).isoformat()
        return view

    # ── 用户在微信里付款 / 退款到账 ─────────────────────────
    def pay(self, out_trade_no: str) -> None:
        order = self.orders[out_trade_no]
        order["trade_state"] = "SUCCESS"
        order["transaction_id"] = "42" + secrets.token_hex(13)
        order["success_time"] = datetime.now(CST).replace(microsecond=0).isoformat()

    def callback(self, event_type: str, resource: dict[str, Any], *, original_type: str) -> tuple[str, dict[str, str]]:
        nonce = secrets.token_hex(6)
        associated = original_type
        ciphertext = AESGCM(API_V3_KEY.encode()).encrypt(
            nonce.encode(), json.dumps(resource).encode(), associated.encode()
        )
        envelope = {
            "id": str(uuid4()),
            "create_time": datetime.now(CST).replace(microsecond=0).isoformat(),
            "resource_type": "encrypt-resource",
            "event_type": event_type,
            "summary": "e2e",
            "resource": {
                "original_type": original_type,
                "algorithm": "AEAD_AES_256_GCM",
                "ciphertext": base64.b64encode(ciphertext).decode(),
                "associated_data": associated,
                "nonce": nonce,
            },
        }
        body = json.dumps(envelope)
        return body, {**self.sign(body), "Content-Type": "application/json"}

    def payment_callback(self, out_trade_no: str, **kw) -> tuple[str, dict[str, str]]:
        return self.callback(
            "TRANSACTION.SUCCESS", self.transaction(out_trade_no, **kw), original_type="transaction"
        )

    def refund_callback(self, out_refund_no: str) -> tuple[str, dict[str, str]]:
        refund = self.refunds[out_refund_no]
        refund["status"] = "SUCCESS"
        view = self.refund_view(out_refund_no, refund)
        view["refund_status"] = view.pop("status")
        return self.callback("REFUND.SUCCESS", view, original_type="refund")


class _HttpxProxy:
    """只替换 transport 模块里的 httpx.AsyncClient，测试自己的 ASGI 客户端不受影响。"""

    def __init__(self, handler):
        self._transport = httpx.MockTransport(handler)

    def __getattr__(self, name):
        return getattr(httpx, name)

    def AsyncClient(self, **kwargs):  # noqa: N802 (与 httpx 同名)
        return httpx.AsyncClient(transport=self._transport, **kwargs)


# ── 夹具 ───────────────────────────────────────────────────────


@dataclass
class World:
    db: Prisma
    http: httpx.AsyncClient
    wechat: FakeWechatPay
    users: list[str]
    merchant_public: rsa.RSAPublicKey

    def auth(self, user_id: str, role: str = "user") -> dict[str, str]:
        from app.services.auth import create_jwt

        return {"Authorization": f"Bearer {create_jwt(user_id, role)}"}

    async def wechat_login(self, path: str, code: str) -> str:
        resp = await self.http.post(path, json={"code": code, "platform": "android"})
        assert resp.status_code == 200, resp.text
        user_id = resp.json()["user_id"]
        if user_id not in self.users:
            self.users.append(user_id)
        return user_id

    async def make_user(self, *, role: str = "user", phone: str | None = None) -> str:
        user = await self.db.user.create(data={"username": f"wxpay-e2e-{uuid4().hex}", "role": role})
        self.users.append(user.id)
        if phone:
            await self.db.authidentity.create(
                data={"user": {"connect": {"id": user.id}}, "provider": "phone",
                      "providerAccountId": phone}
            )
        return user.id

    async def order(self, user_id: str, product_id: str) -> dict[str, Any]:
        resp = await self.http.post(
            "/pay/wechat/orders", json={"product_id": product_id}, headers=self.auth(user_id)
        )
        assert resp.status_code == 200, resp.text
        return resp.json()

    async def notify(self, body: str, headers: dict[str, str]) -> httpx.Response:
        return await self.http.post("/pay/wechat/notifications", content=body, headers=headers)

    async def status(self, user_id: str, out_trade_no: str) -> dict[str, Any]:
        resp = await self.http.get(f"/pay/wechat/orders/{out_trade_no}", headers=self.auth(user_id))
        assert resp.status_code == 200, resp.text
        return resp.json()

    async def row(self, sql: str, *args) -> dict[str, Any]:
        rows = await self.db.query_raw(sql, *args)
        assert rows, sql
        return rows[0]

    async def age_order(self, out_trade_no: str, minutes: int) -> None:
        await self.db.execute_raw(
            "UPDATE wechat_pay_orders SET created_at = created_at - make_interval(mins => $2::int),"
            " updated_at = updated_at - make_interval(mins => $2::int) WHERE out_trade_no = $1",
            out_trade_no, minutes,
        )


def _fake_oauth(prefix: str):
    """code 形如 '<appid标记>:<微信用户名>'：同一微信用户在 App/服务号 openid 不同、unionid 相同。"""
    from app.services.wechat_auth import WeChatTokenPayload

    async def exchange(code: str, **_kw):
        _, person = code.split(":", 1)
        return WeChatTokenPayload(
            openid=f"{prefix}-openid-{person}",
            unionid=f"union-{person}",
            scope="snsapi_userinfo",
            raw={"nickname": person},
        )

    return exchange


@pytest.fixture
async def world(monkeypatch, tmp_path):
    url = os.environ.get("PROACTIVE_E2E_DATABASE_URL", "")
    redis_url = os.environ.get("PROACTIVE_E2E_REDIS_URL", "")
    if not url or not redis_url:
        pytest.skip("WeChat Pay e2e requires isolated PostgreSQL + Redis")
    assert urlsplit(url).hostname in {"localhost", "127.0.0.1"}
    assert urlsplit(url).path == "/companion_proactive_e2e"
    assert urlsplit(redis_url).hostname in {"localhost", "127.0.0.1"}

    import app.db as app_db
    from app import redis_client
    from app.api.admin import wechat_pay as admin_route
    from app.api.public import auth as auth_route
    from app.api.public import wechat_pay as pay_route
    from app.services.payments.wechat import transport

    database = Prisma(datasource={"url": url}, http={"trust_env": False})
    await database.connect()
    # 业务模块都是 `from app.db import db` 绑定的名字：逐个换成连隔离库的客户端。
    # 按类型而不是按身份匹配：上一个用例里才首次导入的模块绑定的是上一个用例的客户端。
    for module in list(sys.modules.values()):
        if (getattr(module, "__name__", "") or "").startswith(("app.", "jobs.")) and isinstance(
            getattr(module, "db", None), Prisma
        ):
            monkeypatch.setattr(module, "db", database)
    monkeypatch.setattr(app_db, "db", database)

    await redis_client.close_redis()
    monkeypatch.setattr(settings, "redis_url", redis_url)

    merchant_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    platform_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    (tmp_path / "apiclient_key.pem").write_bytes(_pem_private(merchant_key))
    (tmp_path / "pub_key.pem").write_bytes(_pem_public(platform_key))
    for name, value in {
        "jwt_secret": "wechatpay-e2e-" + "x" * 40,
        "wechatpay_enabled": True,
        "wechatpay_mch_id": MCH_ID,
        "wechatpay_cert_serial_no": "E2ESERIAL",
        "wechatpay_private_key_path": str(tmp_path / "apiclient_key.pem"),
        "wechatpay_public_key_path": str(tmp_path / "pub_key.pem"),
        "wechatpay_public_key_id": PUB_KEY_ID,
        "wechatpay_api_v3_key": API_V3_KEY,
        "wechatpay_notify_url": NOTIFY_URL,
        "wechat_h5_app_id": APP_ID,
        "wechat_h5_app_secret": "e2e-secret",
    }.items():
        monkeypatch.setattr(settings, name, value)
    transport.reset_key_cache()

    wechat = FakeWechatPay(merchant_key=merchant_key, platform_key=platform_key)
    monkeypatch.setattr(transport, "httpx", _HttpxProxy(wechat.handle))
    monkeypatch.setattr(auth_route, "exchange_wechat_code", _fake_oauth("app"))
    monkeypatch.setattr(auth_route, "exchange_wechat_h5_code", _fake_oauth("oa"))
    monkeypatch.setattr(pay_route, "exchange_wechat_h5_code", _fake_oauth("oa"))

    gate = await database.systemconfig.find_unique(where={"id": 1})
    previous_gate = bool(getattr(gate, "realMoneyPaymentsEnabled", False))
    await database.systemconfig.upsert(
        where={"id": 1},
        data={"create": {"id": 1, "realMoneyPaymentsEnabled": True},
              "update": {"realMoneyPaymentsEnabled": True}},
    )

    app = FastAPI()
    for router in (auth_route.router, pay_route.router, admin_route.router):
        app.include_router(router)
    users: list[str] = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://e2e.test"
    ) as http:
        try:
            yield World(database, http, wechat, users, merchant_key.public_key())
        finally:
            from app.services.runtime.tasks import pending_background_tasks
            import asyncio

            pending = pending_background_tasks()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if users:
                await database.execute_raw(
                    "DELETE FROM wallet_ledger WHERE user_id = ANY($1::text[])", users
                )
                await database.execute_raw("DELETE FROM users WHERE id = ANY($1::text[])", users)
            await database.systemconfig.update(
                where={"id": 1}, data={"realMoneyPaymentsEnabled": previous_gate}
            )
            await redis_client.close_redis()
            transport.reset_key_cache()
            await database.disconnect()


# ── 用例 ───────────────────────────────────────────────────────


def _verify_pay_sign(world: World, params: dict[str, str]) -> None:
    """H5 拿到的 paySign 必须能被微信（商户公钥）验过，否则收银台调不起来。"""
    message = (
        f"{params['appId']}\n{params['timeStamp']}\n{params['nonceStr']}\n{params['package']}\n"
    ).encode()
    world.merchant_public.verify(
        base64.b64decode(params["paySign"]), message, padding.PKCS1v15(), hashes.SHA256()
    )


async def test_app_wechat_account_buys_in_service_account_and_refunds(world: World):
    """主流程：App 微信登录的账号 → 服务号 H5 登录 → 买 VIP/钞票 → 到账 → 退款回收。"""
    app_user = await world.wechat_login("/auth/wechat/mobile", "app:alice")
    h5_user = await world.wechat_login("/auth/wechat/h5", "oa:alice")
    assert h5_user == app_user, "同一微信（unionid）必须落到同一账号"
    assert world.users == [app_user]

    catalog = (await world.http.get("/pay/wechat/catalog", headers=world.auth(app_user))).json()
    assert catalog["enabled"] and catalog["openid_bound"]
    assert [p["price_fen"] for p in catalog["vip_plans"]] == [3900, 8900, 24900]

    # 下单：服务号 openid 下单，JSAPI 参数签名有效。
    vip = await world.order(app_user, "wx.vip.month")
    _verify_pay_sign(world, vip["pay_params"])
    sent = world.wechat.orders[vip["out_trade_no"]]["request"]
    assert sent["payer"]["openid"] == "oa-openid-alice"
    assert sent["amount"] == {"total": 3900, "currency": "CNY"}
    assert (await world.status(app_user, vip["out_trade_no"]))["status"] == "created"

    # 付款 → 签名加密的回调 → 到账。
    world.wechat.pay(vip["out_trade_no"])
    body, headers = world.wechat.payment_callback(vip["out_trade_no"])
    resp = await world.notify(body, headers)
    assert resp.status_code == 200 and resp.json() == {"code": "SUCCESS"}
    polled = await world.status(app_user, vip["out_trade_no"])
    assert polled["status"] == "granted" and polled["vip_is_active"]
    vip_until = datetime.fromisoformat(polled["vip_until"].replace("Z", "+00:00"))
    assert timedelta(days=30) < vip_until - datetime.now(timezone.utc) <= timedelta(days=31)

    # 微信重推同一回调：幂等，不重复发放。
    assert (await world.notify(body, headers)).status_code == 200
    assert (await world.status(app_user, vip["out_trade_no"]))["vip_until"] == polled["vip_until"]

    # 钞票
    tickets = await world.order(app_user, "wx.ticket.80")
    world.wechat.pay(tickets["out_trade_no"])
    assert (await world.notify(*world.wechat.payment_callback(tickets["out_trade_no"]))).status_code == 200
    assert (await world.status(app_user, tickets["out_trade_no"]))["ticket_balance"] == 80

    # App 端看到的是同一个钱包（直接查同一 user_id 的钱包行）。
    wallet = await world.row("SELECT ticket_balance FROM user_wallets WHERE user_id = $1", app_user)
    assert int(wallet["ticket_balance"]) == 800  # subunits ×10

    # 后台全额退款 → 退款回调 → 回收。
    admin = await world.make_user(role="admin")
    for out_trade_no in (tickets["out_trade_no"], vip["out_trade_no"]):
        resp = await world.http.post(
            "/admin-api/wechat-pay/refunds",
            json={"out_trade_no": out_trade_no, "reason": "e2e"},
            headers=world.auth(admin, "admin"),
        )
        assert resp.status_code == 200, resp.text
        refund = resp.json()
        assert refund["status"] == "pending"
        # 双击：回放同一张退款单，不重复出款。
        again = await world.http.post(
            "/admin-api/wechat-pay/refunds", json={"out_trade_no": out_trade_no},
            headers=world.auth(admin, "admin"),
        )
        assert again.json()["out_refund_no"] == refund["out_refund_no"]
        assert len([r for r in world.wechat.refunds.values()
                    if r["request"]["out_trade_no"] == out_trade_no]) == 1
        assert (await world.notify(*world.wechat.refund_callback(refund["out_refund_no"]))).status_code == 200

    listing = await world.http.get(
        "/admin-api/wechat-pay/orders", params={"user_id": app_user},
        headers=world.auth(admin, "admin"),
    )
    assert listing.status_code == 200, listing.text
    items = {item["out_trade_no"]: item for item in listing.json()["items"]}
    item = items[vip["out_trade_no"]]
    assert item["notification_count"] >= 2 and item["last_notification_error"] is None
    assert item["created_at"] and item["success_time"] and item["last_notification_at"]
    forbidden = await world.http.get("/admin-api/wechat-pay/orders", headers=world.auth(app_user))
    assert forbidden.status_code == 403

    final = await world.status(app_user, vip["out_trade_no"])
    assert final["status"] == "refunded"
    assert not final["vip_is_active"] and final["ticket_balance"] == 0
    assert (await world.status(app_user, tickets["out_trade_no"]))["status"] == "refunded"


async def test_phone_account_cannot_pay_with_another_accounts_wechat(world: World):
    """账号不分裂：已属于 A 的微信不能被手机号账号 B 拿去付款；全新微信可以。"""
    owner = await world.wechat_login("/auth/wechat/mobile", "app:bob")
    phone_user = await world.make_user(phone="+8613800001234")

    resp = await world.http.post(
        "/pay/wechat/orders", json={"product_id": "wx.vip.month"}, headers=world.auth(phone_user)
    )
    assert resp.status_code == 409 and resp.json()["detail"] == "wechat_authorization_required"

    resp = await world.http.post(
        "/pay/wechat/authorize", json={"code": "oa:bob"}, headers=world.auth(phone_user)
    )
    assert resp.status_code == 409 and resp.json()["detail"] == "wechat_bound_to_other_account"
    rows = await world.db.query_raw(
        "SELECT user_id FROM wechat_pay_openids WHERE openid = 'oa-openid-bob'"
    )
    assert rows == []

    resp = await world.http.post(
        "/pay/wechat/authorize", json={"code": "oa:carol"}, headers=world.auth(phone_user)
    )
    assert resp.status_code == 200
    order = await world.order(phone_user, "wx.ticket.10")
    assert world.wechat.orders[order["out_trade_no"]]["request"]["payer"]["openid"] == "oa-openid-carol"
    catalog = (await world.http.get("/pay/wechat/catalog", headers=world.auth(phone_user))).json()
    assert "3800001234"[-4:] in (catalog["account_label"] or "")

    # 别人的订单号查不到。
    other = await world.http.get(
        f"/pay/wechat/orders/{order['out_trade_no']}", headers=world.auth(owner)
    )
    assert other.status_code == 404


async def test_forged_and_mismatched_callbacks_never_grant(world: World):
    user = await world.wechat_login("/auth/wechat/h5", "oa:dave")
    order = await world.order(user, "wx.vip.year")
    world.wechat.pay(order["out_trade_no"])

    # 篡改报文：验签失败 401，不落库不发货。
    body, headers = world.wechat.payment_callback(order["out_trade_no"])
    resp = await world.notify(body.replace("e2e", "e2f"), headers)
    assert resp.status_code == 401
    # 换钥匙（Serial 不一致）
    resp = await world.notify(body, {**headers, "Wechatpay-Serial": "PUB_KEY_ID_OTHER"})
    assert resp.status_code == 401
    assert (await world.status(user, order["out_trade_no"]))["status"] == "created"

    # 签名合法但金额被压低：落 failed，不发 VIP。
    resp = await world.notify(*world.wechat.payment_callback(order["out_trade_no"], total=1))
    assert resp.status_code == 200
    polled = await world.status(user, order["out_trade_no"])
    assert polled["status"] == "failed" and not polled["vip_is_active"]
    row = await world.row(
        "SELECT fail_reason FROM wechat_pay_orders WHERE out_trade_no = $1", order["out_trade_no"]
    )
    assert row["fail_reason"] == "amount_mismatch"

    # failed 订单可以全额退款（钱收了没发货）。
    admin = await world.make_user(role="admin")
    resp = await world.http.post(
        "/admin-api/wechat-pay/refunds", json={"out_trade_no": order["out_trade_no"]},
        headers=world.auth(admin, "admin"),
    )
    assert resp.status_code == 200
    assert (await world.notify(*world.wechat.refund_callback(resp.json()["out_refund_no"]))).status_code == 200
    assert (await world.status(user, order["out_trade_no"]))["status"] == "refunded"


async def test_reconcile_recovers_lost_callbacks_and_closes_unpaid(world: World):
    """回调全丢的情况下，对账 cron 靠查单补发；超时未付的关单；退款查单收尾。"""
    from app.services.payments.wechat import reconcile, refunds

    user = await world.wechat_login("/auth/wechat/h5", "oa:erin")
    paid = await world.order(user, "wx.ticket.180")
    unpaid = await world.order(user, "wx.ticket.10")
    world.wechat.pay(paid["out_trade_no"])
    await world.age_order(paid["out_trade_no"], 5)
    await world.age_order(unpaid["out_trade_no"], 45)

    counts = await reconcile.reconcile_pending_orders(limit=500)
    assert counts["settled"] >= 1 and counts["closed"] >= 1
    assert (await world.status(user, paid["out_trade_no"]))["status"] == "granted"
    assert (await world.status(user, unpaid["out_trade_no"]))["status"] == "closed"
    assert world.wechat.orders[unpaid["out_trade_no"]]["trade_state"] == "CLOSED"
    assert (await world.status(user, paid["out_trade_no"]))["ticket_balance"] == 180

    # 用户把钞票花掉一部分后退款：扣到 0 为止，退款不被阻断。
    await world.db.execute_raw(
        "UPDATE user_wallets SET ticket_balance = 500 WHERE user_id = $1", user
    )
    result = await refunds.create_refund(out_trade_no=paid["out_trade_no"], actor_id="e2e")
    world.wechat.refunds[result["out_refund_no"]]["status"] = "SUCCESS"  # 退款回调丢失
    await world.db.execute_raw(
        "UPDATE wechat_pay_refunds SET updated_at = updated_at - interval '5 minutes'"
        " WHERE out_refund_no = $1", result["out_refund_no"],
    )
    refund_counts = await refunds.reconcile_pending_refunds(limit=500)
    assert refund_counts["settled"] >= 1
    polled = await world.status(user, paid["out_trade_no"])
    assert polled["status"] == "refunded" and polled["ticket_balance"] == 0
    ledger = await world.row(
        "SELECT delta FROM wallet_ledger WHERE source = 'wechat_pay_refund' AND source_id = $1",
        result["out_refund_no"],
    )
    assert int(ledger["delta"]) == -500


async def test_trial_is_granted_once_even_if_two_orders_are_paid(world: World):
    user = await world.wechat_login("/auth/wechat/h5", "oa:frank")
    first = await world.order(user, "wx.vip.trial")
    second = await world.order(user, "wx.vip.trial")  # 都没付时允许两单
    for order in (first, second):
        world.wechat.pay(order["out_trade_no"])
        assert (await world.notify(*world.wechat.payment_callback(order["out_trade_no"]))).status_code == 200

    assert (await world.status(user, first["out_trade_no"]))["status"] == "granted"
    second_row = await world.row(
        "SELECT status, fail_reason FROM wechat_pay_orders WHERE out_trade_no = $1",
        second["out_trade_no"],
    )
    assert (second_row["status"], second_row["fail_reason"]) == ("failed", "trial_already_used")
    resp = await world.http.post(
        "/pay/wechat/orders", json={"product_id": "wx.vip.trial"}, headers=world.auth(user)
    )
    assert resp.status_code == 409 and resp.json()["detail"] == "trial_already_used"


async def test_wechat_vip_stacks_after_apple_vip(world: World):
    """Apple 已买月卡的账号在服务号再买季卡：天数接在 Apple 后面，互不覆盖。"""
    user = await world.wechat_login("/auth/wechat/mobile", "app:grace")
    await world.wechat_login("/auth/wechat/h5", "oa:grace")
    apple_at = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=1)
    await world.db.execute_raw(
        """
        INSERT INTO iap_transactions (transaction_id, product_id, kind, environment, user_id,
                                      purchase_date, status)
        VALUES ($1, 'com.bansheng.vip.month', 'consumable', 'Sandbox', $2, $3::timestamp, 'granted')
        """,
        f"e2e-{uuid4().hex}", user, apple_at.replace(tzinfo=None),
    )
    order = await world.order(user, "wx.vip.quarter")
    world.wechat.pay(order["out_trade_no"])
    assert (await world.notify(*world.wechat.payment_callback(order["out_trade_no"]))).status_code == 200

    polled = await world.status(user, order["out_trade_no"])
    vip_until = datetime.fromisoformat(polled["vip_until"].replace("Z", "+00:00"))
    assert abs(vip_until - (apple_at + timedelta(days=31 + 93))) < timedelta(minutes=1)


async def test_phone_account_authorizing_wechat_then_app_login_is_same_account(world: World):
    """手机号账号在服务号授权并付款后，用同一微信在 App 登录必须回到这个账号。"""
    phone_user = await world.make_user(phone="+8613900005678")
    resp = await world.http.post(
        "/pay/wechat/authorize", json={"code": "oa:henry"}, headers=world.auth(phone_user)
    )
    assert resp.status_code == 200, resp.text
    order = await world.order(phone_user, "wx.vip.month")
    world.wechat.pay(order["out_trade_no"])
    assert (await world.notify(*world.wechat.payment_callback(order["out_trade_no"]))).status_code == 200

    app_user = await world.wechat_login("/auth/wechat/mobile", "app:henry")
    assert app_user == phone_user, "App 微信登录不能新建一个看不到 VIP 的空账号"
    me = await world.status(app_user, order["out_trade_no"])
    assert me["vip_is_active"]


async def test_platform_refund_with_lost_callback_never_grants(world: World):
    """支付回调丢了，运营又在商户平台退了款：对账查到 REFUND，不能再发货。"""
    from app.services.payments.wechat import reconcile

    user = await world.wechat_login("/auth/wechat/h5", "oa:ivy")
    order = await world.order(user, "wx.ticket.300")
    world.wechat.pay(order["out_trade_no"])
    world.wechat.platform_refund(order["out_trade_no"])
    await world.age_order(order["out_trade_no"], 5)

    await reconcile.reconcile_pending_orders(limit=500)

    polled = await world.status(user, order["out_trade_no"])
    assert polled["status"] == "refunded" and polled["ticket_balance"] == 0
    row = await world.row(
        "SELECT fail_reason FROM wechat_pay_orders WHERE out_trade_no = $1", order["out_trade_no"]
    )
    assert row["fail_reason"] == "refunded_before_grant"


async def test_platform_refund_while_our_refund_submit_was_lost(world: World):
    """我方退款提交超时（pending 无 refund_id），运营改去商户平台退：回调要能收回权益。"""
    user = await world.wechat_login("/auth/wechat/h5", "oa:jack")
    order = await world.order(user, "wx.vip.month")
    world.wechat.pay(order["out_trade_no"])
    assert (await world.notify(*world.wechat.payment_callback(order["out_trade_no"]))).status_code == 200
    assert (await world.status(user, order["out_trade_no"]))["vip_is_active"]

    admin = await world.make_user(role="admin")
    world.wechat.fail_next_refund_submit = True
    resp = await world.http.post(
        "/admin-api/wechat-pay/refunds", json={"out_trade_no": order["out_trade_no"]},
        headers=world.auth(admin, "admin"),
    )
    assert resp.status_code == 502
    ours = await world.row(
        "SELECT out_refund_no, status, refund_id FROM wechat_pay_refunds WHERE out_trade_no = $1",
        order["out_trade_no"],
    )
    assert ours["status"] == "pending" and ours["refund_id"] is None

    platform_no = world.wechat.platform_refund(order["out_trade_no"])
    resp = await world.notify(*world.wechat.refund_callback(platform_no))
    assert resp.status_code == 200, resp.text

    polled = await world.status(user, order["out_trade_no"])
    assert polled["status"] == "refunded" and not polled["vip_is_active"]
    superseded = await world.row(
        "SELECT status FROM wechat_pay_refunds WHERE out_refund_no = $1", ours["out_refund_no"]
    )
    assert superseded["status"] == "closed"
