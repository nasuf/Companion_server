"""服务号 openid 绑定与配置校验：账号不分裂的那条链路。

用户关心的场景：App 微信登录的账号，从服务号 H5 付款，钱必须落在同一账号上。
实现上这由三件事保证，这里逐条测：
1. openid 单独存，不覆盖 `auth_identities`（否则 App 登录会被影响）。
2. 查 openid 时限定 app_id，不会把移动应用的 openid 拿去 JSAPI 下单。
3. 没有 openid 时明确报错要求重新授权，而不是静默用错的身份下单。
"""

import pytest

from app.config import settings
from app.services.payments.wechat import accounts
from app.services.payments.wechat.errors import WechatPayOpenidMissingError


class _FakeDb:
    def __init__(self, rows=()):
        self._rows = list(rows)
        self.queries: list[tuple[str, tuple]] = []
        self.executes: list[tuple[str, tuple]] = []

    async def query_raw(self, sql, *args):
        self.queries.append((sql, args))
        return self._rows.pop(0) if self._rows else []

    async def execute_raw(self, sql, *args):
        self.executes.append((sql, args))
        return 1


async def test_bind_openid_writes_own_table_only(monkeypatch):
    """绑定只写 wechat_pay_openids：绝不 UPDATE auth_identities。"""
    fake = _FakeDb()
    monkeypatch.setattr(accounts, "db", fake)

    await accounts.bind_openid(
        user_id="u1", openid="o-h5", app_id="wx-oa", unionid="union-1"
    )

    sql, args = fake.executes[0]
    assert "INSERT INTO wechat_pay_openids" in sql
    assert "auth_identities" not in sql
    assert args == ("o-h5", "wx-oa", "u1", "union-1")


async def test_rebinding_same_openid_updates_user(monkeypatch):
    """同一微信号换绑到别的账号：以最后一次授权为准（历史订单不受影响）。"""
    fake = _FakeDb()
    monkeypatch.setattr(accounts, "db", fake)

    await accounts.bind_openid(user_id="u2", openid="o-h5", app_id="wx-oa")

    sql, _ = fake.executes[0]
    assert "ON CONFLICT (openid) DO UPDATE" in sql
    assert "user_id = EXCLUDED.user_id" in sql


async def test_bind_preserves_existing_unionid_when_absent(monkeypatch):
    """这次授权没拿到 unionid 时不要把已存的覆盖成 NULL。"""
    fake = _FakeDb()
    monkeypatch.setattr(accounts, "db", fake)

    await accounts.bind_openid(user_id="u1", openid="o-h5", app_id="wx-oa")

    sql, _ = fake.executes[0]
    assert "COALESCE(EXCLUDED.unionid, wechat_pay_openids.unionid)" in sql


async def test_openid_lookup_is_scoped_to_app_id(monkeypatch):
    """必须按 app_id 过滤：移动应用的 openid 在服务号下无效。"""
    fake = _FakeDb(rows=[[{"openid": "o-h5"}]])
    monkeypatch.setattr(accounts, "db", fake)

    openid = await accounts.openid_for_user("u1", app_id="wx-oa")

    assert openid == "o-h5"
    sql, args = fake.queries[0]
    assert "app_id = $2" in sql
    assert args == ("u1", "wx-oa")


async def test_missing_openid_raises_instead_of_guessing(monkeypatch):
    """没有 openid 就报错让前端重新授权，不能拿别的身份凑一个。"""
    monkeypatch.setattr(accounts, "db", _FakeDb(rows=[[]]))

    with pytest.raises(WechatPayOpenidMissingError):
        await accounts.openid_for_user("u1", app_id="wx-oa")


@pytest.mark.parametrize("bad", [None, "", 123])
async def test_blank_openid_row_treated_as_missing(monkeypatch, bad):
    monkeypatch.setattr(accounts, "db", _FakeDb(rows=[[{"openid": bad}]]))

    with pytest.raises(WechatPayOpenidMissingError):
        await accounts.openid_for_user("u1", app_id="wx-oa")


async def test_lookup_takes_most_recent_authorization(monkeypatch):
    fake = _FakeDb(rows=[[{"openid": "o-new"}]])
    monkeypatch.setattr(accounts, "db", fake)
    await accounts.openid_for_user("u1", app_id="wx-oa")
    sql, _ = fake.queries[0]
    assert "ORDER BY updated_at DESC" in sql


# ── 生产配置自检 ──────────────────────────────────────────────


def _production_settings(**overrides):
    """构造一份打开微信支付的生产配置。"""
    from app.config import Settings

    base = {
        "app_env": "production",
        "jwt_secret": "x" * 48,
        "cors_allowed_origins": "https://banshengcomp.com",
        # 隔离出微信支付校验分支：Apple 的校验由 test_iap_config.py 覆盖，
        # 且本地 .env 可能把它打开，会先抛 APPLE_IAP 的错盖住要测的分支。
        "apple_iap_enabled": False,
        "wechatpay_enabled": True,
        "wechatpay_mch_id": "1007054057",
        "wechatpay_cert_serial_no": "ABC123",
        "wechatpay_public_key_id": "PUB_KEY_ID_0117533412802026112500291836001403",
        "wechatpay_private_key_path": "/app/companion-secrets/wechatpay/apiclient_key.pem",
        "wechatpay_public_key_path": "/app/companion-secrets/wechatpay/pub_key.pem",
        "wechatpay_api_v3_key": "0" * 32,
        "wechatpay_notify_url": "https://banshengcomp.com/api/pay/wechat/notifications",
        "wechat_h5_app_id": "wxfe66dc37ba1bc621",
    }
    base.update(overrides)
    return Settings, base


@pytest.mark.parametrize(
    "field",
    [
        "wechatpay_mch_id",
        "wechatpay_cert_serial_no",
        "wechatpay_private_key_path",
        "wechatpay_public_key_path",
        "wechat_h5_app_id",
    ],
)
def test_production_requires_each_credential(field):
    """缺任一凭证都必须启动失败，而不是上线后每笔支付报错。"""
    Settings, kwargs = _production_settings(**{field: ""})
    instance = Settings(**kwargs)
    with pytest.raises(RuntimeError, match="WECHATPAY_ENABLED"):
        instance.validate_security_config()


@pytest.mark.parametrize("key", ["", "short", "a" * 31, "a" * 33])
def test_production_requires_32_byte_api_v3_key(key):
    """长度不对会让每笔回调都解不开 = 静默丢单，必须启动时拦住。"""
    Settings, kwargs = _production_settings(wechatpay_api_v3_key=key)
    instance = Settings(**kwargs)
    with pytest.raises(RuntimeError, match="WECHATPAY_API_V3_KEY"):
        instance.validate_security_config()


@pytest.mark.parametrize(
    "url",
    ["", "http://banshengcomp.com/api/pay/wechat/notifications", "ftp://x/y"],
)
def test_production_requires_https_notify_url(url):
    """回调必须 https：微信按 https 校验，http 地址收不到回调。"""
    Settings, kwargs = _production_settings(wechatpay_notify_url=url)
    instance = Settings(**kwargs)
    with pytest.raises(RuntimeError, match="WECHATPAY_NOTIFY_URL"):
        instance.validate_security_config()


@pytest.mark.parametrize(
    "url",
    [
        "https://banshengcomp.com/api/pay/wechat/notify",
        "https://banshengcomp.com/",
    ],
)
def test_production_requires_notify_url_pointing_at_route(url):
    """path 写错时回调全部 404，必须在启动时拦住。"""
    Settings, kwargs = _production_settings(wechatpay_notify_url=url)
    with pytest.raises(RuntimeError, match="WECHATPAY_NOTIFY_URL"):
        Settings(**kwargs).validate_security_config()


@pytest.mark.parametrize("key_id", ["", "66B7169FC57DB721FF5E2B71EED41BC2FD1BC5DB"])
def test_production_requires_public_key_id(key_id):
    """公钥 ID 缺失或误填成证书序列号都要拦住。"""
    Settings, kwargs = _production_settings(wechatpay_public_key_id=key_id)
    with pytest.raises(RuntimeError, match="WECHATPAY_PUBLIC_KEY_ID"):
        Settings(**kwargs).validate_security_config()


def test_production_passes_with_complete_config():
    Settings, kwargs = _production_settings()
    Settings(**kwargs).validate_security_config()


def test_disabled_wechatpay_skips_validation():
    """未启用时不校验，便于只跑 Apple 的部署保持原样。"""
    Settings, kwargs = _production_settings(
        wechatpay_enabled=False, wechatpay_mch_id="", wechatpay_api_v3_key=""
    )
    Settings(**kwargs).validate_security_config()


def test_notify_path_matches_route(monkeypatch):
    """配置的回调地址 path 必须与实际路由一致，否则回调 404。"""
    from app.api.public.wechat_pay import NOTIFY_ROUTE_PATH
    from app.services.payments.wechat import transport

    monkeypatch.setattr(
        settings,
        "wechatpay_notify_url",
        f"https://banshengcomp.com/api{NOTIFY_ROUTE_PATH}",
    )
    # 线上 nginx 把 /api 前缀剥掉后转给应用（见 /api/health 与 /health 均可用）。
    assert transport.notify_path() == f"/api{NOTIFY_ROUTE_PATH}"


# ── 付款授权：已登录账号登记 openid 前的冲突检测 ─────────────────


async def test_conflict_when_unionid_belongs_to_other_account(monkeypatch):
    """手机号账号 B 去授权，但这个微信是账号 A 的登录方式：必须拦下。"""
    fake = _FakeDb(rows=[[{"user_id": "user-A"}]])
    monkeypatch.setattr(accounts, "db", fake)

    owner = await accounts.conflicting_owner(
        user_id="user-B", openid="o-h5", unionid="union-A"
    )

    assert owner == "user-A"
    sql, args = fake.queries[0]
    assert "unionid = $1" in sql and "user_id <> $2" in sql
    assert args == ("union-A", "user-B")


async def test_conflict_falls_back_to_openid_without_unionid(monkeypatch):
    fake = _FakeDb(rows=[[{"user_id": "user-A"}]])
    monkeypatch.setattr(accounts, "db", fake)

    owner = await accounts.conflicting_owner(
        user_id="user-B", openid="o-h5", unionid=None
    )

    assert owner == "user-A"
    sql, _ = fake.queries[0]
    assert "auth_identities" in sql and "wechat_pay_openids" in sql


async def test_no_conflict_for_same_account(monkeypatch):
    fake = _FakeDb(rows=[[], []])
    monkeypatch.setattr(accounts, "db", fake)

    assert (
        await accounts.conflicting_owner(
            user_id="user-A", openid="o-h5", unionid="union-A"
        )
        is None
    )


def _authorize_client(monkeypatch, *, owner, unionid="union-A", identity_conflict=False):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.jwt_auth import require_user
    from app.api.public import wechat_pay as route
    from app.services.wechat_auth import WeChatTokenPayload

    bound = []

    async def fake_exchange(code):
        return WeChatTokenPayload(
            openid="o-h5", unionid=unionid, scope="snsapi_base", raw={}
        )

    async def fake_owner(**kwargs):
        return owner

    async def fake_bind(**kwargs):
        bound.append(kwargs)

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(route, "exchange_wechat_h5_code", fake_exchange)
    monkeypatch.setattr(route.accounts, "conflicting_owner", fake_owner)
    monkeypatch.setattr(route.accounts, "bind_openid", fake_bind)
    identities = []

    async def fake_bind_identity(user_id, token):
        if identity_conflict:
            from app.services.phone_auth import IdentityConflict

            raise IdentityConflict("wechat_taken")
        identities.append((user_id, token.unionid))

    monkeypatch.setattr(route, "bind_wechat_to_user", fake_bind_identity)
    for name in ("enforce_login_rate_limit", "record_login_failure", "clear_login_failures"):
        monkeypatch.setattr(route, name, noop)
    monkeypatch.setattr(settings, "wechat_h5_app_id", "wx-oa")

    app = FastAPI()
    app.include_router(route.router)
    app.dependency_overrides[require_user] = lambda: {"sub": "user-B"}
    client = TestClient(app)
    client.identities = identities
    return client, bound


def test_authorize_rejects_wechat_of_another_account(monkeypatch):
    client, bound = _authorize_client(monkeypatch, owner="user-A")

    resp = client.post("/pay/wechat/authorize", json={"code": "c1"})

    assert resp.status_code == 409
    assert resp.json()["detail"] == "wechat_bound_to_other_account"
    assert bound == []  # 冲突时绝不把 openid 记到当前账号


def test_authorize_binds_to_jwt_user(monkeypatch):
    client, bound = _authorize_client(monkeypatch, owner=None)

    resp = client.post("/pay/wechat/authorize", json={"code": "c1"})

    assert resp.status_code == 200
    assert resp.json() == {"openid_bound": True}
    assert bound == [
        {"user_id": "user-B", "openid": "o-h5", "app_id": "wx-oa", "unionid": "union-A"}
    ]


def test_authorize_requires_unionid(monkeypatch):
    """拿不到 unionid 就无法判断这个微信是不是 App 里另一个账号：fail closed。"""
    client, bound = _authorize_client(monkeypatch, owner=None, unionid=None)

    resp = client.post("/pay/wechat/authorize", json={"code": "c1"})

    assert resp.status_code == 409
    assert resp.json()["detail"] == "wechat_unionid_unavailable"
    assert bound == []


def test_authorize_also_binds_wechat_as_login_identity(monkeypatch):
    """只记付款 openid 不够：用户之后在 App 用同一微信登录会被新建成另一个账号。"""
    client, bound = _authorize_client(monkeypatch, owner=None)

    resp = client.post("/pay/wechat/authorize", json={"code": "c1"})

    assert resp.status_code == 200
    assert client.identities == [("user-B", "union-A")]


def test_authorize_identity_conflict_does_not_record_openid(monkeypatch):
    client, bound = _authorize_client(monkeypatch, owner=None, identity_conflict=True)

    resp = client.post("/pay/wechat/authorize", json={"code": "c1"})

    assert resp.status_code == 409
    assert resp.json()["detail"] == "wechat_bound_to_other_account"
    assert bound == []
