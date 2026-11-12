"""实名门槛: 证件号校验、年龄、档位、核验流程与各入口拦截 (services/identity/realname.py)。"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.services.identity import realname

# 合成证件号 (校验位按 GB 11643 计算, 不对应真实个人)
ADULT_ID = "11010519491231002X"


def _with_check(first17: str) -> str:
    total = sum(int(d) * w for d, w in zip(first17, realname._WEIGHTS))
    return first17 + realname._CHECK[total % 11]


def _born(day: date) -> str:
    return _with_check(f"110105{day:%Y%m%d}123")


class FakeRedis:
    def __init__(self):
        self.store: dict[str, int] = {}

    async def incr(self, key):
        self.store[key] = self.store.get(key, 0) + 1
        return self.store[key]

    async def expire(self, key, ttl):
        return True


@pytest.fixture
def settings(monkeypatch):
    from app.config import settings as s

    monkeypatch.setattr(s, "realname_mode", "id_card")
    monkeypatch.setattr(s, "realname_mock_enabled", True)
    monkeypatch.setattr(s, "realname_hash_key", "")
    monkeypatch.setattr(s, "realname_daily_attempts", 5)
    monkeypatch.setattr(s, "app_env", "development")
    return s


@pytest.fixture
def store(monkeypatch, settings):
    """In-memory user_identity_verifications + phone identities."""
    rows: dict[str, dict] = {}
    phones: set[str] = {"u1", "u2"}
    redis = FakeRedis()

    async def query_raw(sql, *args):
        if "SELECT status FROM user_identity_verifications" in sql:
            row = rows.get(args[0])
            return [{"status": row["status"]}] if row else []
        if "WHERE id_digest=$1" in sql:
            return [{"1": 1} for uid, r in rows.items()
                    if r["id_digest"] == args[0] and uid != args[1]
                    and r["status"] in ("verified", "minor")]
        raise AssertionError(sql)

    async def execute_raw(sql, *args):
        if sql.startswith("DELETE"):
            return 1 if rows.pop(args[0], None) else 0
        (_id, uid, status, masked, digest, adult, provider, reason,
         birth, gender, region, province, name_enc, id_enc) = args
        prev = rows.get(uid, {}).get("attempts", 0)
        rows[uid] = dict(status=status, masked_name=masked, id_digest=digest, is_adult=adult,
                         provider=provider, failure_reason=reason, attempts=prev + 1,
                         birth_date=birth, gender=gender, region_code=region, province=province,
                         name_encrypted=name_enc, id_number_encrypted=id_enc)
        return 1

    async def find_first(*, where):
        return SimpleNamespace(id="p") if where["userId"] in phones else None

    async def find_user(*, where):
        return SimpleNamespace(role="admin" if where["id"] == "admin" else "user")

    db = SimpleNamespace(query_raw=query_raw, execute_raw=execute_raw,
                         authidentity=SimpleNamespace(find_first=find_first),
                         user=SimpleNamespace(find_unique=find_user))
    monkeypatch.setattr(realname, "db", db)
    monkeypatch.setattr("app.redis_client.get_redis", AsyncMock(return_value=redis))
    monkeypatch.setattr(realname, "_today", lambda: date(2026, 11, 12))
    return SimpleNamespace(rows=rows, phones=phones, redis=redis)


def test_id_number_validation():
    assert realname.normalize_id_number(ADULT_ID.lower()) == ADULT_ID
    assert realname.normalize_id_number(ADULT_ID[:-1] + "1") is None  # 校验位错
    assert realname.normalize_id_number(_with_check("11010520250230123")) is None  # 2 月 30 日
    assert realname.normalize_id_number("1101051949123100") is None
    assert realname.normalize_name(" 张 三 ") == "张三"
    assert realname.normalize_name("Zhang") is None
    assert realname.normalize_name("阿依·买买提") == "阿依·买买提"
    assert realname.mask_name("张三丰") == "张**"


def test_age_boundary_is_birthday_inclusive():
    today = date(2026, 11, 12)
    assert realname.age_on(_born(date(2008, 11, 12)), today) == 18
    assert realname.age_on(_born(date(2008, 11, 13)), today) == 17


def test_required_steps_per_mode():
    assert realname.RealnameState("off", False, None).required == []
    assert realname.RealnameState("phone", False, None).required == ["phone"]
    assert realname.RealnameState("phone", True, None).required == []
    assert realname.RealnameState("id_card", False, None).required == ["phone", "id_card"]
    assert realname.RealnameState("id_card", True, "rejected").required == ["id_card"]
    assert realname.RealnameState("id_card", True, "verified").required == []
    assert realname.RealnameState("id_card", True, "minor").required == ["minor_blocked"]


@pytest.mark.asyncio
async def test_admin_and_off_mode_never_blocked(store, settings, monkeypatch):
    assert await realname.chat_block_reason("nobody", role="admin") is None
    # 后台任务不带角色时自己查: 管理员照样豁免
    assert await realname.chat_block_reason("admin") is None
    assert await realname.chat_block_reason("nobody") == "phone"
    monkeypatch.setattr(settings, "realname_mode", "off")
    assert await realname.chat_block_reason("nobody") is None
    monkeypatch.setattr(settings, "realname_mode", "bogus")
    assert realname.mode() == "off"


@pytest.mark.asyncio
async def test_verify_adult_then_gate_opens(store):
    assert await realname.chat_block_reason("u1") == "id_card"
    assert await realname.verify("u1", name="张三", id_number=ADULT_ID) == "verified"
    row = store.rows["u1"]
    assert row["masked_name"] == "张*" and row["is_adult"] is True
    assert ADULT_ID not in str(row)  # 证件号原文不落库
    assert await realname.chat_block_reason("u1") is None
    # 已通过再提交: 不重复核验
    assert await realname.verify("u1", name="李四", id_number=ADULT_ID) == "verified"
    assert store.rows["u1"]["attempts"] == 1


@pytest.mark.asyncio
async def test_minor_is_locked_and_never_sent_to_provider(store, monkeypatch, settings):
    monkeypatch.setattr(settings, "realname_mock_enabled", False)
    monkeypatch.setattr(settings, "tencent_realname_secret_id", "id")
    monkeypatch.setattr(settings, "tencent_realname_secret_key", "key")
    provider = AsyncMock(return_value=True)
    monkeypatch.setattr(realname, "_tencent_check", provider)
    minor = _born(date(2010, 5, 1))
    assert await realname.verify("u1", name="张三", id_number=minor) == "minor"
    provider.assert_not_awaited()
    assert await realname.chat_block_reason("u1") == "minor_blocked"
    # 锁定后换成年人证件也不能解锁
    assert await realname.verify("u1", name="张三", id_number=ADULT_ID) == "minor"
    assert store.rows["u1"]["status"] == "minor"
    # 只有管理员重置才能重来
    assert await realname.admin_reset("u1") == 1
    assert await realname.chat_block_reason("u1") == "id_card"


@pytest.mark.asyncio
async def test_one_id_binds_one_account(store):
    await realname.verify("u1", name="张三", id_number=ADULT_ID)
    with pytest.raises(realname.RealnameError, match="id_in_use"):
        await realname.verify("u2", name="张三", id_number=ADULT_ID)
    assert store.rows["u2"]["status"] == "rejected" and store.rows["u2"]["id_digest"] is None
    # 家长证件被未成年账号先占用后, 也不能给另一个账号用
    minor = _born(date(2012, 1, 1))
    store.phones.add("u3")
    store.phones.add("u4")
    await realname.verify("u3", name="王五", id_number=minor)
    with pytest.raises(realname.RealnameError, match="id_in_use"):
        await realname.verify("u4", name="王五", id_number=minor)


@pytest.mark.asyncio
async def test_mismatch_and_input_errors(store, monkeypatch, settings):
    monkeypatch.setattr(settings, "realname_mock_enabled", False)
    monkeypatch.setattr(settings, "tencent_realname_secret_id", "id")
    monkeypatch.setattr(settings, "tencent_realname_secret_key", "key")
    monkeypatch.setattr(realname, "_tencent_check", AsyncMock(return_value=False))
    with pytest.raises(realname.RealnameError, match="mismatch"):
        await realname.verify("u1", name="张三", id_number=ADULT_ID)
    assert store.rows["u1"] == dict(store.rows["u1"], status="rejected", id_digest=None,
                                    failure_reason="mismatch")
    with pytest.raises(realname.RealnameError, match="invalid_id_number"):
        await realname.verify("u1", name="张三", id_number=ADULT_ID[:-1] + "1")
    with pytest.raises(realname.RealnameError, match="invalid_name"):
        await realname.verify("u1", name="Zhang", id_number=ADULT_ID)


@pytest.mark.asyncio
async def test_phone_required_first_and_daily_limit(store, settings, monkeypatch):
    store.phones.discard("u1")
    with pytest.raises(realname.RealnameError, match="phone_required"):
        await realname.verify("u1", name="张三", id_number=ADULT_ID)
    store.phones.add("u1")
    monkeypatch.setattr(settings, "realname_daily_attempts", 2)
    monkeypatch.setattr(realname, "_mock_enabled", lambda: False)
    monkeypatch.setattr(settings, "tencent_realname_secret_id", "id")
    monkeypatch.setattr(settings, "tencent_realname_secret_key", "key")
    monkeypatch.setattr(realname, "_tencent_check", AsyncMock(return_value=False))
    for _ in range(2):
        with pytest.raises(realname.RealnameError, match="mismatch"):
            await realname.verify("u1", name="张三", id_number=ADULT_ID)
    with pytest.raises(realname.RealnameError, match="rate_limited"):
        await realname.verify("u1", name="张三", id_number=ADULT_ID)


@pytest.mark.asyncio
async def test_production_never_uses_mock_and_needs_credentials(store, settings, monkeypatch):
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "tencent_realname_secret_id", "")
    with pytest.raises(realname.RealnameError, match="not_configured"):
        await realname.verify("u1", name="张三", id_number=ADULT_ID)
    monkeypatch.setattr(settings, "realname_mode", "phone")
    with pytest.raises(realname.RealnameError, match="not_required"):
        await realname.verify("u1", name="张三", id_number=ADULT_ID)


@pytest.mark.asyncio
async def test_tencent_result_mapping(monkeypatch, settings):
    import httpx

    calls = []

    def handler(request):
        calls.append(request)
        result = "0" if b"11010519491231002X" in request.content else "-1"
        return httpx.Response(200, json={"Response": {"Result": result, "RequestId": "r"}})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(realname.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(settings, "tencent_realname_secret_id", "AKID")
    monkeypatch.setattr(settings, "tencent_realname_secret_key", "secret")
    assert await realname._tencent_check("张三", ADULT_ID) is True
    assert await realname._tencent_check("张三", _born(date(1990, 1, 1))) is False
    req = calls[0]
    assert req.headers["X-TC-Action"] == "IdCardVerification"
    assert req.headers["Host"] == "faceid.tencentcloudapi.com"
    assert "Credential=AKID/" in req.headers["Authorization"] and "/faceid/tc3_request" in req.headers["Authorization"]

    def failing(request):
        return httpx.Response(200, json={"Response": {"Error": {"Code": "AuthFailure"}}})

    monkeypatch.setattr(realname.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(failing)))
    with pytest.raises(realname.RealnameError, match="provider_unavailable"):
        await realname._tencent_check("张三", ADULT_ID)


def test_production_config_validation(monkeypatch):
    from app.config import Settings

    base = dict(app_env="production", jwt_secret="x" * 40, cors_allowed_origins="https://a.example",
                wechat_login_enabled=False, apple_iap_enabled=False, sms_enabled=True)
    Settings(**base, realname_mode="off").validate_security_config()
    with pytest.raises(RuntimeError, match="REALNAME_MODE must be"):
        Settings(**base, realname_mode="strict").validate_security_config()
    with pytest.raises(RuntimeError, match="requires SMS_ENABLED"):
        Settings(**{**base, "sms_enabled": False}, realname_mode="phone").validate_security_config()
    with pytest.raises(RuntimeError, match="TENCENT_REALNAME_SECRET_ID"):
        Settings(**base, realname_mode="id_card", realname_hash_key="k" * 40).validate_security_config()
    with pytest.raises(RuntimeError, match="at least 32"):
        Settings(**base, realname_mode="id_card", realname_hash_key="short",
                 tencent_realname_secret_id="a", tencent_realname_secret_key="b").validate_security_config()
    Settings(**base, realname_mode="id_card", realname_hash_key="k" * 40,
             tencent_realname_secret_id="a", tencent_realname_secret_key="b").validate_security_config()


@pytest.mark.asyncio
async def test_chat_entry_points_reject_with_step(monkeypatch):
    from app.api.public import chat

    monkeypatch.setattr(realname, "chat_block_reason", AsyncMock(return_value="phone"))
    with pytest.raises(HTTPException) as exc:
        await chat.require_realname({"sub": "u1", "role": "user"})
    assert exc.value.status_code == 403
    assert exc.value.detail == {"reason": "realname_required", "step": "phone",
                                "message": "请先完成实名认证"}
    monkeypatch.setattr(realname, "chat_block_reason", AsyncMock(return_value=None))
    await chat.require_realname({"sub": "u1", "role": "user"})


@pytest.mark.asyncio
async def test_proactive_gate_stops_unverified_users(monkeypatch):
    from app.services.proactive import gates

    monkeypatch.setattr(gates.db, "query_raw", AsyncMock(return_value=[{"status": "active"}]))
    state = SimpleNamespace(workspace_id="w", user_id="u1")
    monkeypatch.setattr(realname, "chat_block_reason", AsyncMock(return_value="id_card"))
    gate = await gates.check_workspace(state)
    assert (gate.action, gate.reason) == ("stop", "realname_required")
    monkeypatch.setattr(realname, "chat_block_reason", AsyncMock(return_value=None))
    assert await gates.check_workspace(state) is None


@pytest.mark.asyncio
async def test_concurrent_claim_of_same_id_maps_to_in_use(store, monkeypatch):
    """两个账号同时提交同一张证: 唯一索引让后到的写入失败, 转成 id_in_use 而不是 500。"""
    original = realname._record
    calls = []

    async def racing(user_id, **kw):
        calls.append(kw["status"])
        if kw["status"] == "verified":
            raise RuntimeError('duplicate key value violates unique constraint "user_identity_verifications_digest_uq"')
        return await original(user_id, **kw)

    monkeypatch.setattr(realname, "_record", racing)
    with pytest.raises(realname.RealnameError, match="id_in_use"):
        await realname.verify("u1", name="张三", id_number=ADULT_ID)
    assert calls == ["verified", "rejected"]
    assert store.rows["u1"]["failure_reason"] == "id_in_use"


@pytest.mark.asyncio
async def test_verified_profile_is_stored_and_identity_is_encrypted(store):
    assert await realname.verify("u1", name="张三", id_number=ADULT_ID) == "verified"
    row = store.rows["u1"]
    assert (row["birth_date"], row["gender"], row["region_code"], row["province"]) == (
        "1949-12-31", "female", "110105", "北京")
    # 姓名和证件号只有密文, 能用同一把派生密钥解开
    assert "张三" not in row["name_encrypted"] and ADULT_ID not in row["id_number_encrypted"]
    assert realname.decrypt(row["name_encrypted"]) == "张三"
    assert realname.decrypt(row["id_number_encrypted"]) == ADULT_ID


@pytest.mark.asyncio
async def test_rejected_attempt_stores_no_identity(store, monkeypatch, settings):
    monkeypatch.setattr(realname, "_mock_enabled", lambda: False)
    monkeypatch.setattr(settings, "tencent_realname_secret_id", "id")
    monkeypatch.setattr(settings, "tencent_realname_secret_key", "key")
    monkeypatch.setattr(realname, "_tencent_check", AsyncMock(return_value=False))
    with pytest.raises(realname.RealnameError, match="mismatch"):
        await realname.verify("u1", name="张三", id_number=ADULT_ID)
    row = store.rows["u1"]
    assert row["name_encrypted"] is None and row["id_number_encrypted"] is None
    assert row["birth_date"] is None and row["gender"] is None


def test_id_profile_gender_and_province():
    male = _with_check("44030619900101123")  # 第 17 位 3 → 男
    info = realname.profile_of(male)
    assert (info.gender, info.province, info.region_code) == ("male", "广东", "440306")
    assert realname.profile_of(_with_check("99000019900101124")).province is None


def test_ciphertext_is_randomized_and_tamper_evident():
    a, b = realname.encrypt("张三"), realname.encrypt("张三")
    assert a != b and a.startswith("v1:")
    broken = a[:-4] + ("AAAA" if not a.endswith("AAAA") else "BBBB")
    with pytest.raises(Exception):
        realname.decrypt(broken)


def test_policy_parsing_and_fallback(monkeypatch, settings):
    from app.services import runtime_config

    assert realname.parse_policy({"mode": "id_card", "scope": "users", "user_ids": ["a"]}) == \
        realname.Policy("id_card", "users", frozenset({"a"}), "admin")
    for bad in (None, [], {"mode": "strict"}, {"mode": "phone", "scope": "some"},
                {"mode": "phone", "user_ids": "a"}, {"mode": "phone", "user_ids": [""]}):
        assert realname.parse_policy(bad) is None
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {})
    assert realname.policy() == realname.Policy("id_card")  # 没保存过: 环境变量兜底
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE",
                        {"realnamePolicy": {"mode": "off", "scope": "all", "user_ids": []}})
    assert realname.policy().mode == "off"  # 后台保存过就以后台为准, 环境变量不再生效


@pytest.mark.asyncio
async def test_user_scope_only_gates_listed_accounts(store, monkeypatch):
    from app.services import runtime_config

    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {"realnamePolicy": {
        "mode": "id_card", "scope": "users", "user_ids": ["u1"]}})
    assert await realname.chat_block_reason("u1") == "id_card"
    # 名单外 (例如还在用旧版 App 的内测用户) 照常使用, 也不能主动去核验
    assert await realname.chat_block_reason("u2") is None
    assert (await realname.load_state("u2")).required == []
    with pytest.raises(realname.RealnameError, match="not_required"):
        await realname.verify("u2", name="张三", id_number=ADULT_ID)
    assert await realname.verify("u1", name="张三", id_number=ADULT_ID) == "verified"
    monkeypatch.setattr(runtime_config, "_GLOBAL_CACHE", {"realnamePolicy": {
        "mode": "id_card", "scope": "all", "user_ids": []}})
    assert await realname.chat_block_reason("u2") == "id_card"


@pytest.mark.asyncio
async def test_minor_lock_keeps_only_digest(store):
    minor = _born(date(2011, 3, 4))
    assert await realname.verify("u1", name="张三", id_number=minor) == "minor"
    row = store.rows["u1"]
    assert row["id_digest"] and row["is_adult"] is False
    for key in ("masked_name", "name_encrypted", "id_number_encrypted", "birth_date",
                "gender", "region_code", "province"):
        assert row[key] is None, key
