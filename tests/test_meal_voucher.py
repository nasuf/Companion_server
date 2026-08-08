"""霸王餐: 两阶段扫码券状态机, 商家匹配, 端点映射."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import settings
from app.services import meal_voucher as mv


@pytest.mark.asyncio
async def test_validation_toggle_updates_only_enabled_flag(monkeypatch):
    upsert = AsyncMock()
    _mock_db(monkeypatch, systemconfig=SimpleNamespace(upsert=upsert))

    await mv.set_code_enabled(True)
    data = upsert.await_args.kwargs["data"]
    assert data["update"] == {"mealCodeEnabled": True}

    await mv.set_code_enabled(False)
    data = upsert.await_args.kwargs["data"]
    assert data["update"] == {"mealCodeEnabled": False}


# ── voucher state machine (db mocked) ─────────────────────────────


def _voucher(status: str, merchant_id: str | None = None):
    return SimpleNamespace(
        id="v-1",
        userId="user-1",
        status=status,
        activatedAt=None,
        redeemedAt=None,
        merchantId=merchant_id,
    )


class _FakeTx:
    """Async CM standing in for prisma ``db.tx()``; yields the same mocked db so
    ``tx.query_raw`` / ``tx.mealvoucher.*`` map onto the test's mocks."""

    def __init__(self, db):
        self._db = db

    async def __aenter__(self):
        return self._db

    async def __aexit__(self, *exc):
        return False


def _mock_db(monkeypatch, **tables):
    db = SimpleNamespace(**tables)
    if not hasattr(db, "query_raw"):
        db.query_raw = AsyncMock(return_value=[])
    db.tx = lambda: _FakeTx(db)
    monkeypatch.setattr(mv, "db", db)
    return db


@pytest.mark.asyncio
async def test_activate_requires_enabled(monkeypatch):
    monkeypatch.setattr(mv, "is_code_enabled", AsyncMock(return_value=False))

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.activate_voucher_by_staff("v-1", "user-1")
    assert exc.value.reason == "disabled"


@pytest.mark.asyncio
async def test_activate_happy_path(monkeypatch):
    monkeypatch.setattr(mv, "is_code_enabled", AsyncMock(return_value=True))
    updated = _voucher(mv.VOUCHER_ACTIVATED)
    db = _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(
                side_effect=[_voucher(mv.VOUCHER_INACTIVE), updated]
            ),
            update_many=AsyncMock(return_value=1),
        ),
    )

    result = await mv.activate_voucher_by_staff("v-1", "user-1")

    assert result.status == mv.VOUCHER_ACTIVATED
    kwargs = db.mealvoucher.update_many.await_args.kwargs
    # conditional transition: only flips inactive vouchers
    assert kwargs["where"]["status"] == mv.VOUCHER_INACTIVE
    assert kwargs["data"]["status"] == mv.VOUCHER_ACTIVATED
    assert kwargs["data"]["activatedAt"] is not None


@pytest.mark.asyncio
async def test_activate_concurrent_race_maps_to_already_activated(monkeypatch):
    monkeypatch.setattr(mv, "is_code_enabled", AsyncMock(return_value=True))
    _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(return_value=_voucher(mv.VOUCHER_INACTIVE)),
            update_many=AsyncMock(return_value=0),  # lost the race
        ),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.activate_voucher_by_staff("v-1", "user-1")
    assert exc.value.reason == "already_activated"


@pytest.mark.asyncio
async def test_activate_rejects_mismatched_qr_identity(monkeypatch):
    monkeypatch.setattr(mv, "is_code_enabled", AsyncMock(return_value=True))
    _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(return_value=_voucher(mv.VOUCHER_INACTIVE)),
        ),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.activate_voucher_by_staff("v-1", "other-user")
    assert exc.value.reason == "invalid_qr"


@pytest.mark.asyncio
async def test_activate_idempotency_guards(monkeypatch):
    monkeypatch.setattr(mv, "is_code_enabled", AsyncMock(return_value=True))
    for status, reason in [
        (mv.VOUCHER_ACTIVATED, "already_activated"),
        (mv.VOUCHER_REDEEMED, "already_redeemed"),
    ]:
        _mock_db(
            monkeypatch,
            mealvoucher=SimpleNamespace(
                find_unique=AsyncMock(return_value=_voucher(status)),
            ),
        )
        with pytest.raises(mv.MealVoucherError) as exc:
            await mv.activate_voucher_by_staff("v-1", "user-1")
        assert exc.value.reason == reason


@pytest.mark.asyncio
async def test_merchant_qr_redeem_requires_activated(monkeypatch):
    merchant = SimpleNamespace(id="m-1", name="伴生宴", codeActive=True)
    _mock_db(
        monkeypatch,
        mealmerchant=SimpleNamespace(find_unique=AsyncMock(return_value=merchant)),
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(return_value=_voucher(mv.VOUCHER_INACTIVE)),
        ),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.redeem_voucher_by_merchant("v-1", "user-1", "m-1")
    assert exc.value.reason == "not_activated"


@pytest.mark.asyncio
async def test_merchant_qr_redeem_rejects_double_redeem(monkeypatch):
    merchant = SimpleNamespace(id="m-1", name="伴生宴", codeActive=True)
    _mock_db(
        monkeypatch,
        mealmerchant=SimpleNamespace(find_unique=AsyncMock(return_value=merchant)),
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(return_value=_voucher(mv.VOUCHER_REDEEMED)),
        ),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.redeem_voucher_by_merchant("v-1", "user-1", "m-1")
    assert exc.value.reason == "already_redeemed"


@pytest.mark.asyncio
async def test_concurrent_merchant_qr_redeem_has_single_winner(monkeypatch):
    merchant = SimpleNamespace(id="m-1", name="伴生宴", codeActive=True)
    _mock_db(
        monkeypatch,
        mealmerchant=SimpleNamespace(find_unique=AsyncMock(return_value=merchant)),
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(return_value=_activated_voucher()),
            update_many=AsyncMock(return_value=0),
            count=AsyncMock(return_value=0),
        ),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.redeem_voucher_by_merchant("v-1", "user-1", "m-1")
    assert exc.value.reason == "already_redeemed"


@pytest.mark.asyncio
async def test_merchant_qr_redeem_binds_authenticated_merchant(monkeypatch):
    merchant = SimpleNamespace(id="m-auth", name="伴生宴", codeActive=True)
    updated = _voucher(mv.VOUCHER_REDEEMED, merchant_id="m-auth")
    updated.redeemedAt = datetime.now(UTC)
    db = _mock_db(
        monkeypatch,
        mealmerchant=SimpleNamespace(find_unique=AsyncMock(return_value=merchant)),
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(
                side_effect=[_activated_voucher(), updated]
            ),
            update_many=AsyncMock(return_value=1),
            count=AsyncMock(return_value=0),
        ),
    )

    result = await mv.redeem_voucher_by_merchant("v-1", "user-1", "m-auth")

    assert result.merchantId == "m-auth"
    update = db.mealvoucher.update_many.await_args.kwargs["data"]
    assert update["merchantId"] == "m-auth"


# ── admin clear (清除校验/核销) ────────────────────────────────────


@pytest.mark.asyncio
async def test_clear_redemption_conditional_transition(monkeypatch):
    db = _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(update_many=AsyncMock(return_value=1)),
    )

    await mv.clear_redemption("v-1")

    kwargs = db.mealvoucher.update_many.await_args.kwargs
    assert kwargs["where"] == {"id": "v-1", "status": mv.VOUCHER_REDEEMED}
    assert kwargs["data"] == {
        "status": mv.VOUCHER_ACTIVATED,
        "redeemedAt": None,
        "merchantId": None,
    }


@pytest.mark.asyncio
async def test_clear_redemption_rejects_wrong_state(monkeypatch):
    _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(update_many=AsyncMock(return_value=0)),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.clear_redemption("v-1")
    assert exc.value.reason == "not_redeemed"


@pytest.mark.asyncio
async def test_clear_activation_resets_everything(monkeypatch):
    db = _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(update_many=AsyncMock(return_value=1)),
    )

    await mv.clear_activation("v-1")

    kwargs = db.mealvoucher.update_many.await_args.kwargs
    assert kwargs["where"]["status"] == {
        "in": [mv.VOUCHER_ACTIVATED, mv.VOUCHER_REDEEMED]
    }
    assert kwargs["data"] == {
        "status": mv.VOUCHER_INACTIVE,
        "activatedAt": None,
        "redeemedAt": None,
        "merchantId": None,
    }


@pytest.mark.asyncio
async def test_clear_activation_rejects_inactive(monkeypatch):
    _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(update_many=AsyncMock(return_value=0)),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.clear_activation("v-1")
    assert exc.value.reason == "not_activated"


@pytest.mark.asyncio
async def test_admin_clear_endpoints_map_errors(monkeypatch):
    from fastapi import HTTPException

    from app.api.admin import meal as admin_meal

    # 404: voucher missing
    monkeypatch.setattr(
        admin_meal,
        "db",
        SimpleNamespace(
            mealvoucher=SimpleNamespace(find_unique=AsyncMock(return_value=None))
        ),
    )
    with pytest.raises(HTTPException) as exc:
        await admin_meal.clear_redemption("nope", payload={"sub": "admin-1"})
    assert exc.value.status_code == 404

    # 400: wrong state
    monkeypatch.setattr(
        admin_meal,
        "db",
        SimpleNamespace(
            mealvoucher=SimpleNamespace(
                find_unique=AsyncMock(return_value=SimpleNamespace(id="v-1"))
            )
        ),
    )
    monkeypatch.setattr(
        admin_meal.mv,
        "clear_activation",
        AsyncMock(side_effect=mv.MealVoucherError("not_activated", "该券当前未激活")),
    )
    with pytest.raises(HTTPException) as exc:
        await admin_meal.clear_activation("v-1", payload={"sub": "admin-1"})
    assert exc.value.status_code == 400


# ── merchant contact matching ─────────────────────────────────────


def _merchant(contact_name=None, contact_phone=None):
    return SimpleNamespace(contactName=contact_name, contactPhone=contact_phone)


def test_contact_match_by_name():
    assert mv.merchant_contact_matches(_merchant(contact_name="王老板"), "王老板")
    assert not mv.merchant_contact_matches(_merchant(contact_name="王老板"), "李老板")


def test_contact_match_by_phone_ignores_separators():
    m = _merchant(contact_phone="13812345678")
    assert mv.merchant_contact_matches(m, "138 1234 5678")
    assert mv.merchant_contact_matches(m, "13812345678")
    assert not mv.merchant_contact_matches(m, "13800000000")


def test_contact_match_rejects_empty_both_sides():
    # merchant without any contact info can never self-serve login
    assert not mv.merchant_contact_matches(_merchant(), "")
    assert not mv.merchant_contact_matches(_merchant(), "任意")


# ── endpoint wiring (direct call, deps monkeypatched) ─────────────


class FakeRequest:
    headers = {}
    client = SimpleNamespace(host="1.2.3.4")


@pytest.mark.asyncio
async def test_staff_login_rejects_wrong_key(monkeypatch):
    from app.api.public import meal as meal_api
    from fastapi import HTTPException

    monkeypatch.setattr(meal_api.settings, "meal_staff_key", "sekret")

    with pytest.raises(HTTPException) as exc:
        await meal_api.staff_login(
            meal_api.StaffLoginRequest(key="wrong"), FakeRequest()
        )
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_staff_login_accepts_chinese_key(monkeypatch):
    # hmac.compare_digest 不支持非 ASCII str; 中文口令须按 UTF-8 bytes 比对。
    from app.api.public import meal as meal_api
    from app.services.auth import decode_jwt

    monkeypatch.setattr(meal_api.settings, "meal_staff_key", "千味央厨")

    body = await meal_api.staff_login(
        meal_api.StaffLoginRequest(key="千味央厨"), FakeRequest()
    )
    payload = decode_jwt(body["token"])
    assert payload["role"] == "meal_staff"


@pytest.mark.asyncio
async def test_staff_login_rejects_wrong_chinese_key(monkeypatch):
    from app.api.public import meal as meal_api
    from fastapi import HTTPException

    monkeypatch.setattr(meal_api.settings, "meal_staff_key", "千味央厨")

    with pytest.raises(HTTPException) as exc:
        await meal_api.staff_login(
            meal_api.StaffLoginRequest(key="霸王餐"), FakeRequest()
        )
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_staff_login_issues_scoped_token(monkeypatch):
    from app.api.public import meal as meal_api
    from app.services.auth import decode_jwt

    monkeypatch.setattr(meal_api.settings, "meal_staff_key", "sekret")

    body = await meal_api.staff_login(
        meal_api.StaffLoginRequest(key="SEKRET"), FakeRequest()
    )
    payload = decode_jwt(body["token"])
    assert payload["sub"] == "meal_staff"
    assert payload["role"] == "meal_staff"


@pytest.mark.asyncio
async def test_staff_scan_endpoint_activates_bound_voucher(monkeypatch):
    from app.api.public import meal as meal_api
    activated_at = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)
    voucher = SimpleNamespace(
        id="v-1", userId="user-1", activatedAt=activated_at
    )

    monkeypatch.setattr(meal_api, "_require_staff", lambda _: "meal_staff")
    monkeypatch.setattr(meal_api.mv, "is_code_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(
        meal_api.meal_qr,
        "consume",
        AsyncMock(return_value={"voucher_id": "v-1", "user_id": "user-1"}),
    )
    activate = AsyncMock(return_value=voucher)
    monkeypatch.setattr(meal_api.mv, "activate_voucher_by_staff", activate)
    monkeypatch.setattr(
        meal_api.mv, "resolve_user_display", AsyncMock(return_value="小明")
    )

    body = await meal_api.staff_activate_scan(
        meal_api.MealScanRequest(value="CPMEAL:1:" + "a" * 43),
        FakeRequest(),
    )
    meal_api.meal_qr.consume.assert_awaited_once_with(
        "CPMEAL:1:" + "a" * 43, "activate"
    )
    activate.assert_awaited_once_with("v-1", "user-1")
    assert body["user_display"] == "小明"


@pytest.mark.asyncio
async def test_staff_scan_disabled_does_not_consume_qr(monkeypatch):
    from app.api.public import meal as meal_api
    from fastapi import HTTPException

    monkeypatch.setattr(meal_api, "_require_staff", lambda _: "meal_staff")
    monkeypatch.setattr(meal_api.mv, "is_code_enabled", AsyncMock(return_value=False))
    consume = AsyncMock()
    monkeypatch.setattr(meal_api.meal_qr, "consume", consume)

    with pytest.raises(HTTPException) as exc:
        await meal_api.staff_activate_scan(
            meal_api.MealScanRequest(value="CPMEAL:1:" + "a" * 43),
            FakeRequest(),
        )
    assert exc.value.status_code == 403
    consume.assert_not_awaited()


@pytest.mark.asyncio
async def test_merchant_login_issues_scoped_token(monkeypatch):
    from app.api.public import meal as meal_api
    from app.services.auth import decode_jwt

    merchant = SimpleNamespace(
        id="m-1", name="张记食堂", contactName="王老板", contactPhone=None
    )
    monkeypatch.setattr(meal_api, "enforce_login_rate_limit", AsyncMock())
    monkeypatch.setattr(
        meal_api,
        "db",
        SimpleNamespace(
            mealmerchant=SimpleNamespace(find_unique=AsyncMock(return_value=merchant))
        ),
    )

    body = await meal_api.merchant_login(
        meal_api.MerchantLoginRequest(merchant_id="m-1", contact="王老板"),
        FakeRequest(),
    )

    payload = decode_jwt(body["token"])
    assert payload["sub"] == "m-1"
    assert payload["role"] == "meal_merchant"
    assert body["merchant"]["name"] == "张记食堂"


@pytest.mark.asyncio
async def test_merchant_login_returns_qwyc_group_flag(monkeypatch):
    from app.api.public import meal as meal_api

    merchant = SimpleNamespace(
        id="g-1",
        name="千味央厨",
        contactName="王老板",
        contactPhone=None,
        qwycGroup=True,
    )
    monkeypatch.setattr(meal_api, "enforce_login_rate_limit", AsyncMock())
    monkeypatch.setattr(
        meal_api,
        "db",
        SimpleNamespace(
            mealmerchant=SimpleNamespace(find_unique=AsyncMock(return_value=merchant))
        ),
    )

    body = await meal_api.merchant_login(
        meal_api.MerchantLoginRequest(merchant_id="g-1", contact="王老板"),
        FakeRequest(),
    )
    assert body["merchant"]["qwyc_group"] is True


@pytest.mark.asyncio
async def test_qwyc_summary_aggregates_members(monkeypatch):
    members = [
        SimpleNamespace(id="m-1", name="A店"),
        SimpleNamespace(id="m-2", name="B店"),
    ]
    # (merchant_id, is_today) -> count
    counts = {
        ("m-1", False): 5,
        ("m-1", True): 2,
        ("m-2", False): 3,
        ("m-2", True): 1,
    }

    async def count(where):
        return counts[(where["merchantId"], "redeemedAt" in where)]

    _mock_db(
        monkeypatch,
        mealmerchant=SimpleNamespace(find_many=AsyncMock(return_value=members)),
        mealvoucher=SimpleNamespace(count=count),
    )

    result = await mv.qwyc_summary()

    assert result["cumulative_total"] == 8
    assert result["today_total"] == 3
    assert result["members"][0] == {
        "merchant_id": "m-1",
        "name": "A店",
        "today_redeemed": 2,
        "total_redeemed": 5,
    }


@pytest.mark.asyncio
async def test_qwyc_summary_endpoint_rejects_non_group(monkeypatch):
    from app.api.public import meal as meal_api
    from fastapi import HTTPException

    monkeypatch.setattr(meal_api, "_require_merchant", lambda _: "m-1")
    monkeypatch.setattr(
        meal_api,
        "db",
        SimpleNamespace(
            mealmerchant=SimpleNamespace(
                find_unique=AsyncMock(
                    return_value=SimpleNamespace(id="m-1", name="张记", qwycGroup=False)
                )
            )
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await meal_api.qwyc_summary(FakeRequest())
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_qwyc_summary_endpoint_returns_for_group(monkeypatch):
    from app.api.public import meal as meal_api

    monkeypatch.setattr(meal_api, "_require_merchant", lambda _: "g-1")
    monkeypatch.setattr(
        meal_api,
        "db",
        SimpleNamespace(
            mealmerchant=SimpleNamespace(
                find_unique=AsyncMock(
                    return_value=SimpleNamespace(
                        id="g-1", name="千味央厨", qwycGroup=True
                    )
                )
            )
        ),
    )
    monkeypatch.setattr(
        meal_api.mv,
        "qwyc_summary",
        AsyncMock(
            return_value={"members": [], "today_total": 0, "cumulative_total": 0}
        ),
    )
    monkeypatch.setattr(
        meal_api.mv,
        "daily_redeem_quota",
        AsyncMock(
            return_value={"daily_cap": 500, "daily_used": 12, "daily_remaining": 488}
        ),
    )

    body = await meal_api.qwyc_summary(FakeRequest())
    assert body["merchant_name"] == "千味央厨"
    assert body["today_total"] == 0
    assert body["daily_cap"] == 500
    assert body["daily_used"] == 12
    assert body["daily_remaining"] == 488


@pytest.mark.asyncio
async def test_admin_create_merchant_sets_qwyc_flags(monkeypatch):
    from app.api.admin import meal as admin_meal

    created = SimpleNamespace(
        id="m-1",
        name="A店",
        contactName=None,
        contactPhone=None,
        codeActive=True,
        qwycMember=True,
        qwycGroup=False,
        createdAt=None,
    )
    create = AsyncMock(return_value=created)
    monkeypatch.setattr(
        admin_meal,
        "db",
        SimpleNamespace(mealmerchant=SimpleNamespace(create=create)),
    )

    body = await admin_meal.create_merchant(
        admin_meal.MerchantCreateRequest(name="A店", qwyc_member=True)
    )
    data = create.await_args.kwargs["data"]
    assert data["qwycMember"] is True
    assert data["qwycGroup"] is False
    assert body["qwyc_member"] is True
    assert body["qwyc_group"] is False


@pytest.mark.asyncio
async def test_admin_update_merchant_sets_qwyc_flags(monkeypatch):
    from app.api.admin import meal as admin_meal

    updated = SimpleNamespace(
        id="m-1",
        name="A店",
        contactName=None,
        contactPhone=None,
        codeActive=True,
        qwycMember=False,
        qwycGroup=True,
        createdAt=None,
    )
    update = AsyncMock(return_value=updated)
    monkeypatch.setattr(
        admin_meal,
        "db",
        SimpleNamespace(
            mealmerchant=SimpleNamespace(
                find_unique=AsyncMock(return_value=SimpleNamespace(id="m-1")),
                update=update,
            ),
            query_raw=AsyncMock(return_value=[]),
        ),
    )

    body = await admin_meal.update_merchant(
        "m-1", admin_meal.MerchantUpdateRequest(qwyc_group=True)
    )
    data = update.await_args.kwargs["data"]
    assert data["qwycGroup"] is True
    assert body["qwyc_group"] is True


@pytest.mark.asyncio
async def test_merchant_login_rejects_mismatch(monkeypatch):
    from app.api.public import meal as meal_api
    from fastapi import HTTPException

    merchant = SimpleNamespace(
        id="m-1", name="张记食堂", contactName="王老板", contactPhone=None
    )
    monkeypatch.setattr(meal_api, "enforce_login_rate_limit", AsyncMock())
    monkeypatch.setattr(meal_api, "record_login_failure", AsyncMock())
    monkeypatch.setattr(
        meal_api,
        "db",
        SimpleNamespace(
            mealmerchant=SimpleNamespace(find_unique=AsyncMock(return_value=merchant))
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await meal_api.merchant_login(
            meal_api.MerchantLoginRequest(merchant_id="m-1", contact="李老板"),
            FakeRequest(),
        )
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_merchant_token_rejected_for_user_endpoints(monkeypatch):
    """A meal_merchant token must not pass require_user-based... (role check)."""
    from app.api.public import meal as meal_api
    from fastapi import HTTPException

    # user JWT used against merchant endpoint -> 403
    from app.services.auth import create_jwt

    user_token = create_jwt("user-1", "user")
    request = SimpleNamespace(headers={"authorization": f"Bearer {user_token}"})
    with pytest.raises(HTTPException) as exc:
        meal_api._require_merchant(request)
    assert exc.value.status_code == 403


# ── admin merchants validation ────────────────────────────────────


@pytest.mark.asyncio
async def test_admin_redemptions_detail(monkeypatch):
    from datetime import UTC, datetime

    from app.api.admin import meal as admin_meal

    merchant = SimpleNamespace(id="m-1", name="伴生宴")
    row = SimpleNamespace(
        id="v-1",
        userId="user-1",
        activatedAt=datetime(2026, 7, 7, 5, 0, tzinfo=UTC),
        redeemedAt=datetime(2026, 7, 8, 5, 0, tzinfo=UTC),
    )
    monkeypatch.setattr(
        admin_meal,
        "db",
        SimpleNamespace(
            mealmerchant=SimpleNamespace(find_unique=AsyncMock(return_value=merchant)),
            mealvoucher=SimpleNamespace(
                find_many=AsyncMock(return_value=[row]),
                count=AsyncMock(return_value=1),
            ),
        ),
    )
    monkeypatch.setattr(
        admin_meal.mv,
        "resolve_user_redemption_profiles",
        AsyncMock(
            return_value={
                "user-1": {
                    "user_id": "user-1",
                    "username": "wx_user",
                    "user_display": "小明",
                    "phone_masked": "138****5678",
                    "wechat_nickname": "小明",
                    "wechat_avatar_url": "https://example.com/avatar.jpg",
                    "wechat_openid": "openid-1",
                    "wechat_unionid": "unionid-1",
                }
            }
        ),
    )

    body = await admin_meal.merchant_redemptions("m-1")

    assert body["merchant_name"] == "伴生宴"
    assert body["total"] == 1
    assert body["items"][0]["user_display"] == "小明"
    assert body["items"][0]["user_id"] == "user-1"
    assert body["items"][0]["activated_at"].startswith("2026-07-07")
    assert body["items"][0]["redeemed_at"].startswith("2026-07-08")


@pytest.mark.asyncio
async def test_admin_redemptions_404_unknown_merchant(monkeypatch):
    from fastapi import HTTPException

    from app.api.admin import meal as admin_meal

    monkeypatch.setattr(
        admin_meal,
        "db",
        SimpleNamespace(
            mealmerchant=SimpleNamespace(find_unique=AsyncMock(return_value=None))
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await admin_meal.merchant_redemptions("nope")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_admin_range_stats_fills_missing_days(monkeypatch):
    from app.api.admin import meal as admin_meal

    query = AsyncMock(
        side_effect=[
            [{"day": "2026-07-06", "cnt": 3}, {"day": "2026-07-08", "cnt": 1}],
            [{"day": "2026-07-07", "cnt": 2}],
        ]
    )
    monkeypatch.setattr(admin_meal, "db", SimpleNamespace(query_raw=query))

    body = await admin_meal.range_stats("2026-07-06", "2026-07-08")

    assert body["activated_total"] == 4
    assert body["redeemed_total"] == 2
    assert [d["date"] for d in body["days"]] == [
        "2026-07-06",
        "2026-07-07",
        "2026-07-08",
    ]
    assert body["days"][1] == {"date": "2026-07-07", "activated": 0, "redeemed": 2}
    # CN 自然日 → UTC 边界: 07-06 00:00 CN = 07-05 16:00 UTC
    first_call_args = query.await_args_list[0].args
    assert first_call_args[1] == "2026-07-05 16:00:00"
    assert first_call_args[2] == "2026-07-08 16:00:00"


@pytest.mark.asyncio
async def test_admin_range_stats_validation(monkeypatch):
    from fastapi import HTTPException

    from app.api.admin import meal as admin_meal

    monkeypatch.setattr(
        admin_meal, "db", SimpleNamespace(query_raw=AsyncMock(return_value=[]))
    )

    with pytest.raises(HTTPException) as exc:
        await admin_meal.range_stats("2026/07/06", "2026-07-08")
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        await admin_meal.range_stats("2026-07-08", "2026-07-06")
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        await admin_meal.range_stats("2024-01-01", "2026-07-08")
    assert exc.value.status_code == 400


# ── validity / daily cap ──────────────────────────────────────────


def _activated_voucher(*, days_ago: float = 0):
    return SimpleNamespace(
        id="v-1",
        userId="user-1",
        status=mv.VOUCHER_ACTIVATED,
        activatedAt=datetime.now(UTC) - timedelta(days=days_ago),
        redeemedAt=None,
        merchantId=None,
    )


def test_voucher_expires_at_and_expired_flag():
    voucher = _activated_voucher(days_ago=2)
    expires = mv.voucher_expires_at(voucher)
    assert expires is not None
    assert expires == voucher.activatedAt + timedelta(days=settings.meal_validity_days)
    assert mv.is_voucher_expired(voucher) is False

    old = _activated_voucher(days_ago=settings.meal_validity_days + 0.01)
    assert mv.is_voucher_expired(old) is True
    assert mv.is_voucher_expired(_voucher(mv.VOUCHER_INACTIVE)) is False


@pytest.mark.asyncio
async def test_redeem_rejects_expired_voucher(monkeypatch):
    merchant = SimpleNamespace(id="m-1", name="张记食堂", codeActive=True)
    _mock_db(
        monkeypatch,
        mealmerchant=SimpleNamespace(find_unique=AsyncMock(return_value=merchant)),
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(return_value=_activated_voucher(days_ago=8)),
        ),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.redeem_voucher_by_merchant("v-1", "user-1", "m-1")
    assert exc.value.reason == "expired"


@pytest.mark.asyncio
async def test_redeem_rejects_daily_cap_and_records_failure(monkeypatch):
    merchant = SimpleNamespace(id="m-1", name="张记食堂", codeActive=True)
    db = _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(return_value=_activated_voucher()),
            count=AsyncMock(return_value=settings.meal_daily_redeem_cap),
        ),
        mealmerchant=SimpleNamespace(find_unique=AsyncMock(return_value=merchant)),
        mealredemptionfailure=SimpleNamespace(
            find_first=AsyncMock(return_value=None),
            create=AsyncMock(),
        ),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.redeem_voucher_by_merchant("v-1", "user-1", "m-1")
    assert exc.value.reason == "daily_cap"
    db.mealredemptionfailure.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_redeem_serializes_cap_check_with_advisory_lock(monkeypatch):
    """核销的「计数 → 条件转移」必须在事务内加 advisory 锁串行化 (防超额)。"""
    merchant = SimpleNamespace(id="m-1", name="张记食堂", codeActive=True)
    updated = _voucher(mv.VOUCHER_REDEEMED, merchant_id="m-1")
    updated.redeemedAt = datetime.now(UTC)
    db = _mock_db(
        monkeypatch,
        mealmerchant=SimpleNamespace(find_unique=AsyncMock(return_value=merchant)),
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(side_effect=[_activated_voucher(), updated]),
            update_many=AsyncMock(return_value=1),
            count=AsyncMock(return_value=0),
        ),
    )

    await mv.redeem_voucher_by_merchant("v-1", "user-1", "m-1")

    # 计数与条件更新都发生在同一事务里 (tx == db), 且先取了 advisory 锁。
    lock_sql = db.query_raw.await_args_list[0].args[0]
    assert "pg_advisory_xact_lock" in lock_sql
    db.mealvoucher.count.assert_awaited_once()
    db.mealvoucher.update_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_redeem_over_cap_does_not_transition(monkeypatch):
    """当日已达上限时, 不得写入核销状态 (只留痕 + 抛 daily_cap)。"""
    merchant = SimpleNamespace(id="m-1", name="张记食堂", codeActive=True)
    db = _mock_db(
        monkeypatch,
        mealmerchant=SimpleNamespace(find_unique=AsyncMock(return_value=merchant)),
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(return_value=_activated_voucher()),
            update_many=AsyncMock(return_value=1),
            count=AsyncMock(return_value=settings.meal_daily_redeem_cap),
        ),
        mealredemptionfailure=SimpleNamespace(
            find_first=AsyncMock(return_value=None),
            create=AsyncMock(),
        ),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.redeem_voucher_by_merchant("v-1", "user-1", "m-1")
    assert exc.value.reason == "daily_cap"
    db.mealvoucher.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_daily_redeem_quota(monkeypatch):
    _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(count=AsyncMock(return_value=3)),
    )
    quota = await mv.daily_redeem_quota()
    assert quota["daily_cap"] == settings.meal_daily_redeem_cap
    assert quota["daily_used"] == 3
    assert quota["daily_remaining"] == settings.meal_daily_redeem_cap - 3


@pytest.mark.asyncio
async def test_daily_redeem_quota_clamps_remaining_at_zero(monkeypatch):
    _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(
            count=AsyncMock(return_value=settings.meal_daily_redeem_cap + 5)
        ),
    )
    quota = await mv.daily_redeem_quota()
    assert quota["daily_remaining"] == 0


@pytest.mark.asyncio
async def test_quota_endpoint_returns_snapshot(monkeypatch):
    from app.api.public import meal as meal_api

    monkeypatch.setattr(
        meal_api.mv,
        "daily_redeem_quota",
        AsyncMock(return_value={"daily_cap": 500, "daily_used": 10, "daily_remaining": 490}),
    )
    body = await meal_api.voucher_quota(payload={"sub": "user-1"})
    assert body == {"daily_cap": 500, "daily_used": 10, "daily_remaining": 490}


@pytest.mark.asyncio
async def test_merchant_stats_includes_daily_quota(monkeypatch):
    from app.api.public import meal as meal_api

    merchant = SimpleNamespace(id="m-1", name="伴生宴")
    monkeypatch.setattr(meal_api, "_require_merchant", lambda _: "m-1")
    monkeypatch.setattr(
        meal_api,
        "db",
        SimpleNamespace(
            mealmerchant=SimpleNamespace(find_unique=AsyncMock(return_value=merchant)),
            mealvoucher=SimpleNamespace(
                find_many=AsyncMock(return_value=[]),
                count=AsyncMock(return_value=7),
            ),
        ),
    )
    monkeypatch.setattr(
        meal_api.mv, "resolve_user_displays", AsyncMock(return_value={})
    )
    monkeypatch.setattr(
        meal_api.mv,
        "daily_redeem_quota",
        AsyncMock(
            return_value={"daily_cap": 500, "daily_used": 120, "daily_remaining": 380}
        ),
    )

    body = await meal_api.merchant_stats(FakeRequest())

    assert body["merchant_name"] == "伴生宴"
    assert body["redeemed_total"] == 7
    assert body["daily_cap"] == 500
    assert body["daily_remaining"] == 380


@pytest.mark.asyncio
async def test_record_redemption_failure_dedupes_per_user_day(monkeypatch):
    existing = SimpleNamespace(id="f-1")
    db = _mock_db(
        monkeypatch,
        mealredemptionfailure=SimpleNamespace(
            find_first=AsyncMock(return_value=existing),
            create=AsyncMock(),
        ),
    )

    await mv.record_redemption_failure("user-1", "m-1", mv.FAILURE_DAILY_CAP)

    db.mealredemptionfailure.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_expired_vouchers_feed(monkeypatch):
    row = _activated_voucher(days_ago=10)
    _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(find_many=AsyncMock(return_value=[row])),
    )
    monkeypatch.setattr(
        mv, "resolve_user_displays", AsyncMock(return_value={"user-1": "小明"})
    )

    items = await mv.expired_vouchers_feed(limit=10)

    assert len(items) == 1
    assert items[0]["user_display"] == "小明"
    assert items[0]["expired_at"] is not None


@pytest.mark.asyncio
async def test_redemption_failures_feed(monkeypatch):
    row = SimpleNamespace(
        userId="user-1",
        merchantId="m-1",
        createdAt=datetime(2026, 7, 10, 12, 0, tzinfo=UTC),
    )
    merchant = SimpleNamespace(id="m-1", name="伴生宴")
    _mock_db(
        monkeypatch,
        mealredemptionfailure=SimpleNamespace(find_many=AsyncMock(return_value=[row])),
        mealmerchant=SimpleNamespace(find_many=AsyncMock(return_value=[merchant])),
    )
    monkeypatch.setattr(
        mv, "resolve_user_displays", AsyncMock(return_value={"user-1": "小明"})
    )

    items = await mv.redemption_failures_feed()

    assert len(items) == 1
    assert items[0]["user_display"] == "小明"
    assert items[0]["merchant_name"] == "伴生宴"


@pytest.mark.asyncio
async def test_admin_expired_and_failures_endpoints(monkeypatch):
    from app.api.admin import meal as admin_meal

    monkeypatch.setattr(
        admin_meal.mv,
        "expired_vouchers_feed",
        AsyncMock(return_value=[{"voucher_id": "v-1", "user_display": "小明"}]),
    )
    monkeypatch.setattr(
        admin_meal.mv,
        "redemption_failures_feed",
        AsyncMock(
            return_value=[
                {
                    "user_display": "小红",
                    "merchant_name": "伴生宴",
                    "failed_at": "2026-07-10T12:00:00+00:00",
                }
            ]
        ),
    )

    expired = await admin_meal.expired_vouchers()
    assert expired[0]["user_display"] == "小明"

    failures = await admin_meal.redemption_failures(date="2026-07-10")
    assert failures["date"] == "2026-07-10"
    assert failures["total"] == 1
    assert failures["items"][0]["user_display"] == "小红"
