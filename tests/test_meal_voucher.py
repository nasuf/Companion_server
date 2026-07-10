"""霸王餐: 轮换码确定性/宽限期, 券状态机, 商家匹配, 端点映射."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import settings
from app.services import meal_voucher as mv


# ── rotating code ─────────────────────────────────────────────────


def test_code_is_deterministic_within_window():
    base = 1_760_000_000 - (1_760_000_000 % 300)  # window start
    code_a, _ = mv.current_activation_code(now=base + 10)
    code_b, _ = mv.current_activation_code(now=base + 290)
    assert code_a == code_b
    assert len(code_a) == 6 and code_a.isdigit()


def test_code_rotates_across_windows():
    base = 1_760_000_000 - (1_760_000_000 % 300)
    code_a, _ = mv.current_activation_code(now=base + 10)
    code_next, _ = mv.current_activation_code(now=base + 310)
    assert code_a != code_next  # 2^-? collision chance ~1e-6; deterministic seed here


def test_expires_in_counts_down():
    base = 1_760_000_000 - (1_760_000_000 % 300)
    _, expires = mv.current_activation_code(now=base + 100)
    assert expires == 200


def test_verify_accepts_current_window():
    now = 1_760_000_123
    code, _ = mv.current_activation_code(now=now)
    assert mv.verify_activation_code(code, now=now) is True


def test_verify_accepts_previous_window_within_grace():
    base = 1_760_000_000 - (1_760_000_000 % 300)
    prev_code, _ = mv.current_activation_code(now=base - 10)  # previous window
    # 15s into the new window -> grace (30s) still accepts the previous code
    assert mv.verify_activation_code(prev_code, now=base + 15) is True
    # 45s in -> grace over
    assert mv.verify_activation_code(prev_code, now=base + 45) is False


def test_verify_rejects_malformed():
    assert mv.verify_activation_code("12345") is False
    assert mv.verify_activation_code("abcdef") is False
    assert mv.verify_activation_code("") is False


# ── anchor (关闭→重新开启 重新生成) ────────────────────────────────


def test_different_anchor_regenerates_code_same_wall_clock():
    now = 1_760_000_123
    code_a, _ = mv.current_activation_code(now=now, anchor=0)
    code_b, _ = mv.current_activation_code(now=now, anchor=1_760_000_100)
    assert code_a != code_b  # deterministic seeds chosen to differ


def test_fresh_anchor_gives_full_countdown():
    now = 1_760_000_123
    _, expires = mv.current_activation_code(now=now, anchor=now)
    assert expires == mv.CODE_WINDOW_SECONDS


def test_old_anchor_code_rejected_after_reanchor():
    now = 1_760_000_123
    old_code, _ = mv.current_activation_code(now=now, anchor=0)
    # after re-enable the anchor moves to `now` — the old code must die
    assert mv.verify_activation_code(old_code, now=now + 1, anchor=now) is False
    new_code, _ = mv.current_activation_code(now=now + 1, anchor=now)
    assert mv.verify_activation_code(new_code, now=now + 1, anchor=now) is True


def test_grace_is_anchor_relative():
    anchor = 1_760_000_000
    boundary = anchor + mv.CODE_WINDOW_SECONDS  # first rotation under anchor
    prev_code, _ = mv.current_activation_code(now=boundary - 5, anchor=anchor)
    assert mv.verify_activation_code(prev_code, now=boundary + 15, anchor=anchor)
    assert not mv.verify_activation_code(prev_code, now=boundary + 45, anchor=anchor)


@pytest.mark.asyncio
async def test_enable_refreshes_anchor_disable_keeps_it(monkeypatch):
    upsert = AsyncMock()
    _mock_db(monkeypatch, systemconfig=SimpleNamespace(upsert=upsert))

    await mv.set_code_enabled(True)
    data = upsert.await_args.kwargs["data"]
    assert data["update"]["mealCodeEnabled"] is True
    assert data["update"]["mealCodeAnchor"] > 0  # regenerated

    await mv.set_code_enabled(False)
    data = upsert.await_args.kwargs["data"]
    assert data["update"] == {"mealCodeEnabled": False}  # anchor untouched


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


def _mock_db(monkeypatch, **tables):
    db = SimpleNamespace(**tables)
    monkeypatch.setattr(mv, "db", db)
    return db


@pytest.mark.asyncio
async def test_activate_requires_enabled(monkeypatch):
    monkeypatch.setattr(mv, "is_code_enabled", AsyncMock(return_value=False))

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.activate_voucher("user-1", "123456")
    assert exc.value.reason == "disabled"


@pytest.mark.asyncio
async def test_activate_happy_path(monkeypatch):
    monkeypatch.setattr(mv, "is_code_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(mv, "verify_activation_code_now", AsyncMock(return_value=True))
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

    result = await mv.activate_voucher("user-1", "123456")

    assert result.status == mv.VOUCHER_ACTIVATED
    kwargs = db.mealvoucher.update_many.await_args.kwargs
    # conditional transition: only flips inactive vouchers
    assert kwargs["where"]["status"] == mv.VOUCHER_INACTIVE
    assert kwargs["data"]["status"] == mv.VOUCHER_ACTIVATED
    assert kwargs["data"]["activatedAt"] is not None


@pytest.mark.asyncio
async def test_activate_concurrent_race_maps_to_already_activated(monkeypatch):
    monkeypatch.setattr(mv, "is_code_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(mv, "verify_activation_code_now", AsyncMock(return_value=True))
    _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(return_value=_voucher(mv.VOUCHER_INACTIVE)),
            update_many=AsyncMock(return_value=0),  # lost the race
        ),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.activate_voucher("user-1", "123456")
    assert exc.value.reason == "already_activated"


@pytest.mark.asyncio
async def test_activate_rejects_wrong_code(monkeypatch):
    monkeypatch.setattr(mv, "is_code_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(mv, "verify_activation_code_now", AsyncMock(return_value=False))
    _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(return_value=_voucher(mv.VOUCHER_INACTIVE)),
        ),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.activate_voucher("user-1", "000000")
    assert exc.value.reason == "bad_code"


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
            await mv.activate_voucher("user-1", "123456")
        assert exc.value.reason == reason


@pytest.mark.asyncio
async def test_redeem_requires_activated(monkeypatch):
    _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(return_value=_voucher(mv.VOUCHER_INACTIVE)),
        ),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.redeem_voucher("user-1", "654321")
    assert exc.value.reason == "not_activated"


@pytest.mark.asyncio
async def test_redeem_happy_path(monkeypatch):
    merchant = SimpleNamespace(id="m-1", name="张记食堂", codeActive=True)
    updated = _voucher(mv.VOUCHER_REDEEMED, merchant_id="m-1")
    db = _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(
                side_effect=[_voucher(mv.VOUCHER_ACTIVATED), updated]
            ),
            update_many=AsyncMock(return_value=1),
            count=AsyncMock(return_value=0),  # 当日核销数远低于上限
        ),
        mealmerchant=SimpleNamespace(find_first=AsyncMock(return_value=merchant)),
    )

    result = await mv.redeem_voucher("user-1", "654321")

    assert result.status == mv.VOUCHER_REDEEMED
    where = db.mealmerchant.find_first.await_args.kwargs["where"]
    assert where == {"redeemCode": "654321", "codeActive": True}
    kwargs = db.mealvoucher.update_many.await_args.kwargs
    assert kwargs["where"]["status"] == mv.VOUCHER_ACTIVATED
    assert kwargs["data"]["merchantId"] == "m-1"


@pytest.mark.asyncio
async def test_redeem_rejects_unknown_or_inactive_code(monkeypatch):
    _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(return_value=_voucher(mv.VOUCHER_ACTIVATED)),
        ),
        mealmerchant=SimpleNamespace(find_first=AsyncMock(return_value=None)),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.redeem_voucher("user-1", "654321")
    assert exc.value.reason == "bad_code"


@pytest.mark.asyncio
async def test_redeem_rejects_double_redeem(monkeypatch):
    _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(return_value=_voucher(mv.VOUCHER_REDEEMED)),
        ),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.redeem_voucher("user-1", "654321")
    assert exc.value.reason == "already_redeemed"


@pytest.mark.asyncio
async def test_redeem_at_second_store_rejected(monkeypatch):
    """在 A 店核销后, 拿 B 店的码再核销必须被拒 — 一人只能核销一家店."""
    db = _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(return_value=_voucher(mv.VOUCHER_REDEEMED, "m-a")),
        ),
        mealmerchant=SimpleNamespace(find_first=AsyncMock()),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.redeem_voucher("user-1", "111111")  # B 店的码
    assert exc.value.reason == "already_redeemed"
    # 状态守卫在商家查询之前 — B 店的码根本不会被查
    db.mealmerchant.find_first.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_redeem_two_stores_single_winner(monkeypatch):
    """并发拿两家店的码核销: 条件转移只让一个成功, 输家收到已核销."""
    merchant_b = SimpleNamespace(id="m-b", name="B店", codeActive=True)
    _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(
            # 读的时候还是 activated (竞态窗口), 但条件 update 匹配 0 行
            find_unique=AsyncMock(return_value=_voucher(mv.VOUCHER_ACTIVATED)),
            update_many=AsyncMock(return_value=0),
            count=AsyncMock(return_value=0),
        ),
        mealmerchant=SimpleNamespace(find_first=AsyncMock(return_value=merchant_b)),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.redeem_voucher("user-1", "222222")
    assert exc.value.reason == "already_redeemed"


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
async def test_staff_code_requires_key_when_configured(monkeypatch):
    from app.api.public import meal as meal_api
    from fastapi import HTTPException

    monkeypatch.setattr(meal_api.settings, "meal_staff_key", "sekret")

    with pytest.raises(HTTPException) as exc:
        await meal_api.staff_code(key="wrong")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_staff_code_disabled_returns_no_code(monkeypatch):
    from app.api.public import meal as meal_api

    monkeypatch.setattr(meal_api.settings, "meal_staff_key", "")
    monkeypatch.setattr(meal_api.mv, "is_code_enabled", AsyncMock(return_value=False))

    body = await meal_api.staff_code(key="")
    assert body == {"enabled": False, "code": None, "expires_in": None}


@pytest.mark.asyncio
async def test_staff_code_returns_current_code(monkeypatch):
    from app.api.public import meal as meal_api

    monkeypatch.setattr(meal_api.settings, "meal_staff_key", "sekret")
    monkeypatch.setattr(meal_api.mv, "is_code_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(meal_api.mv, "get_code_anchor", AsyncMock(return_value=0))

    body = await meal_api.staff_code(key="sekret")
    assert body["enabled"] is True
    assert len(body["code"]) == 6
    assert 0 < body["expires_in"] <= 300


@pytest.mark.asyncio
async def test_activate_endpoint_maps_domain_error_to_400(monkeypatch):
    from app.api.public import meal as meal_api
    from fastapi import HTTPException

    monkeypatch.setattr(meal_api, "enforce_login_rate_limit", AsyncMock())
    monkeypatch.setattr(meal_api, "record_login_failure", AsyncMock())
    monkeypatch.setattr(
        meal_api.mv,
        "activate_voucher",
        AsyncMock(side_effect=mv.MealVoucherError("bad_code", "校验码错误或已过期")),
    )

    with pytest.raises(HTTPException) as exc:
        await meal_api.activate_voucher(
            meal_api.VoucherCodeRequest(code="000000"),
            FakeRequest(),
            payload={"sub": "user-1"},
        )
    assert exc.value.status_code == 400
    meal_api.record_login_failure.assert_awaited_once()


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
async def test_admin_update_rejects_bad_redeem_code(monkeypatch):
    from app.api.admin import meal as admin_meal
    from fastapi import HTTPException

    merchant = SimpleNamespace(id="m-1")
    monkeypatch.setattr(
        admin_meal,
        "db",
        SimpleNamespace(
            mealmerchant=SimpleNamespace(find_unique=AsyncMock(return_value=merchant))
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await admin_meal.update_merchant(
            "m-1", admin_meal.MerchantUpdateRequest(redeem_code="12a456")
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_admin_update_rejects_duplicate_redeem_code(monkeypatch):
    from app.api.admin import meal as admin_meal
    from fastapi import HTTPException

    mine = SimpleNamespace(id="m-1")
    other = SimpleNamespace(id="m-2")

    async def find_unique(where):
        if "id" in where:
            return mine
        return other  # redeemCode lookup hits another merchant

    monkeypatch.setattr(
        admin_meal,
        "db",
        SimpleNamespace(mealmerchant=SimpleNamespace(find_unique=find_unique)),
    )

    with pytest.raises(HTTPException) as exc:
        await admin_meal.update_merchant(
            "m-1", admin_meal.MerchantUpdateRequest(redeem_code="123456")
        )
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_admin_redemptions_detail(monkeypatch):
    from datetime import UTC, datetime

    from app.api.admin import meal as admin_meal

    merchant = SimpleNamespace(id="m-1", name="伴生宴")
    row = SimpleNamespace(
        id="v-1",
        userId="user-1",
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
        "resolve_user_displays",
        AsyncMock(return_value={"user-1": "小明"}),
    )

    body = await admin_meal.merchant_redemptions("m-1")

    assert body["merchant_name"] == "伴生宴"
    assert body["total"] == 1
    assert body["items"][0]["user_display"] == "小明"
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
    _mock_db(
        monkeypatch,
        mealvoucher=SimpleNamespace(
            find_unique=AsyncMock(return_value=_activated_voucher(days_ago=8)),
        ),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.redeem_voucher("user-1", "654321")
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
        mealmerchant=SimpleNamespace(find_first=AsyncMock(return_value=merchant)),
        mealredemptionfailure=SimpleNamespace(
            find_first=AsyncMock(return_value=None),
            create=AsyncMock(),
        ),
    )

    with pytest.raises(mv.MealVoucherError) as exc:
        await mv.redeem_voucher("user-1", "654321")
    assert exc.value.reason == "daily_cap"
    db.mealredemptionfailure.create.assert_awaited_once()


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
async def test_redeem_endpoint_structured_errors(monkeypatch):
    from fastapi import HTTPException

    from app.api.public import meal as meal_api

    monkeypatch.setattr(meal_api, "enforce_login_rate_limit", AsyncMock())
    for reason, message in [
        ("daily_cap", "今日霸王餐已被抢完啦，明天再来试试吧（记得在券的有效期内哦）"),
        ("expired", "霸王餐券已过有效期"),
    ]:
        monkeypatch.setattr(
            meal_api.mv,
            "redeem_voucher",
            AsyncMock(side_effect=mv.MealVoucherError(reason, message)),
        )
        with pytest.raises(HTTPException) as exc:
            await meal_api.redeem_voucher(
                meal_api.VoucherCodeRequest(code="123456"),
                FakeRequest(),
                payload={"sub": "user-1"},
            )
        assert exc.value.status_code == 400
        assert exc.value.detail == {"message": message, "reason": reason}


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


@pytest.mark.asyncio
async def test_generate_unique_redeem_code_retries_on_collision(monkeypatch):
    taken = SimpleNamespace(id="m-x")
    finder = AsyncMock(side_effect=[taken, taken, None])
    monkeypatch.setattr(
        mv,
        "db",
        SimpleNamespace(mealmerchant=SimpleNamespace(find_unique=finder)),
    )

    code = await mv.generate_unique_redeem_code()

    assert len(code) == 6 and code.isdigit()
    assert finder.await_count == 3
