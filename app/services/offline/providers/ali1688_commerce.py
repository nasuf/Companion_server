from __future__ import annotations

import logging
from typing import Any

from app.services.offline.providers.ali1688_client import Ali1688Client
from app.services.offline.providers.ali1688_token import get_access_token
from app.services.offline.providers.gift_types import (
    GiftOrderResult,
    GiftProductCandidate,
    GiftProviderError,
    RecipientAddress,
)

logger = logging.getLogger(__name__)

# ⚠️ 接口名以官方文档为准，这里给的是按公开资料核实到的默认值，可被 settings 覆盖。
_SEARCH_NAMESPACE = "com.alibaba.product"
_SEARCH_API = "alibaba.product.search"            # 关键词搜商品
_PREVIEW_NAMESPACE = "com.alibaba.trade"
_PREVIEW_API = "alibaba.createOrder.preview"      # 下单预览：算价/运费/地址校验
_PAYCHECK_API = "alibaba.trade.payment.check"     # 检测是否已开通免密代扣
_CREATE_API = "alibaba.trade.createCrossOrder"    # 创建订单（支持单品/跨店）

# 召回后返回给上层做 LLM 精选的候选数（与 gift_selection._LLM_PICK_TOP_K 配套：
# 留足候选让精选模型挑，多于 top-K 的部分由精选阶段裁掉）。
_SELECTION_SHORTLIST = 8


class Ali1688GiftCommerceProvider:
    """1688 采购下单 provider。

    search_products: 关键词召回 → 硬过滤（预算/起订量/一件代发/库存）→ 质量粗排。
    返回的候选已是「预算内、可一件代发、按质量排序」的干净列表；语义精选交给
    上层 gift_selection.select_best_candidate（LLM 复核）。

    place_order: 预览(算价+地址) → 免密代扣检测 → 创建订单（免密自动付款）。
    """

    name = "ali1688"

    def __init__(
        self,
        *,
        app_key: str,
        app_secret: str,
        access_token: str,
        timeout_s: float = 12.0,
        recall_size: int = 40,
        require_one_piece: bool = True,
    ) -> None:
        self._client = Ali1688Client(
            app_key=app_key,
            app_secret=app_secret,
            access_token=access_token,
            access_token_getter=get_access_token,
            timeout_s=timeout_s,
        )
        self._recall_size = max(10, recall_size)
        self._require_one_piece = require_one_piece

    async def search_products(
        self,
        *,
        query: str,
        min_amount_cents: int,
        max_amount_cents: int,
        limit: int = 5,
    ) -> list[GiftProductCandidate]:
        data = await self._client.call(
            namespace=_SEARCH_NAMESPACE,
            api_name=_SEARCH_API,
            biz_params={
                "keywords": query,
                "priceStart": f"{min_amount_cents / 100:.2f}",
                "priceEnd": f"{max_amount_cents / 100:.2f}",
                "pageSize": self._recall_size,
                "page": 1,
                # 优先综合排序；按文档可调成 "monthSold"/"price" 等
                "sortType": "综合",
            },
        )
        raw_items = (
            data.get("result")
            or data.get("products")
            or data.get("productList")
            or []
        )
        if isinstance(raw_items, dict):
            raw_items = raw_items.get("items") or raw_items.get("list") or []
        if not isinstance(raw_items, list):
            raise GiftProviderError("1688 搜索返回结构无法解析为商品列表")

        candidates: list[GiftProductCandidate] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            cand = _candidate_from_item(item)
            if cand is None:
                continue
            if not _passes_hard_filter(
                cand,
                min_amount_cents=min_amount_cents,
                max_amount_cents=max_amount_cents,
                require_one_piece=self._require_one_piece,
            ):
                continue
            candidates.append(cand)

        candidates.sort(key=_coarse_quality_score, reverse=True)
        logger.info(
            "[ali1688] search query=%r recall=%d kept=%d",
            query,
            len(raw_items),
            len(candidates),
        )
        return candidates[: max(limit, _SELECTION_SHORTLIST)]

    async def place_order(
        self,
        *,
        candidate: GiftProductCandidate,
        address: RecipientAddress,
        idempotency_key: str,
    ) -> GiftOrderResult:
        offer_id = candidate.external_product_id
        spec_id = str(candidate.raw.get("specId") or candidate.raw.get("skuId") or "")
        cargo = [{"offerId": offer_id, "specId": spec_id, "quantity": 1}] if spec_id else [
            {"offerId": offer_id, "quantity": 1}
        ]
        address_param = _address_param(address)

        # 1) 预览：拿运费 + 校验地址 + 取下单所需参数
        await self._client.call(
            namespace=_PREVIEW_NAMESPACE,
            api_name=_PREVIEW_API,
            biz_params={"flow": "general", "cargoParamList": cargo, "addressParam": address_param},
        )

        # 2) 免密代扣检测：未开通则报错并带授权链接，运营去签「先采后付/免密支付协议」
        pay_check = await self._client.call(
            namespace=_PREVIEW_NAMESPACE,
            api_name=_PAYCHECK_API,
            biz_params={"flow": "general"},
        )
        if not _passwordless_enabled(pay_check):
            auth_url = pay_check.get("authUrl") or pay_check.get("auth_url") or ""
            raise GiftProviderError(
                f"1688 企业账号未开通免密代扣（先采后付），无法自动付款。授权链接: {auth_url}"
            )

        # 3) 创建订单（免密自动付款）。outOrderId 用 gift_id 做幂等键。
        order = await self._client.call(
            namespace=_PREVIEW_NAMESPACE,
            api_name=_CREATE_API,
            biz_params={
                "flow": "general",
                "outOrderId": idempotency_key,
                "message": "",
                "cargoParamList": cargo,
                "addressParam": address_param,
                "payChannel": "alipay",  # 走免密代扣自动付款
            },
        )
        order_id = str(
            order.get("orderId")
            or order.get("order_id")
            or (order.get("result") or {}).get("orderId")
            or ""
        )
        if not order_id:
            raise GiftProviderError("1688 创建订单成功但未返回 orderId")

        return GiftOrderResult(
            provider=self.name,
            provider_order_id=order_id,
            status="ordered",
            paid_amount_cents=_amount_to_cents(order.get("totalSuccessAmount"), candidate.price_cents),
            product_image_url=candidate.image_url,
            tracking_number=None,  # 物流单号下单后才生成，由 logistics provider 拉取
            shipped_at=None,
            delivered_at=None,
            raw=order,
        )


def _candidate_from_item(item: dict[str, Any]) -> GiftProductCandidate | None:
    offer_id = str(
        item.get("offerId") or item.get("productId") or item.get("id") or ""
    )
    title = str(item.get("subject") or item.get("title") or item.get("name") or "")
    price_cents = _price_cents(item)
    if not offer_id or not title or price_cents <= 0:
        return None
    return GiftProductCandidate(
        external_product_id=offer_id,
        title=title[:120],
        price_cents=price_cents,
        image_url=item.get("imageUrl") or item.get("image") or item.get("picUrl"),
        product_url=item.get("detailUrl") or item.get("productUrl"),
        shop_name=item.get("companyName") or item.get("sellerName") or item.get("shopName"),
        source="ali1688",
        raw={
            # 给筛选层用的质量/可售信号，全部塞进 raw，避免改 schema
            "moq": _to_int(item.get("minOrderQuantity") or item.get("moq"), 1),
            "sold": _to_int(item.get("monthSold") or item.get("saledCount") or item.get("sale"), 0),
            "repurchase_rate": _to_float(item.get("repurchaseRate"), 0.0),
            "support_one_piece": bool(
                item.get("supportOnePiece")
                or item.get("isOnePiece")
                or item.get("mixWholesale")
            ),
            "stock": _to_int(item.get("amountOnSale") or item.get("stock"), 1),
            "tp_member": bool(item.get("tpMember") or item.get("isTp")),  # 实力商家
            "cert_years": _to_int(item.get("tradeMedalLevel") or item.get("memberDays"), 0),
            "specId": item.get("specId") or item.get("skuId"),
            "_origin": item,
        },
    )


def _passes_hard_filter(
    cand: GiftProductCandidate,
    *,
    min_amount_cents: int,
    max_amount_cents: int,
    require_one_piece: bool,
) -> bool:
    if not (min_amount_cents <= cand.price_cents <= max_amount_cents):
        return False
    raw = cand.raw
    if raw.get("stock", 1) <= 0:
        return False
    # 1688 是批发：起订量>1 的商品送一件礼物不可行，要求支持一件代发或 moq<=1
    moq = raw.get("moq", 1)
    one_piece = bool(raw.get("support_one_piece"))
    if require_one_piece and moq > 1 and not one_piece:
        return False
    # 标题级排除：批发包/赠品/样品/配件等明显不适合做礼物的词
    if any(bad in cand.title for bad in _TITLE_BLOCKLIST):
        return False
    return True


_TITLE_BLOCKLIST = (
    "批发", "一批", "整箱", "整件", "称重", "散装", "赠品", "样品", "清仓尾货",
    "配件", "替换装", "仅拍", "下单备注",
)


def _coarse_quality_score(cand: GiftProductCandidate) -> float:
    raw = cand.raw
    score = 0.0
    score += min(raw.get("sold", 0), 5000) / 5000 * 4.0          # 销量（封顶）
    score += min(raw.get("repurchase_rate", 0.0), 0.5) / 0.5 * 2.0  # 复购率
    score += 1.5 if raw.get("support_one_piece") else 0.0        # 支持一件代发
    score += 1.0 if raw.get("tp_member") else 0.0                # 实力商家
    score += min(raw.get("cert_years", 0), 6) / 6 * 1.0          # 诚信通年限
    score += 0.5 if cand.image_url else 0.0
    return score


def _passwordless_enabled(pay_check: dict[str, Any]) -> bool:
    # 只认明确表达「已开通免密代扣」的字段。不拿 success/result（它们通常只表示
    # 接口调用成功，与授权状态无关）当判据，否则会误判为已授权而触发真实扣款。
    # 具体字段名以官方文档为准；缺这些字段时保守返回 False（宁可拦下也不误扣）。
    for key in ("hasPaymentAuth", "passwordFree", "paymentAuth", "authorized"):
        value = pay_check.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"}:
            return True
    return False


def _address_param(address: RecipientAddress) -> dict[str, str]:
    return {
        "fullName": address.recipient_name,
        "mobile": address.phone,
        "phone": address.phone,
        "provinceText": address.province,
        "cityText": address.city,
        "areaText": address.district,
        "address": address.detail,
    }


def _price_cents(item: dict[str, Any]) -> int:
    # 1688 价格可能是元(float/str) 或区间，尽量取一个可下单单价
    for key in ("price", "consignPrice", "sellPrice", "promotionPrice"):
        value = item.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            value = value[0] if value else None
        try:
            return round(float(value) * 100)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return 0


def _amount_to_cents(value: Any, fallback_cents: int) -> int:
    """把 1688 金额字段转成「分」。

    ⚠️ 1688 不同交易接口的金额单位不统一（有的「元」有的「分」），上线前务必按
    官方文档确认 totalSuccessAmount 的单位。这里做防御性判断：按「元」换算(×100)
    后若远超候选单价（>10 倍），判定原值本就是「分」，直接取整，避免金额放大 100 倍。
    """
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return fallback_cents
    as_cents_from_yuan = round(raw * 100)
    if fallback_cents > 0 and as_cents_from_yuan > fallback_cents * 10:
        return int(raw)  # 原值疑似已是「分」
    return as_cents_from_yuan


def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
