"""支付领域异常。API 层按类型映射到 HTTP 状态（仿现有 ValueError→HTTPException）。"""

from __future__ import annotations


class PaymentError(Exception):
    """支付领域基类。"""


class UnknownProductError(PaymentError):
    """product_id 不在服务端 catalog 中（客户端与后端商品表脱节）。"""

    def __init__(self, product_id: str):
        super().__init__(f"unknown_product:{product_id}")
        self.product_id = product_id


class AppleVerificationError(PaymentError):
    """Apple 侧校验失败：JWS 验签不通过、或 App Store Server API 拒绝。"""


class TransactionNotFoundError(AppleVerificationError):
    """transactionId 在 production 与 sandbox 都查不到（伪造 / 尚未同步）。"""
