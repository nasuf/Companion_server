"""微信支付模块的错误类型（与 Apple 侧的 payments/errors.py 平行，互不依赖）。"""

from __future__ import annotations


class WechatPayError(Exception):
    """微信支付模块的错误基类。"""


class WechatPayConfigError(WechatPayError):
    """缺少或非法的配置（密钥路径、商户号、APIv3 密钥等）。属于运维问题。"""


class WechatPaySignatureError(WechatPayError):
    """应答或回调验签失败 / 时间戳重放。一律拒绝处理。"""


class WechatPayDecryptError(WechatPayError):
    """回调报文解密失败或明文格式非预期。"""


class WechatPayUpstreamError(WechatPayError):
    """微信支付接口返回错误。``code``/``message`` 仅用于后台排查与日志。"""

    def __init__(
        self,
        detail: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        message: str | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.message = message


class WechatPayUnknownProductError(WechatPayError):
    """下单请求里的 product_id 不在服务端目录中。"""

    def __init__(self, product_id: str) -> None:
        super().__init__(f"unknown product: {product_id}")
        self.product_id = product_id


class WechatPayOpenidMissingError(WechatPayError):
    """当前账号没有服务号 openid——H5 需要先走一次网页授权。"""


class WechatPayTrialUsedError(WechatPayError):
    """¥1 体验账号级限购一次（与 Apple 渠道共用同一标记）。"""


class WechatPayOrderStateError(WechatPayError):
    """订单状态不允许当前操作（如对未支付订单发起退款）。"""
