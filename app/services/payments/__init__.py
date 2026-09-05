"""支付领域（Apple IAP / 预留 google/wechat）。

Direct Apple 自建：App Store Server API 校验交易 + Server Notifications V2 +
JWS 验签，所有交易/通知落自己的库。到账复用 app.services.wallet /
app.services.vip.grants 的事务化账本，不另起一套货币。
"""
