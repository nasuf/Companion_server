"""SMS verification-code subsystem (Tencent Cloud SMS).

* ``tencent``  — minimal TC3-HMAC-SHA256 signed SendSms call (httpx, no SDK dep)
* ``service``  — code lifecycle: generation, Redis storage, rate limits, verify
"""

from app.services.sms.service import (
    SmsRateLimited,
    normalize_cn_phone,
    send_login_code,
    verify_code,
)
from app.services.sms.tencent import SmsSendError

__all__ = [
    "SmsRateLimited",
    "SmsSendError",
    "normalize_cn_phone",
    "send_login_code",
    "verify_code",
]
