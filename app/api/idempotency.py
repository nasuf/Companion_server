"""防重复提交: 短窗口内同一用户的同一份创建请求只生效一次.

要防的是双击、网络重试、以及移动端在弱网下的自动重发 —— 这些都会在几秒内发出内容
完全相同的两个请求, 结果是用户列表里凭空多出一条。多 worker 之后两个请求会真正并行,
窗口从"事件循环交错"变成"真并行", 更容易撞上。

刻意没有采用客户端传 `Idempotency-Key` 的标准做法: 那要求 web / flutter / 小程序
三端同时改造, 而真实故障形态是"同一份内容连发两次", 服务端按内容指纹去重就能覆盖,
且对客户端零改动。代价是无法区分"用户真的想连建两条一模一样的记录" —— 窗口取十几
秒正是为了让这种意图仍然可行。

依赖 Redis SET NX 的原子性。Redis 不可用时**放行**: 这层是体验保护而非正确性保护,
为它挡掉用户的正常创建是本末倒置。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# 双击间隔通常 < 1s, 弱网自动重试一般在几秒内。15 秒足够覆盖, 又不会长到把用户
# "确实想再建一条同样的"挡在门外。
DEFAULT_WINDOW_S = 15

_KEY_PREFIX = "idem:submit:"


def fingerprint(payload: Any) -> str:
    """把请求体归一成稳定指纹.

    sort_keys 是必须的 —— 客户端两次序列化的字段顺序可能不同, 不排序会让同一份
    内容算出两个指纹, 去重直接失效。
    """
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass
class SubmissionGuard:
    """一次创建请求的去重结果.

    `duplicate_of` 非空说明这是重复提交, 里面是首次那一次创建出的资源 id ——
    调用方应当返回那个既有资源, 而不是报错。对用户来说"点了两下拿到同一条",
    比"第二下报错"体感好得多, 也免得客户端要处理一个新的错误分支。
    """

    key: str
    duplicate_of: str | None = None
    _claimed: bool = False

    @property
    def is_duplicate(self) -> bool:
        return self.duplicate_of is not None

    async def record(self, resource_id: str) -> None:
        """把创建出来的资源 id 写回, 供窗口内的重复请求取用."""
        if not self._claimed:
            return
        try:
            from app.redis_client import get_redis

            redis = await get_redis()
            await redis.set(self.key, resource_id, ex=DEFAULT_WINDOW_S)
        except Exception as exc:
            logger.warning(f"idempotency: record failed for {self.key}: {exc}")

    async def release(self) -> None:
        """创建失败时撤销占位, 让用户能立刻重试.

        不撤的话, 用户在窗口内重试会被判成"重复"并拿到一个空的 duplicate_of ——
        既没创建成功, 又不让重试, 是最糟的组合。
        """
        if not self._claimed:
            return
        try:
            from app.redis_client import get_redis

            redis = await get_redis()
            await redis.delete(self.key)
        except Exception as exc:
            logger.warning(f"idempotency: release failed for {self.key}: {exc}")


async def claim_submission(
    scope: str,
    user_id: str,
    payload: Any,
    *,
    window_s: int = DEFAULT_WINDOW_S,
) -> SubmissionGuard:
    """占坑。返回的 guard 若 is_duplicate 为真, 说明窗口内已有同样的提交。

    占位值先写 "-"(表示"创建进行中"), 成功后由 record 换成真实资源 id。这样并发的
    第二个请求会看到 "-" 而不是空 —— 它知道有人正在创建, 但拿不到 id, 此时按重复
    处理并返回 409 比返回一个半成品更诚实。
    """
    key = f"{_KEY_PREFIX}{scope}:{user_id}:{fingerprint(payload)}"
    try:
        from app.redis_client import get_redis

        redis = await get_redis()
        claimed = await redis.set(key, "-", nx=True, ex=window_s)
        if claimed:
            return SubmissionGuard(key=key, _claimed=True)
        existing = await redis.get(key)
        prior = existing.decode() if isinstance(existing, (bytes, bytearray)) else existing
        return SubmissionGuard(key=key, duplicate_of=prior or "-")
    except Exception as exc:
        # Redis 挂了就放行。这层保护的是体验, 不是正确性 —— 为它拦掉用户的正常创建
        # 是本末倒置。
        logger.warning(f"idempotency: claim failed for {scope}, allowing: {exc}")
        return SubmissionGuard(key=key)
