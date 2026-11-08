"""每个 API worker 实际读到的提示词修订号, 作为提示词是否真的生效的证据 (O01.03.V01)。

后台能看到保存的内容和发布号, 但保存不等于 worker 真的在用: 聊天按每个 worker 定期刷新的
配置快照 (runtime_config._PROMPTS_CACHE) 取提示词, 刷新失败会继续用上一份, 多 worker 之间
也有传播延迟。这里每个 worker 在装载快照 (以及不走快照的直接读取) 时, 把它实际持有的
revision 记进按部署主机分桶的有界 registry; 后台把它和数据库当前 revision 并排显示:

- verified: 有 worker 正在读当前 revision;
- stale: 有 worker 还在读更旧的 revision (传播延迟/缓存漏通知);
- unverified: 没有任何 worker 证据 (没人读过, 或 Redis 不可用)。

只记 revision 和采样时刻, 不记提示词正文, 也不记密钥或上下文。记录走后台、去重: revision
有变化时立即写, 没变化时每 _REFRESH_SECONDS 才续一次; 超过 _EXPIRE_SECONDS 没续的 worker
(空闲或已退出) 从 registry 里移除, 不会被误报成滞后。配置刷新失败、沿用上一份快照的 worker
照样续报它手里的旧 revision, 这正是"保存了但没生效"要暴露的情况。
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from datetime import UTC, datetime
from uuid import uuid4

from app.redis_client import get_redis
from app.services.runtime.tasks import fire_background

logger = logging.getLogger(__name__)

_HOST = socket.gethostname()
_INCARNATION = f"{os.getpid()}:{uuid4().hex}"
_PREFIX = f"prompt:served:{_HOST}"
_LIMIT = 64  # 每台主机最多这么多 worker 身份, 防 registry 无界增长
_TTL_SECONDS = 86400
_REFRESH_SECONDS = 600
_EXPIRE_SECONDS = 1800
# 本进程上报过的 (key -> revision) 和上次写入时刻; 没变且未到续报时间就不写 Redis
_reported: dict[str, int] = {}
_last_sent = 0.0

# 一次 HSET + 过期维护 + 驱逐过期 worker; registry 满且本 worker 未登记时放弃(返回 0)
_RECORD = """
local now = redis.call('TIME')
local at = tonumber(now[1]) + tonumber(now[2])/1000000
local cutoff = at - tonumber(ARGV[3])
local rows = redis.call('HGETALL', KEYS[1])
for i=1,#rows,2 do
  if (cjson.decode(rows[i+1]).sampled_at or 0) < cutoff then redis.call('HDEL', KEYS[1], rows[i]) end
end
local existing = redis.call('HGET', KEYS[1], ARGV[1])
if not existing and redis.call('HLEN', KEYS[1]) >= tonumber(ARGV[4]) then return 0 end
local served = {}
if existing then served = cjson.decode(existing).served or {} end
for key, revision in pairs(cjson.decode(ARGV[2])) do served[key] = revision end
redis.call('HSET', KEYS[1], ARGV[1], cjson.encode({served=served, sampled_at=at}))
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[5]))
return 1
"""


def note_served(revisions: dict[str, int]) -> None:
    """本 worker 现在按这些 revision 提供提示词; 没变化时只按间隔续报。"""
    global _last_sent
    changed = {key: int(rev) for key, rev in revisions.items() if _reported.get(key) != int(rev)}
    now = time.monotonic()
    if not changed:
        if not revisions or now - _last_sent < _REFRESH_SECONDS:
            return
        changed = {key: int(rev) for key, rev in revisions.items()}
    _reported.update(changed)
    _last_sent = now
    fire_background(_record(changed))


def note_served_safely(revisions: dict[str, int]) -> None:
    """note_served for call sites on the prompt read path: evidence never breaks serving a prompt."""
    try:
        note_served(revisions)
    except Exception as exc:
        logger.debug("[PROMPT-PROPAGATION] note failed: %s", type(exc).__name__)


async def _record(revisions: dict[str, int]) -> None:
    try:
        redis = await get_redis()
        kept = await redis.eval(
            _RECORD, 1, _PREFIX, _INCARNATION, json.dumps(revisions),
            str(_EXPIRE_SECONDS), str(_LIMIT), str(_TTL_SECONDS),
        )
        if not kept:
            logger.warning("[PROMPT-PROPAGATION] registry full host=%s", _HOST)
    except Exception as exc:  # 证据写失败绝不影响提供提示词
        for key in revisions:  # 让下次装载或读取重试
            _reported.pop(key, None)
        logger.debug("[PROMPT-PROPAGATION] record failed keys=%d: %s", len(revisions), type(exc).__name__)


async def observed_revisions() -> dict[str, dict]:
    """每个 key 当前有 worker 证据的 revision -> {worker 数, 最近采样时刻}。

    key -> {revision: {"workers": n, "last_seen": iso}}; Redis 不可用时返回 {}。
    """
    try:
        redis = await get_redis()
        raw = await redis.hgetall(_PREFIX)
    except Exception as exc:
        logger.debug("[PROMPT-PROPAGATION] read failed: %s", type(exc).__name__)
        return {}
    out: dict[str, dict] = {}
    for value in list(raw.values())[:_LIMIT]:
        try:
            entry = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            continue
        seen = entry.get("sampled_at")
        iso = datetime.fromtimestamp(seen, UTC).isoformat() if isinstance(seen, (int, float)) else None
        for key, revision in (entry.get("served") or {}).items():
            bucket = out.setdefault(key, {})
            slot = bucket.setdefault(int(revision), {"workers": 0, "last_seen": None})
            slot["workers"] += 1
            if iso and (slot["last_seen"] is None or iso > slot["last_seen"]):
                slot["last_seen"] = iso
    return out


def propagation_status(current_revision: int, observed: dict | None) -> dict:
    """把一个 key 的当前 revision 和观测证据合成后台用的传播状态。"""
    observed = observed or {}
    if not observed:
        return {"propagation": "unverified", "served_revisions": [], "sampled_at": None}
    served = sorted(observed.keys(), reverse=True)
    newest = max((v["last_seen"] for v in observed.values() if v["last_seen"]), default=None)
    # A worker on an older revision means a save has not propagated. A worker on
    # current or newer (its read raced a save) counts as serving the live text.
    stale = any(int(rev) < int(current_revision) for rev in observed)
    status = "stale" if stale else "verified"
    return {
        "propagation": status,
        "served_revisions": [
            {"revision": rev, "workers": observed[rev]["workers"], "last_seen": observed[rev]["last_seen"]}
            for rev in served
        ],
        "sampled_at": newest,
    }
