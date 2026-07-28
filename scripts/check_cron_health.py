"""列出每个定时任务最近一次成功/失败的时刻, 找出已经死掉的那些.

写这个脚本的直接原因: L2 动态分级 cron 因为一个 SQL 类型错每晚崩了几个月, 而
从任何一个界面上都看不出来 —— 失败走 warning, 成功默认不出声, 于是"从来没成功
过"和"这次没事可做"完全无法区分. 巡检一遍 26 个任务发现同样的 swallow-and-warn
写法有 14 处, 每一处都可能藏着同样的故障.

判读方式:
    从未成功        该任务大概率一直是坏的 (或者从没到过触发时刻)
    fail_at 比 ok_at 新   上一轮跑失败了
    ok_at 远早于它的周期  排程没触发, 或者实例一直在它的时刻前重启

注意"从未成功"对低频任务会误报: 周任务如果服务近期才部署, 可能确实还没到点.
对照 add_job 里的周期看, 不要只看这一列.

用法 (在生产容器内):
    python check_cron_health.py
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.redis_client import get_redis

HEALTH_KEY = "scheduler:health"


async def main() -> None:
    redis = await get_redis()
    raw = await redis.hgetall(HEALTH_KEY)
    if not raw:
        print("没有任何记录 —— 要么调度器还没跑过一轮, 要么这个版本早于健康记录。")
        return

    def _text(v) -> str:
        return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)

    jobs: dict[str, dict[str, str]] = {}
    for key, value in raw.items():
        name, _, field = _text(key).rpartition(":")
        jobs.setdefault(name, {})[field] = _text(value)

    now = datetime.now(timezone.utc)

    def _age(stamp: str | None) -> str:
        if not stamp:
            return "从未"
        try:
            delta = now - datetime.fromisoformat(stamp)
        except ValueError:
            return stamp
        hours = delta.total_seconds() / 3600
        if hours < 1:
            return f"{int(delta.total_seconds() / 60)} 分钟前"
        if hours < 48:
            return f"{hours:.1f} 小时前"
        return f"{hours / 24:.1f} 天前"

    print(f"{'任务':<32}{'上次成功':<16}{'上次失败':<16}")
    suspects: list[tuple[str, str]] = []
    for name in sorted(jobs):
        ok_at = jobs[name].get("ok_at")
        fail_at = jobs[name].get("fail_at")
        print(f"{name:<32}{_age(ok_at):<16}{_age(fail_at):<16}")
        if not ok_at:
            suspects.append((name, "从未成功过"))
        elif fail_at and fail_at > ok_at:
            reason = jobs[name].get("fail_reason", "")
            suspects.append((name, f"最近一轮失败: {reason[:80]}"))

    if suspects:
        print("\n需要排查:")
        for name, why in suspects:
            print(f"  {name}: {why}")
    else:
        print("\n所有有记录的任务最近一轮都成功了。")


if __name__ == "__main__":
    asyncio.run(main())
