"""命令行版的定时任务健康 + 数据不变量巡检.

判读逻辑不在这里, 在 app/services/ops/cron_health.py 和 invariants.py —— 后台
页面读的是同一份。命令行和页面各写一套判读, 迟早会给出不一样的结论, 而排查故障
时最不需要的就是"两个地方说法不一致"。

写这个的直接原因: L2 动态分级 cron 因为一个 SQL 类型错每晚崩了几个月, 从任何界
面都看不出来 —— 失败走 warning, 成功默认不出声, "从来没成功过"和"这次没事可做"
在日志里长得一模一样。

用法 (在生产容器内)。PYTHONPATH 不能省 —— python 把脚本自己的目录加进 sys.path,
不是工作目录, 少了它 import app.services 会失败:

    PYTHONPATH=/app python scripts/check_cron_health.py            # 读上次结果
    PYTHONPATH=/app python scripts/check_cron_health.py --recheck  # 当场重跑
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

_MARK = {
    "healthy": "✓", "ok": "✓",
    "warn": "!", "drifted": "!", "stale": "!",
    "failing": "✗", "violated": "✗", "error": "✗",
    "unknown": "·", "retired": "·",
}


def _age(stamp: str | None, now: datetime) -> str:
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


async def main(recheck: bool) -> None:
    from app.services.ops.cron_health import collect_cron_health
    from app.services.ops.invariants import load_last_report, run_and_store

    now = datetime.now(timezone.utc)

    report = await collect_cron_health(now=now)
    if not report.definitions_available:
        # 命令行进程不启动调度器, 拿不到 trigger, 也就量不出周期。把这点说明白,
        # 否则"没有 stale 项"会被读成"任务都健康", 而实际是根本没判。
        print(
            "注意: 本进程未启动调度器, 拿不到任务周期 —— 只能判「上一轮有没有失败」,\n"
            "     判不出「停跑」和「时刻偏移」。完整判读见后台的资源监控页。\n"
        )
    print(f"{'任务':<34}{'判读':<10}{'上次成功':<14}说明")
    for job in report.jobs:
        mark = _MARK.get(job.verdict, "?")
        print(
            f"{mark} {job.job_id:<32}{job.verdict:<10}"
            f"{_age(job.last_ok.isoformat() if job.last_ok else None, now):<14}"
            f"{job.detail}"
        )
    for job in report.retired:
        print(f"· {job.job_id:<32}{'retired':<10}{'':<14}{job.detail}")

    print()
    if recheck:
        from app.db import db

        await db.connect()
        try:
            await run_and_store()
        finally:
            await db.disconnect()
    payload = await load_last_report()

    checked_at = payload.get("checked_at")
    if not checked_at:
        print("数据不变量: 尚未巡检过 (每日 06:00 跑, 或加 --recheck 当场跑一次)")
    else:
        print(f"数据不变量 (巡检于 {_age(checked_at, now)}):")
        for item in payload.get("results", []):
            mark = _MARK.get(item.get("status", ""), "?")
            print(f"  {mark} {item.get('title', ''):<16}{item.get('detail', '')}")

    bad = report.unhealthy_count + int(payload.get("violated_count") or 0)
    print(f"\n需要排查的项: {bad}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recheck", action="store_true", help="当场重跑数据不变量, 而不是读上次结果"
    )
    asyncio.run(main(parser.parse_args().recheck))
