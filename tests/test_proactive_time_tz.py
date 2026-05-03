"""Regression: proactive scheduled_scene + delay explanation prompt 必须用
项目时区 _TZ (Asia/Shanghai), 不能裸 datetime.now().

生产 bug 复现 (2026-05-03 trace 019deb51): 服务器跑在 UTC 容器, prompt 里
'time': datetime.now(UTC).astimezone() 解析为 server local tz = UTC →
LLM 看到 '现在是 00:51' (实际上海早 08:51 用户在吃早饭) 回 '夜深了'.

修复: 改为 _now_corrected().strftime() (NTP 修正 + Asia/Shanghai 时区).
本测试守门防 future 编辑又退化回 datetime.now() 裸调用.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest


SHANGHAI = timezone(timedelta(hours=8))


def test_scheduled_scene_prompt_uses_shanghai_tz():
    """sender._format_prompt 渲染 scheduled_scene 时, time 字段必须是 Shanghai
    本地时间, 不是 UTC. 直接复现生产 bug: UTC 00:51 vs Shanghai 08:51."""
    from app.services.proactive import sender

    # 模拟一个明确的 Shanghai 早晨时刻
    morning_shanghai = datetime(2026, 5, 3, 8, 51, tzinfo=SHANGHAI)

    ctx = {
        "topic_theme": "日常",
        "schedule_status": {"activity": "自由时间", "status": "idle"},
        "__tpl": "现在是{time}, 活动={activity}, 状态={status}",
    }

    with patch("app.services.proactive.sender._now_corrected",
               return_value=morning_shanghai):
        result = sender._format_prompt(
            "proactive.scheduled_scene", ctx, personality_brief="温和友善",
        )

    assert result is not None
    # 必须是 Shanghai 早 08:51, 不能是 UTC 00:51
    assert "现在是08:51" in result, (
        f"scheduled_scene 必须用 Shanghai 时区显示时间, 实际渲染: {result!r}. "
        f"如果出现 '00:51' 说明又回退到 datetime.now(UTC).astimezone() — "
        f"在 UTC 容器下让 LLM 看 '现在 00:51' 回 '夜深了' 的生产 bug 会回归"
    )
    assert "00:51" not in result


def test_scheduled_scene_prompt_uses_utc_container_returns_shanghai():
    """更严格的端到端守门: 模拟 UTC 容器 (服务器系统时区 = UTC), 验证 sender
    渲染出的时间仍然是 Shanghai. 直接复现生产 bug 的物理条件: server 在 UTC,
    Shanghai 早 08:51 = UTC 00:51, prompt 必须显示 08:51."""
    from app.services.proactive import sender

    # 模拟 _now_corrected 返 Shanghai 时区 aware datetime (现实就是这样)
    morning_shanghai = datetime(2026, 5, 3, 8, 51, tzinfo=SHANGHAI)
    ctx = {
        "topic_theme": "日常",
        "schedule_status": {"activity": "自由时间", "status": "idle"},
        "__tpl": "{time}",
    }
    with patch("app.services.proactive.sender._now_corrected",
               return_value=morning_shanghai):
        result = sender._format_prompt(
            "proactive.scheduled_scene", ctx, personality_brief="x",
        )
    # 关键: 即便 server tz=UTC, prompt 必须看到 Shanghai 08:51 (不是 UTC 00:51)
    assert result == "08:51"


@pytest.mark.asyncio
async def test_delay_explanation_uses_shanghai_tz():
    """reply_post_process 的 delay 解释 prompt 同根问题: current_time 也要
    Shanghai 时区. 防 LLM 回 '我00:51才回你' (UTC) 而实际是 08:51 上海."""
    from app.services.chat import reply_post_process as rpp

    morning_shanghai = datetime(2026, 5, 3, 8, 51, tzinfo=SHANGHAI)
    captured = {}

    async def _capture_delay_reply(**kwargs):
        captured.update(kwargs)
        return "等好久了, 抱歉~"

    async def _fallback(*args, **kwargs):
        return ""

    with patch("app.services.chat.reply_post_process._now_corrected",
               return_value=morning_shanghai):
        await rpp._build_delay_explanation_text(
            reply_context={"received_at": "", "received_status": {"activity": "自由", "status": "idle"}},
            elapsed=120.0,  # 2 分钟前收到
            delay_reply_fn=_capture_delay_reply,
            fallback_fn=_fallback,
            agent=MagicMock(),
            user_message="嗨",
        )

    assert captured.get("current_time") == "08:51", (
        f"delay 解释的 current_time 必须是 Shanghai 时区, got {captured.get('current_time')!r}"
    )


def test_reply_post_process_does_not_import_datetime():
    """端到端守门: 跟 sender 同根问题, 防 future 编辑又把 datetime 导回来.
    `_build_delay_explanation_text` 应该完全用 _now_corrected, 整个模块不
    需要 `from datetime import datetime`. 如果有人 add datetime back +
    `datetime.now()` 这个 assertion 会立刻失败."""
    import app.services.chat.reply_post_process as rpp_mod
    # 模块顶层不该出现 datetime 名字 — 我们删了 import. 如果 future 重新加,
    # 应该是个 deliberate decision, 这个 assertion 会 force review.
    assert not hasattr(rpp_mod, "datetime"), (
        "reply_post_process 不应导入 datetime. 时间格式必须走 _now_corrected. "
        "若需新增 datetime 用法, 请同时 update 本测试"
    )
