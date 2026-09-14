"""prisma-client 版本对齐守卫 —— 防"admin UI 打开的开关某些 worker 静默不生效".

我 (Claude) 本机复现过一次: schema.prisma 加了 proactiveTrendingEnabled 列,
DB 有 True, 但 prisma client 没重新生成, `getattr(sys_row, 'proactiveTrendingEnabled',
None)` 返 None → _row_to_dict 跳过 → _pick 走 env 兜底 (False). 全程零告警.
多 worker 部署时会出"部分进程按 admin 值、部分按 env 兜底"的一致性错乱.

守卫: load_caches() 加载 sys_row 后 hasattr 一遍所有 required 字段, 缺则 log
ERROR + 打 EVT_RUNTIME_CONFIG_STALE_CLIENT (dashboard 可聚合). 一进程告警一次.
"""

from __future__ import annotations

import logging

from types import SimpleNamespace

from app.services import runtime_config


class TestStaleClientGuard:
    def _reset_verified(self):
        # 守卫用了 module-level flag 只告警一次, 每 test 重置
        runtime_config._STALE_CLIENT_VERIFIED = False

    def test_fires_when_field_missing(self, caplog):
        self._reset_verified()
        # 模拟 stale prisma client: 缺 proactiveTrendingEnabled
        row = SimpleNamespace(
            onlineModel=True, remoteProvider="dashscope",
            remoteChatProvider="dashscope", remoteSmallProvider="dashscope",
            localChatModel="qwen2.5:14b", localSmallModel="qwen2.5:7b",
            remoteChatModel="qwen3.5-plus", remoteSmallModel="qwen3.5-flash",
            visionModel="v", asrModel="a", ttsModel="t", ttsOutputProbability=0,
            webSearchEnabled=False,
            # proactiveTrending* 全部缺
            replyDelayEnabled=False, replyDelayMaxSeconds=300,
            userMessageAggregationEnabled=True,
        )
        with caplog.at_level(logging.ERROR, logger="app.services.runtime_config"):
            runtime_config._verify_prisma_client_fields(row)
        assert any("prisma client stale" in r.message for r in caplog.records), \
            "缺字段时应 log ERROR"
        # 检查 event 字段填了
        rec = next(r for r in caplog.records if "stale" in r.message)
        assert getattr(rec, "event", None) == "runtime_config.stale_client"
        assert getattr(rec, "missing_count", 0) == 4  # 4 个 proactiveTrending* 缺

    def test_silent_when_all_fields_present(self, caplog):
        self._reset_verified()
        row = SimpleNamespace(**{k: True for k in runtime_config._SYSTEM_CONFIG_REQUIRED_FIELDS})
        with caplog.at_level(logging.WARNING, logger="app.services.runtime_config"):
            runtime_config._verify_prisma_client_fields(row)
        assert not any("prisma client stale" in r.message for r in caplog.records), \
            "所有字段齐全时不应告警"

    def test_only_warns_once_per_process(self, caplog):
        self._reset_verified()
        # 缺字段
        row = SimpleNamespace(onlineModel=True)
        with caplog.at_level(logging.ERROR, logger="app.services.runtime_config"):
            runtime_config._verify_prisma_client_fields(row)
            runtime_config._verify_prisma_client_fields(row)  # 再次调用
            runtime_config._verify_prisma_client_fields(row)  # 第三次
        stale_records = [r for r in caplog.records if "prisma client stale" in r.message]
        assert len(stale_records) == 1, \
            f"一进程应只告警一次防刷屏, 实际 {len(stale_records)} 次"

    def test_handles_none_row(self):
        # sys_row 是 None 时不该炸
        self._reset_verified()
        runtime_config._verify_prisma_client_fields(None)  # 不 raise 即可

    def test_row_to_dict_still_works_with_stale_row(self):
        # 守卫只 log 告警, 不 raise —— _row_to_dict 必须继续跑
        self._reset_verified()
        row = SimpleNamespace(onlineModel=True, remoteProvider="dashscope")
        out = runtime_config._row_to_dict(row)
        # 只有 hasattr True 的字段进 dict
        assert out == {"onlineModel": True, "remoteProvider": "dashscope"}


class TestRequiredFieldsContract:
    def test_all_resolved_config_fields_covered(self):
        """ResolvedConfig 每个非 provider 字段都必须在 required 里列出 —— 保证守卫
        实际检查的字段跟 resolve_config_sync 会读的字段一致."""
        import dataclasses
        # ResolvedConfig 字段 → 期望的 SystemConfig 列名 (camelCase)
        # 简单规则: snake_case → camelCase; 已知例外单独列
        rc_fields = {f.name for f in dataclasses.fields(runtime_config.ResolvedConfig)}
        required_snake = set()
        for camel in runtime_config._SYSTEM_CONFIG_REQUIRED_FIELDS:
            # camelCase → snake_case: 简单转
            snake = "".join(
                "_" + c.lower() if c.isupper() else c for c in camel
            ).lstrip("_")
            required_snake.add(snake)

        # remote_provider fallback + provider fields 不 1:1, 单独 allowlist:
        NOT_IN_REQUIRED = {"remote_provider"}  # remoteProvider 是, 只是命名有点绕
        should_be_required = rc_fields - NOT_IN_REQUIRED
        # 每个 ResolvedConfig 字段都应有一个对应的 required column
        # (允许 required 比 rc 多, 因为可能有 "TTL" 类未来字段)
        missing = should_be_required - required_snake - NOT_IN_REQUIRED
        # 允许 provider 命名不对齐 (已在 NOT_IN_REQUIRED)
        assert not missing, (
            f"ResolvedConfig 有字段 {missing} 未列入 _SYSTEM_CONFIG_REQUIRED_FIELDS. "
            f"加新 config 字段时守卫会漏检 → admin UI 静默失效风险"
        )
