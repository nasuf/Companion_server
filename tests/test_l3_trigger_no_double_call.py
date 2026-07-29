"""L3 唤醒判断不该为了兼容旧格式而重复调用大模型.

`l3_trigger_analyze` 先按 JSON 格式问一次, 读不懂时再按旧的「裸标签」格式问一次
—— 后者是为了兼容 web 端可能残留的自定义提示词。

原来的重试条件是「结果为无」, 而"无"恰恰是绝大多数对话的正常答案 (多数消息不需要
唤醒久远记忆)。生产实测: 10 轮里 9 轮都调了不止一次, 每次约 0.37s 挂在用户等待的
关键路径上。

正确的条件是「没读懂」: 解析器现在用 recognised 区分「模型明确说无」和「输出没解析
出来兜底成无」。
"""

from __future__ import annotations

import pytest

from app.services.chat import intent_replies
from app.services.chat.intent_replies import (
    L3TriggerResult,
    _parse_l3_trigger_result,
    l3_trigger_analyze,
)


class TestParse:
    def test_explicit_none_is_recognised(self):
        r = _parse_l3_trigger_result({"label": "无"})
        assert r.label == "无" and r.recognised

    def test_valid_label_with_query_is_recognised(self):
        r = _parse_l3_trigger_result({"label": "请求更久", "retrieval_query": "老家的事"})
        assert (r.label, r.retrieval_query, r.recognised) == ("请求更久", "老家的事", True)

    def test_bare_label_string_is_recognised(self):
        """旧格式直接吐标签, 也算读懂了 —— 不该再重试一次."""
        r = _parse_l3_trigger_result("不满纠正")
        assert r.label == "不满纠正" and r.recognised

    def test_garbage_is_not_recognised(self):
        r = _parse_l3_trigger_result("这不是一个有效输出")
        assert r.label == "无" and not r.recognised

    def test_unknown_label_is_not_recognised(self):
        r = _parse_l3_trigger_result({"label": "莫名其妙的标签"})
        assert r.label == "无" and not r.recognised

    def test_none_label_drops_the_query(self):
        """判定为无时不该带检索 query —— 下游会拿它去做无谓的向量检索."""
        r = _parse_l3_trigger_result({"label": "无", "retrieval_query": "残留"})
        assert r.retrieval_query == ""

    def test_overlong_query_is_truncated(self):
        r = _parse_l3_trigger_result({"label": "请求更久", "retrieval_query": "长" * 80})
        assert len(r.retrieval_query) == 50


class TestNoDoubleCall:
    @pytest.mark.asyncio
    async def test_explicit_none_does_not_trigger_a_second_call(self, monkeypatch):
        """最重要的一条: 模型说"无"就到此为止.

        这是 90% 的对话会走的分支, 多一次调用就是 0.37s 白等。
        """
        calls: list[str] = []

        async def _render(key, params, fn):
            calls.append(key)
            return {"label": "无"}

        async def _classify(key, params, labels):
            calls.append(f"{key}:compat")
            return "无"

        monkeypatch.setattr(intent_replies, "render_prompt", _render)
        monkeypatch.setattr(intent_replies, "_classify_label", _classify)

        result = await l3_trigger_analyze("今天天气不错")

        assert result.label == "无"
        assert calls == ["memory.l3_trigger"], (
            f"调用了 {len(calls)} 次 ({calls}) —— 模型已经明确回答, 不该再问一遍"
        )

    @pytest.mark.asyncio
    async def test_unparseable_output_still_falls_back(self, monkeypatch):
        """兼容路径要保住: 老格式的自定义提示词仍要能用."""
        calls: list[str] = []

        async def _render(key, params, fn):
            calls.append(key)
            return "服务器开小差了"

        async def _classify(key, params, labels):
            calls.append(f"{key}:compat")
            return "请求更久"

        monkeypatch.setattr(intent_replies, "render_prompt", _render)
        monkeypatch.setattr(intent_replies, "_classify_label", _classify)

        result = await l3_trigger_analyze("再跟我说说小时候")

        assert result.label == "请求更久"
        assert calls == ["memory.l3_trigger", "memory.l3_trigger:compat"]

    @pytest.mark.asyncio
    async def test_positive_verdict_does_not_trigger_a_second_call(self, monkeypatch):
        calls: list[str] = []

        async def _render(key, params, fn):
            calls.append(key)
            return {"label": "不满纠正", "retrieval_query": "上次说的地址"}

        async def _classify(key, params, labels):
            calls.append(f"{key}:compat")
            return "无"

        monkeypatch.setattr(intent_replies, "render_prompt", _render)
        monkeypatch.setattr(intent_replies, "_classify_label", _classify)

        result = await l3_trigger_analyze("不对吧，我之前说的不是这个")

        assert result == L3TriggerResult(
            label="不满纠正", retrieval_query="上次说的地址", recognised=True,
        )
        assert calls == ["memory.l3_trigger"]

    @pytest.mark.asyncio
    async def test_compat_path_returning_none_keeps_the_original_result(self, monkeypatch):
        """兼容路径也答不上来时, 保持"无", 不要抛."""

        async def _render(key, params, fn):
            return "看不懂的东西"

        async def _classify(key, params, labels):
            return None

        monkeypatch.setattr(intent_replies, "render_prompt", _render)
        monkeypatch.setattr(intent_replies, "_classify_label", _classify)

        assert (await l3_trigger_analyze("随便说说")).label == "无"
