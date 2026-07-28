"""相似度阈值与 embedding 模型是**一整套**, 不能单独改其中一个.

这个文件存在的唯一目的是防一类静默事故: 有人换了 embedding 模型 (或者反过来,
顺手调了某个阈值), 但没意识到代码里有十个常量挂在余弦相似度的尺度上. 换模型
只改 config 的话:

    检索 0.35 用回 bge-m3 的尺度 → 噪声大量涌入 prompt
    检索 0.50 用在 qwen3 的尺度 → 召回几乎全灭
    去重阈值错配              → 要么重复堆积, 要么新记忆被误并吞掉

这些都不会抛异常, 只会让 AI 悄悄变笨, 而且从日志里看不出来.

所以这里把"当前模型 + 当前阈值"这一组合钉死. 改动其中任何一项都会让测试失败,
失败信息会指向重新标定的脚本. 断言的不是"这些数字是对的" —— 而是"这些数字是
一起标出来的, 别只动一半".

标定过程见 scripts/calibrate_embedding_thresholds.py (百分位对齐)、
calibrate_paired.py (配对映射)、export_near_duplicate_pairs.py (去重端用真实
近重复对). 2026-07 换 bge-m3 → qwen3-embedding:0.6b 的实测依据写在各常量旁边.
"""

from __future__ import annotations

import pytest

from app.config import Settings, settings
from app.services.memory import config as memory_config
from app.services.memory.normalization import SIMILARITY_THRESHOLD as NORMALIZATION_CUT
from app.services.memory.retrieval import context_selector, hybrid, legacy, ranking

# 这一组是一起标定出来的, 要改就整组重标.
CALIBRATED = {
    "hybrid._SIMILARITY_THRESHOLD": (hybrid._SIMILARITY_THRESHOLD, 0.35),
    "hybrid._RELATIONSHIP_RECALL_THRESHOLD": (hybrid._RELATIONSHIP_RECALL_THRESHOLD, 0.24),
    "hybrid._ENTITY_RECALL_SIMILARITY": (hybrid._ENTITY_RECALL_SIMILARITY, 0.78),
    "legacy._L3_SIMILARITY_FLOOR": (legacy._L3_SIMILARITY_FLOOR, 0.52),
    "ranking._HIGH_SIMILARITY_THRESHOLD": (ranking._HIGH_SIMILARITY_THRESHOLD, 0.86),
    "context_selector._HIGH_SIMILARITY_THRESHOLD": (
        context_selector._HIGH_SIMILARITY_THRESHOLD, 0.86),
    "config.DEDUP_THRESHOLD": (memory_config.DEDUP_THRESHOLD, 0.85),
    "config.DELETION_SIMILARITY_THRESHOLD": (
        memory_config.DELETION_SIMILARITY_THRESHOLD, 0.85),
    "normalization.SIMILARITY_THRESHOLD": (NORMALIZATION_CUT, 0.68),
}

_RECALIBRATE = (
    "\n\n阈值与 embedding 模型是一整套. 若确实要换模型, 按 "
    "scripts/calibrate_embedding_thresholds.py 的说明重标全部十个常量, "
    "再同步更新本文件 —— 不要只改这里让测试变绿."
)


def test_code_default_matches_the_calibrated_model():
    """断言代码声明的模型, 不是运行时生效的那个 —— 后者受 .env 影响, 会让这个
    测试的结果取决于跑它的机器. env 与代码不一致是部署问题, 由 main.py 启动时
    的告警负责抓."""
    declared = Settings.model_fields["embedding_model"].default
    assert declared == memory_config.CALIBRATED_EMBEDDING_MODEL, (
        f"config.py 默认模型是 {declared}, 阈值却是按 "
        f"{memory_config.CALIBRATED_EMBEDDING_MODEL} 标定的." + _RECALIBRATE
    )


def test_startup_warns_when_env_overrides_to_an_uncalibrated_model(monkeypatch):
    """线上真正出事的形态: 代码换了模型, 但部署的环境变量还指着老模型.

    直接断言 logger 调用而不是用 caplog —— 项目给 app logger 挂了自定义 handler
    且关掉了 propagate, caplog 抓不到.
    """
    import app.main as main_module

    calls: list[tuple] = []
    monkeypatch.setattr(main_module.logger, "error", lambda *a, **kw: calls.append(a))

    monkeypatch.setattr(settings, "embedding_model", "some-other-model")
    main_module._warn_if_embedding_model_uncalibrated()
    assert calls, "模型不一致时必须告警"
    assert "MISMATCH" in calls[0][0]

    calls.clear()
    monkeypatch.setattr(
        settings, "embedding_model", memory_config.CALIBRATED_EMBEDDING_MODEL
    )
    main_module._warn_if_embedding_model_uncalibrated()
    assert not calls, "一致时不该告警"


@pytest.mark.parametrize("name", sorted(CALIBRATED))
def test_threshold_still_matches_its_calibrated_value(name):
    actual, expected = CALIBRATED[name]
    assert actual == pytest.approx(expected), (
        f"{name} 现在是 {actual}, 标定值是 {expected}." + _RECALIBRATE
    )


def test_retrieval_gate_sits_below_the_dedup_gate():
    """检索门必须远低于去重门. 两者若接近, 说明尺度标错了 —— 能被召回的东西
    不该顺手被判成重复."""
    assert hybrid._SIMILARITY_THRESHOLD < memory_config.DEDUP_THRESHOLD - 0.2


def test_deletion_matches_dedup():
    """删除匹配与去重共用一把尺子, 历史上二者拉开差值造成过误删/漏删,
    现在刻意保持一致 (见 memory/config.py 注释)."""
    assert (memory_config.DELETION_SIMILARITY_THRESHOLD
            == memory_config.DEDUP_THRESHOLD)


def test_l3_floor_sits_above_the_main_gate():
    """L3 是久远模糊记忆, 门槛要比常规检索**高** —— 唤醒它需要更强的证据,
    否则每轮都会掺进陈年往事."""
    assert legacy._L3_SIMILARITY_FLOOR > hybrid._SIMILARITY_THRESHOLD
