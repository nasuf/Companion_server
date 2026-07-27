"""抽取 prompt 声明的 importance 区间必须跟 pipeline 的分层阈值对得上.

生产 2026-07-27: 82 条用户记忆里 41 条 importance 恰好 0.80, 全部落在
[0.80, 0.85) —— prompt 写「Level 1 … importance 0.8-1.0」, 代码判 L1 要
≥0.85. 模型以为自己标了核心记忆, 代码把它降成 L2.

大部分 0.80 是偏好/思维, 落 L2 本来就对, 所以修的是 prompt 的表述而不是代码
阈值 (降阈值会让 L1 被「用户喜欢看哈利波特」灌满). 但这两处数字必须一致,
否则同一份指令自相矛盾, 模型永远在缝里打转.
"""

from __future__ import annotations

import re

import pytest

from app.services.prompting.registry import PROMPT_DEFINITION_MAP

# pipeline.py 的分层阈值 —— 改那里就必须改这里, 反之亦然.
L1_THRESHOLD = 0.85
L2_THRESHOLD = 0.50
DROP_THRESHOLD = 0.10

EXTRACTION_KEYS = ("memory.extraction_user", "memory.extraction_ai")


@pytest.mark.parametrize("key", EXTRACTION_KEYS)
def test_prompt_l1_band_matches_code_threshold(key):
    text = PROMPT_DEFINITION_MAP[key].default_text
    assert "importance **≥ 0.85**" in text, "L1 区间必须写成代码实际用的 ≥0.85"
    assert "0.8-1.0" not in text, "旧表述会让模型把 0.80 当成 L1"


@pytest.mark.parametrize("key", EXTRACTION_KEYS)
def test_no_scoring_band_straddles_a_level_boundary(key):
    """任何一档都不能横跨分层边界 —— 跨了就意味着同一类事实有时 L1 有时 L2,
    模型给的分值落在哪半边纯看运气.

    第一版只查了 L1 边界, 于是漏掉「提醒 0.4-0.6」跨 L2 边界 (0.50) 这处 ——
    那条只是被 pipeline 的 clamp 兜住了, prompt 本身仍然自相矛盾.
    """
    text = PROMPT_DEFINITION_MAP[key].default_text
    section = text.split("importance 评分规则：", 1)[-1].split("额外要求", 1)[0]
    straddling = []
    for line in section.splitlines():
        m = re.search(r"(\d\.\d{1,2})-(\d\.\d{1,2})", line)
        if not m:
            continue
        lo, hi = float(m.group(1)), float(m.group(2))
        for boundary in (L1_THRESHOLD, L2_THRESHOLD, DROP_THRESHOLD):
            if lo < boundary < hi:
                straddling.append(f"{line.strip()} 跨过 {boundary}")
    assert not straddling, f"这些档横跨了分层边界: {straddling}"


@pytest.mark.parametrize("key", EXTRACTION_KEYS)
def test_reminder_band_matches_the_pipeline_clamp(key):
    """prompt 要 0.40-0.49, 代码 clamp 也是 0.40-0.49. 两处必须同步,
    否则 clamp 就是在替 prompt 擦屁股, 而 LLM 一直被要求给错的分。"""
    text = PROMPT_DEFINITION_MAP[key].default_text
    assert "强制 0.40-0.49" in text

    import inspect

    from app.services.memory.recording import pipeline

    src = inspect.getsource(pipeline.process_memory_pipeline)
    assert "max(0.4, min(0.49, float(importance)))" in src


@pytest.mark.parametrize("key", EXTRACTION_KEYS)
def test_level_boundaries_are_contiguous_and_ordered(key):
    text = PROMPT_DEFINITION_MAP[key].default_text
    assert "0.50-0.84" in text, "L2 区间要紧贴 L1 下界, 不能留缝"
    assert "0.10-0.49" in text, "L3 区间下界要跟丢弃阈值对齐"


def test_pipeline_thresholds_are_what_this_file_assumes():
    """守卫: 若有人改了 pipeline 的分层阈值, 这里立刻失败, 提醒同步 prompt."""
    import inspect

    from app.services.memory.recording import pipeline

    src = inspect.getsource(pipeline.process_memory_pipeline)
    assert f"importance >= {L1_THRESHOLD}" in src
    assert f"importance >= {L2_THRESHOLD}" in src
    assert f"importance < {DROP_THRESHOLD}" in src
