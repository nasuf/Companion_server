"""真人打字聊天不写动作/神态旁白: 「（清了清嗓子）」「（突然有点慌）」「*小声*」。

回复要求里早有禁令, 角色模型在唱歌这类表演场景里仍会写 (2026-10-31~11-01 生产 5 条,
全部在开头或独占一条)。所有出口统一经 strip_system_markers 剥掉。
"""
from __future__ import annotations

import pytest

from app.services.chat.reply_formatting import split_and_validate_replies, strip_system_markers


@pytest.mark.parametrize("text, expected", [
    # 生产原样
    ("（清了清嗓子又唱了一段） 风雪是你 平淡是你 清贫也是你", "风雪是你 平淡是你 清贫也是你"),
    ("（突然有点慌）你不会觉得我太啰嗦吧？", "你不会觉得我太啰嗦吧？"),
    ("（清了下嗓子，声音放得很轻） 我听见雨滴落在青青草地", "我听见雨滴落在青青草地"),
    ("（清了清嗓子）", ""),
    ("（清唱了一段温柔的小情歌）", ""),
    ("*小声* 其实我也想你", "其实我也想你"),
    ("(笑)(挠头) 被你发现了", "被你发现了"),
])
def test_leading_stage_directions_are_removed(text, expected):
    assert strip_system_markers(text) == expected


@pytest.mark.parametrize("text", [
    "活动卡片：红小姐咖啡馆(西津渡店)",                 # 店名分店
    "你们已退出共听《No Words (Director's Cut)》",     # 歌名
    "(・ω・) 好呀",                                      # 颜文字
    "我那会儿在北京（出差）",                            # 句中补充说明
    "(1) 先吃饭 (2) 再睡觉",                             # 编号
])
def test_other_parentheses_are_kept(text):
    assert strip_system_markers(text) == text


def test_a_bubble_that_is_only_a_stage_direction_is_dropped():
    replies = split_and_validate_replies("那我唱啦 别嫌难听哦✨||（清唱了一段温柔的小情歌）")
    assert replies == ["那我唱啦 别嫌难听哦✨"]


def test_empty_text_is_returned_as_is():
    from app.services.chat.reply_formatting import strip_stage_direction

    assert strip_stage_direction("") == ""
