"""预筛必须拿到上一句 AI 的话.

记忆管线里判"记/不记"这一级, 曾经只看单条用户消息, 提示词也只有一句"判断这句话
是否值得进入记忆", 没有任何判据. 拿 40 条双评审都认为该记的真实消息实测, **召回
只有 5%**:

    AI: 那你吃完饭准备歇会吗        AI: 你今天还好吗
    用户: 嗯准备睡个午觉            用户: 不好

单看用户那句确实像语气词 —— 含义整个在上一句里. 而这一级的假阴性是终局: 判"不记"
的消息不进抽取, 那条信息永久丢失.

补上下文 + 写清判据后召回 72%, 误收 10%. 中间试过只讲"该记什么"不讲"不该记什么",
召回 82% 但误收 45% —— 明确列出不记的形态才是把误收压下来的关键, 所以那几条
反例在提示词里是承重的, 不能当啰嗦删掉.
"""

from __future__ import annotations

import pytest

from app.services.memory.recording.pipeline import _last_ai_line
from app.services.prompting.registry import PROMPT_DEFINITION_MAP

KEY = "memory.judgement_user"


def _prompt() -> str:
    return PROMPT_DEFINITION_MAP[KEY].default_text


def test_prompt_takes_the_previous_ai_line():
    assert "{prev_ai}" in _prompt(), (
        "预筛提示词丢了上下文占位符 —— 只看单条消息时召回实测只有 5%"
    )
    assert "{message}" in _prompt()


def test_prompt_tells_the_model_to_read_context():
    assert "上一句" in _prompt()


def test_prompt_names_what_not_to_record():
    """只说该记什么会让误收从 10% 涨到 45%; 这几条反例是承重的."""
    text = _prompt()
    assert "不记" in text
    for shape in ("提问", "机械确认", "招呼"):
        assert shape in text, f"少了「{shape}」这类反例, 误收率会显著上升"


def test_prompt_covers_the_three_things_worth_keeping():
    text = _prompt()
    for kind in ("事实", "偏好", "情绪"):
        assert kind in text


@pytest.mark.parametrize("context,expected", [
    ("user: 在吗\nassistant: 你今天还好吗\nuser: 嗯", "你今天还好吗"),
    ("assistant: 第一句\nassistant: 第二句", "第二句"),
    ("", ""),
    ("user: 只有用户说话", ""),
    ("assistant: ", ""),
])
def test_last_ai_line_extraction(context, expected):
    assert _last_ai_line(context) == expected


def test_last_ai_line_never_returns_a_user_turn():
    """把用户的话当成 AI 的上文喂进去, 会让预筛围着错误的对象判断."""
    assert _last_ai_line("assistant: AI说的\nuser: 用户说的") == "AI说的"


def test_missing_context_degrades_instead_of_failing():
    """取不到上文时返回空串 —— 预筛退回只看单条消息, 召回会掉但不会报错."""
    assert _last_ai_line(None) == ""  # type: ignore[arg-type]
