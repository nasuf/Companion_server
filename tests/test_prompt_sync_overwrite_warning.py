"""代码默认值覆盖后台编辑时必须告警.

`ensure_prompt_templates` 在代码默认值变化时会无条件用它盖掉线上 content —— 不管
那份 content 是不是有人在后台精心改过的。项目历史上 `chat.response_instruction`
就这样被抹过一次, 事后只能从版本表把内容捞回来重新提交。

覆盖行为本身是合理的 (代码默认值是终极真理), 但它此前只记一行 info, 跟"例行同步"
长得一模一样。真正要区分的是: 被盖掉的那份内容是不是人工改的。
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from app.services.prompting import store


def _sync_source() -> str:
    return textwrap.dedent(inspect.getsource(store.ensure_prompt_templates))


def test_overwriting_a_hand_edited_prompt_logs_at_error_level():
    """被盖掉的是人工编辑时, info 级别不够 —— 它会淹没在例行同步日志里。"""
    tree = ast.parse(_sync_source())
    error_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "error"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "logger"
    ]
    assert error_calls, "覆盖后台编辑时没有 error 级告警"


def test_the_warning_is_conditional_on_an_actual_hand_edit():
    """每次代码默认值变化都告警的话, 告警就没人看了。只有 content 与
    defaultContent 不同 (即有人在后台改过) 才值得吼。"""
    source = _sync_source()
    assert "existing.content != existing.defaultContent" in source


def test_hand_edit_detection_uses_content_vs_default_not_version_source():
    """版本表的 source 字段判断不出人工编辑 —— bootstrap 播种时它写的是 'db',
    一眼看去像人工操作。判据必须是 content 与 defaultContent 的比较。"""
    source = _sync_source()
    assert 'source == "db"' not in source
    assert "existing.defaultContent" in source


def test_operator_is_told_how_to_recover():
    """告警要说清楚怎么捞回来, 否则看到也不知道下一步做什么。"""
    source = _sync_source()
    assert "prompt_template_versions" in source
    assert "update_prompt_text" in source
