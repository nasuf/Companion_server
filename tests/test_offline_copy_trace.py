"""Activity copy decisions reach the Trace as readable steps."""
from app.services.offline import copy_trace


def test_guard_block_names_the_field_and_the_draft(monkeypatch):
    calls = []
    monkeypatch.setattr(copy_trace, "record_manual_step", lambda **kwargs: calls.append(kwargs))
    copy_trace.record_guard_block("note", 1, ["没有用户原文可依据，删除“你之前”开头这类说用户之前提过的话"])
    (call,) = calls
    assert call["display_name"] == "代码核验 · 推荐寄语第 2 稿"
    assert call["decision_label"].startswith("拦下：没有用户原文可依据") and len(call["decision_label"]) <= 39
    assert call["outputs"]["issues"] and call["inputs"] == {"stage": "note", "draft": 2, "activity": None}


def test_outcome_says_what_the_user_sees_per_field(monkeypatch):
    calls = []
    monkeypatch.setattr(copy_trace, "record_manual_step", lambda **kwargs: calls.append(kwargs))
    copy_trace.record_copy_outcome(
        {"card_copy": "fallback_no_proposal", "detail": "fallback_rejected", "note": "verified_no_evidence"},
        {"detail": ["“静谧留白”：没有环境描述"]},
    )
    (call,) = calls
    assert call["display_name"] == "活动文案结果"
    assert call["decision_label"] == (
        "卡片短文案：模板（选卡没拿到模型结果） · 详情文案：模板（被拒） · 推荐寄语：模型文案")
    assert call["outputs"]["issues"]["detail"] == ["“静谧留白”：没有环境描述"]
    copy_trace.record_copy_outcome({"invite": None})
    assert calls[-1]["decision_label"] == "聊天邀请语：not_recorded"


def test_steps_name_the_activity_when_several_cards_share_a_trace(monkeypatch):
    calls = []
    monkeypatch.setattr(copy_trace, "record_manual_step", lambda **kwargs: calls.append(kwargs))
    copy_trace.record_guard_block("detail", 0, ["x"], subject="镇江新华书店京畿路店 京畿书院(阅畿咖啡)")
    copy_trace.record_selection_fallback("25 秒内没有完成", subject="西津渡小剧场")
    assert calls[0]["display_name"] == "代码核验 · 详情文案第 1 稿 · 镇江新华书店京畿路店 京畿书"
    assert calls[1]["display_name"] == "推荐卡生成 · 未拿到模型结果 · 西津渡小剧场"
    assert calls[1]["decision_label"] == "25 秒内没有完成，改用默认候选和模板摘要"



def test_withheld_evidence_and_partial_card_copy_read_plainly(monkeypatch):
    calls = []
    monkeypatch.setattr(copy_trace, "record_manual_step", lambda **kwargs: calls.append(kwargs))
    copy_trace.record_evidence_withheld("crisis_care")
    copy_trace.record_evidence_withheld("crisis_check_unavailable")
    copy_trace.record_copy_outcome({"card_copy": "verified_partial"})
    assert calls[0]["display_name"] == "用户依据 · 近期有危机照护"
    assert calls[1]["display_name"] == "用户依据 · 危机照护检查不可用"
    assert calls[0]["decision_label"] == "不用最近对话和情绪类记忆，只用长期兴趣"
    assert calls[2]["decision_label"] == "卡片短文案：模型文案（个别项改用默认）"
