from app.services.chat.reply_post_process import strip_duplicate_delay_preface


def test_strip_duplicate_delay_preface_removes_only_leading_delay_text():
    assert (
        strip_duplicate_delay_preface("不好意思，刚才睡着了现在才回。你刚说那个书店，我有印象")
        == "你刚说那个书店，我有印象"
    )
    assert strip_duplicate_delay_preface("刚醒") == ""
    assert strip_duplicate_delay_preface("你刚说那个书店，我有印象") == "你刚说那个书店，我有印象"
