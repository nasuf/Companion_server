from app.services.memory.taxonomy import (
    get_compression_rule,
    get_promotion_rule,
    resolve_taxonomy,
    summarize_batch_taxonomy,
)


class _MemoryStub:
    def __init__(self, main_category: str | None, sub_category: str | None):
        self.mainCategory = main_category
        self.subCategory = sub_category


def test_resolve_taxonomy_defaults_unknown_values():
    result = resolve_taxonomy(main_category="不存在", sub_category="乱填", legacy_type=None)
    assert result.main_category == "生活"
    assert result.sub_category == "其他"


def test_get_promotion_rule_for_identity_name():
    rule = get_promotion_rule("身份", "姓名")
    assert rule["allow_l1"] is True
    assert rule["min_mentions"] == 1


def test_get_promotion_rule_for_unsupported_category():
    rule = get_promotion_rule("情绪", "高兴")
    assert rule["allow_l1"] is False


def test_get_compression_rule_for_preference():
    rule = get_compression_rule("偏好")
    assert rule["allow_cross_subcategory"] is False
    assert rule["batch_size"] == 8


def test_summarize_batch_taxonomy_picks_majority():
    items = [
        _MemoryStub("身份", "姓名"),
        _MemoryStub("身份", "姓名"),
        _MemoryStub("生活", "工作"),
    ]
    assert summarize_batch_taxonomy(items) == ("身份", "姓名")
