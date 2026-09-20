"""思绪碎片引擎纯逻辑单测（spec §4.9 概率阶梯 / §4.10 等级分布 / 去重指纹）。"""

import random
from collections import Counter

from app.services.offline import recognition as rec


def test_probability_ladder():
    assert rec.produce_probability(0) == 1.0
    assert rec.produce_probability(1) == 0.4
    assert rec.produce_probability(2) == 0.25
    assert rec.produce_probability(3) == 0.0
    assert rec.produce_probability(9) == 0.0


def test_first_hit_always_produces():
    rng = random.Random(0)
    assert all(rec.should_produce(0, rng) for _ in range(100))


def test_cap_closes_at_three():
    rng = random.Random(0)
    assert not any(rec.should_produce(3, rng) for _ in range(100))


def test_second_hit_probability_near_040():
    rng = random.Random(1234)
    n = 20000
    hits = sum(1 for _ in range(n) if rec.should_produce(1, rng))
    assert 0.37 <= hits / n <= 0.43


def test_third_hit_probability_near_025():
    rng = random.Random(99)
    n = 20000
    hits = sum(1 for _ in range(n) if rec.should_produce(2, rng))
    assert 0.22 <= hits / n <= 0.28


def test_tier_distribution_near_68_25_7():
    rng = random.Random(42)
    n = 30000
    c = Counter(rec.roll_tier(rng) for _ in range(n))
    assert 0.64 <= c["rare"] / n <= 0.72
    assert 0.22 <= c["epic"] / n <= 0.28
    assert 0.05 <= c["legendary"] / n <= 0.09


def test_fingerprint_orderless_and_case_insensitive():
    a = rec.content_fingerprint(["Tree", "sky ", "cup"])
    b = rec.content_fingerprint(["cup", "SKY", "tree"])
    assert a == b and a
    assert rec.content_fingerprint([]) == ""


def test_lead_in_from_pool():
    for tier in ("rare", "epic", "legendary"):
        assert rec.pick_lead_in(tier) in rec._LEAD_INS[tier]


def test_parse_match_plain_json():
    raw = (
        '{"hit": true, "condition_short_name": "杯子", '
        '"confidence": 0.9, "keywords": ["咖啡", "杯子"]}'
    )
    m = rec._parse_match(raw)
    assert m is not None
    assert m["hit"] is True
    assert m["condition_short_name"] == "杯子"
    assert "咖啡" in m["keywords"]


def test_parse_match_with_code_fence_and_prose():
    raw = "好的，判断如下：\n```json\n{\"hit\": false, \"keywords\": []}\n```"
    m = rec._parse_match(raw)
    assert m is not None and m["hit"] is False
    assert m["condition_short_name"] is None


def test_parse_match_garbage_returns_none():
    assert rec._parse_match("这里没有 JSON") is None
    assert rec._parse_match("") is None


def test_parse_conditions_json_array():
    from app.services.offline import shooting_conditions as sc

    raw = (
        '[{"short_name": "杯子", "criteria": "拍到杯子"}, '
        '{"short_name": "天空", "criteria": "拍到天空"}]'
    )
    conds = sc._parse_conditions(raw)
    assert [c["short_name"] for c in conds] == ["杯子", "天空"]


def test_parse_conditions_code_fence_and_cap_at_5():
    from app.services.offline import shooting_conditions as sc

    items = ", ".join(
        f'{{"short_name": "c{i}", "criteria": "k{i}"}}' for i in range(8)
    )
    conds = sc._parse_conditions(f"```json\n[{items}]\n```")
    assert len(conds) == 5  # capped
    assert conds[0]["short_name"] == "c0"


def test_parse_conditions_drops_incomplete_and_handles_garbage():
    from app.services.offline import shooting_conditions as sc

    raw = '[{"short_name": "只有名字"}, {"criteria": "只有要点"}, {"short_name":"好","criteria":"齐"}]'
    conds = sc._parse_conditions(raw)
    assert conds == [{"short_name": "好", "criteria": "齐"}]
    assert sc._parse_conditions("no json") == []
