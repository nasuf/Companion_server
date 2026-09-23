"""思绪碎片引擎纯逻辑单测（spec §4.9 概率阶梯 / §4.10 等级分布 / 去重指纹）。"""

import random
from collections import Counter
from unittest.mock import AsyncMock

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


def test_parse_subjects_sorted_by_confidence():
    raw = (
        '{"subjects": [{"type": "杯子", "confidence": 0.6}, '
        '{"type": "天空", "confidence": 0.9}]}'
    )
    subs = rec._parse_subjects(raw)
    assert [s["type"] for s in subs] == ["天空", "杯子"]  # 高置信度在前
    assert subs[0]["confidence"] == 0.9


def test_parse_subjects_code_fence_and_prose():
    raw = "识别如下：\n```json\n{\"subjects\": [{\"type\": \"花\", \"confidence\": 0.8}]}\n```"
    subs = rec._parse_subjects(raw)
    assert subs == [{"type": "花", "confidence": 0.8}]


def test_parse_subjects_garbage_returns_empty():
    assert rec._parse_subjects("这里没有 JSON") == []
    assert rec._parse_subjects("") == []
    assert rec._parse_subjects('{"subjects": []}') == []


def test_match_subject_to_item_by_short_name_and_category():
    items = [
        {"id": "i1", "short_name": "咖啡杯", "category": "餐具"},
        {"id": "i2", "short_name": "天空", "category": "天空"},
    ]
    subjects = [{"type": "杯", "confidence": 0.9}]  # 子串命中 short_name
    item, subject = rec._match_subject_to_item(subjects, items)
    assert item is not None and item["id"] == "i1"
    assert subject is not None and subject["type"] == "杯"


def test_match_subject_to_item_no_match_returns_none():
    items = [{"id": "i1", "short_name": "咖啡杯", "category": "餐具"}]
    item, subject = rec._match_subject_to_item(
        [{"type": "汽车", "confidence": 0.9}], items
    )
    assert item is None and subject is None


def test_parse_items_json_object():
    from app.services.offline import shooting_conditions as sc

    raw = (
        '{"items": [{"short_name": "杯子", "category": "餐具"}, '
        '{"short_name": "天空", "category": "天空"}]}'
    )
    items = sc._parse_items(raw)
    assert [i["short_name"] for i in items] == ["杯子", "天空"]
    assert items[0]["category"] == "餐具"


def test_parse_items_code_fence_and_cap_at_5():
    from app.services.offline import shooting_conditions as sc

    entries = ", ".join(
        f'{{"short_name": "c{i}", "category": "cat"}}' for i in range(8)
    )
    items = sc._parse_items(f'```json\n{{"items": [{entries}]}}\n```')
    assert len(items) == 5  # capped
    assert items[0]["short_name"] == "c0"


def test_parse_items_drops_nameless_and_handles_garbage():
    from app.services.offline import shooting_conditions as sc

    raw = '{"items": [{"category": "只有类目"}, {"short_name": "好", "category": "齐"}]}'
    items = sc._parse_items(raw)
    assert len(items) == 1
    assert items[0]["short_name"] == "好"
    assert items[0]["category"] == "齐"
    assert items[0]["criteria"] == ""
    assert items[0]["guidance_profile"] == {"aliases": [], "guidance": {}}
    assert sc._parse_items("no json") == []


def test_ensure_min_items_tops_up_to_three():
    from app.services.offline import shooting_conditions as sc

    # 空 → 补到至少 3（spec §3.4 拍摄物品 3–5）。
    assert len(sc._ensure_min_items([])) >= 3
    # 1 个 → 补到 3，且保留原有、不与兜底 short_name 重复。
    out = sc._ensure_min_items([{"short_name": "喷泉", "category": "水"}])
    assert len(out) >= 3
    assert out[0] == {"short_name": "喷泉", "category": "水"}
    assert len({i["short_name"] for i in out}) == len(out)


def test_ensure_min_items_keeps_when_already_enough():
    from app.services.offline import shooting_conditions as sc

    src = [
        {"short_name": "花", "category": "植物"},
        {"short_name": "湖", "category": "水"},
        {"short_name": "长椅", "category": "设施"},
    ]
    assert sc._ensure_min_items(src) == src  # ≥3 不追加


async def test_photo_match_threshold_downgrades_weak_exact_to_near(monkeypatch):
    monkeypatch.setattr(
        rec,
        "get_prompt_text",
        AsyncMock(
            return_value=(
                "{photo_description}|{focus_condition_id}|{conditions_json}"
            )
        ),
    )
    monkeypatch.setattr(
        rec,
        "invoke_text",
        AsyncMock(
            return_value=(
                '{"relation":"exact","condition_id":"c1","confidence":0.65,'
                '"observed_subject":"彩色细节"}'
            )
        ),
    )
    result = await rec._classify_photo_match(
        "一处彩色细节",
        [
            {
                "id": "c1",
                "short_name": "花",
                "category": "植物",
                "criteria": "主体清楚",
                "guidance_profile": {"aliases": ["花朵"]},
            }
        ],
        focus_condition_id="c1",
    )

    assert result["relation"] == "near"
    assert result["condition_id"] == "c1"


async def test_visible_message_leak_is_rewritten(monkeypatch):
    conditions = [
        {
            "id": "c1",
            "short_name": "花",
            "category": "植物",
            "guidance_profile": {"aliases": ["花朵"]},
        }
    ]
    monkeypatch.setattr(
        rec,
        "get_prompt_text",
        AsyncMock(return_value="{draft}|{forbidden_terms}|{safe_hint}"),
    )
    monkeypatch.setattr(
        rec,
        "invoke_text",
        AsyncMock(return_value='{"text":"换个角度看看附近的颜色和明暗。"}'),
    )

    result = await rec._guard_visible_message(
        "去拍一朵花吧",
        conditions,
        safe_hint="留意颜色和细小纹理",
        fallback="慢慢逛就好。",
    )

    assert result == "换个角度看看附近的颜色和明暗。"
    assert "花" not in result


async def test_legacy_fallback_rejects_low_confidence_match(monkeypatch):
    monkeypatch.setattr(
        rec,
        "get_prompt_text",
        AsyncMock(side_effect=RuntimeError("registry unavailable")),
    )
    monkeypatch.setattr(
        rec,
        "_detect_subjects",
        AsyncMock(return_value=[{"type": "花", "confidence": 0.1}]),
    )

    result = await rec._classify_photo_match(
        "远处似乎有一点颜色",
        [{"id": "c1", "short_name": "花", "category": "植物"}],
        focus_condition_id="c1",
    )

    assert result["relation"] == "none"
    assert result["condition_id"] == ""


def test_directional_category_leak_is_blocked_but_place_name_is_allowed():
    condition = {
        "short_name": "花",
        "category": "植物",
        "guidance_profile": {"aliases": ["花朵"]},
    }

    assert rec.contains_hidden_target("留意一下附近的植物", [condition])
    assert not rec.contains_hidden_target("植物园里逛着舒服吗", [condition])


def test_hidden_progress_language_is_treated_as_mechanical():
    for text in ("有点进展了", "还剩一个", "切换到下一个", "解锁新的阶段"):
        assert rec._MECHANICAL_TERMS_RE.search(text)
