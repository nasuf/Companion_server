"""Guards for the persona.grounding_check wording calibrated on real data.

Calibration evidence: evals/persona_grounding/schedule_replay.py (schedules)
and reply_memory_replay.py (chat replies, AI memories).
"""
import json
from string import Formatter

from app.services.prompting.defaults import PERSONA_GROUNDING_CHECK_PROMPT as POLICY


def _rule(n: int) -> str:
    return next(line for line in POLICY.splitlines() if line.startswith(f"{n}. "))


def _sentence(text: str, start: str) -> str:
    return start + text.split(start, 1)[1].split("。", 1)[0]


def test_placeholders_and_render():
    fields = {name for _, name, _, _ in Formatter().parse(POLICY) if name is not None}
    assert fields == {"facts", "user_location", "kind", "question", "items"}
    rendered = POLICY.format(
        facts=json.dumps({"name": "小伴", "city": "云南省普洱市思茅区"}, ensure_ascii=False),
        user_location="（本次不校验对方位置）",
        kind="schedule", question=json.dumps("", ensure_ascii=False),
        items=json.dumps([{"index": 0, "text": "傍晚在附近公园散步"}], ensure_ascii=False),
    )
    assert "{facts}" not in rendered and "附近公园散步" in rendered
    example = json.loads(rendered.splitlines()[-1])["verdicts"][0]
    assert example["allowed"] is True and example["about"] in {"self", "user"}


def test_unnamed_place_categories_are_generic():
    # 原文把「公园」和街道、小区并列成专名的例子, 核验模型于是把不带名字的「附近
    # 公园」「老城区市集」也判成编造地点: 作息被抹成中性说法, 记忆不入库。
    generic = _rule(3).split("专名是", 1)[0]
    for word in ("公园", "市集", "老城区", "植物园", "街角", "江边"):
        assert word in generic
    for example in ("附近公园散步", "城市公园长椅", "老城区市集", "城郊植物园", "街角咖啡馆"):
        assert f"“{example}”" in generic
    # 被问「你在外地做什么」时答「去老街走走」: 只强调泛称时模型放过了这种顺着外地接话
    presupposed = _sentence(generic, "但提问在问角色人在外地做什么时")
    assert "泛称也等于承认人在那里" in presupposed and presupposed.endswith("拒绝")
    # 2026-10-23: 不限定"问的是角色"时, 用户分享镇江位置后回"你在润州呀 我刚搜了附近
    # 有家手作店", 模型照这条把"附近"算成角色在镇江, 整条回复被换掉。
    shared = _sentence(generic, "对方说或分享自己在哪时")
    assert "“附近”是对方附近，不是角色" in shared


def test_named_places_still_need_the_profile():
    named = _rule(3).split("专名是", 1)[1]
    for example in ("天桥支路", "润州区", "顺峰山公园"):
        assert example in named
    assert "必须字面出现在档案中，否则拒绝" in named
    assert "不能因为“可能在同城”放行" in named
    assert "用户的位置和经历不受这项限制" in named


def test_past_trips_dishes_and_leaving_town():
    named = _rule(3).split("专名是", 1)[1]
    # 人在家里只顺带说「之前在某外地街区捡的瓷片」, 模型曾当成回忆放行; 主语限定在
    # 角色, 并配用户经历的放行对照 —— 只写拒绝例子时「整理用户从…带回的」也被拒;
    # 把放行对照并进这一句, 回顾外地行程的那类又开始漏, 所以对照单独放在句末。
    trips = _sentence(named, "角色说自己去过那里")
    assert "亲自从那里带回东西、回顾自己的那次行程，同样算亲历" in trips
    assert "“整理用户从厦门带回的特产照片”允许" in named
    # 这条核验也守聊天回复, 误拒会把整条回复换成「地点说错了」; 只说「菜名不是地点」
    # 时「在镇江吃了锅盖面」也被放行。
    dishes = _sentence(named, "菜名、特产里的地名不是地点")
    assert "角色仍在档案城市，允许" in dishes and "写明“在镇江吃”就是人在镇江" in dishes
    leaving = _sentence(named, "没写地名但等于离开档案城市的说法也拒绝")
    for example in ("回老家", "去外地", "海边散步", "只是想起、聊起老家不算"):
        assert example in leaving
