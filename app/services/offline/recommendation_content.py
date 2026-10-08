"""Readable detail fallbacks from canonical facts and explicitly optional advice."""

from __future__ import annotations

import re

from app.services.offline.content import plain_text
from app.services.offline.event_schedule import schedule_label


def useful_detail(text: str) -> bool:
    """An address or a repeated invitation is not a detail-page introduction."""
    body = plain_text(text)
    return len(body) >= 120 and len([p for p in body.split("\n\n") if p.strip()]) >= 3


def fallback_summary(card: dict) -> str:
    name = plain_text(card.get("location_name"))
    category = str(card.get("category") or "")
    if (card.get("discovery_metadata") or {}).get("kind") == "event":
        return f"可以留意{plain_text(card.get('title'))}，看看时间合不合适。"
    if "手作" in category:
        return f"想换个方式放松，可以去{name}了解一下手作体验。"
    if any(word in name for word in ("旧址", "遗址", "故居")):
        return f"喜欢看看老地方的话，可以把{name}放进散步路线。"
    if any(word in category for word in ("咖啡", "茶饮", "小吃", "小馆")):
        return f"想找个地方歇歇或吃点东西，可以考虑{name}。"
    if any(word in category for word in ("博物", "展览", "图书", "书店")):
        return f"想留一点时间慢慢看看，可以考虑{name}。"
    return f"想换条路线走走，可以把{name}作为这次出门的一站。"


def fallback_detail(card: dict) -> str:
    """Never infer facilities, admission, access or historical facts from a name."""
    name = plain_text(card.get("location_name"))
    address = plain_text(card.get("address"))
    category = str(card.get("category") or "")
    metadata = card.get("discovery_metadata") or {}
    event = metadata.get("event") or {}
    title = plain_text(card.get("title")) or name
    if metadata.get("kind") == "event":
        intro = plain_text(event.get("description") or event.get("schedule_evidence"))
        idea = (
            "可以先看看活动内容是不是自己感兴趣的，再按这一场的时间安排出门。"
            "如果需要门票或报名，出发前确认本场的参加方式，给路上留一点余量。"
        )
    elif "手作" in category:
        intro = f"{name}是这次可以考虑的一处手作小店。"
        idea = (
            "如果想把注意力从屏幕上挪开，可以先选一项自己感兴趣的手作方向。"
            "到店后问问当天能体验哪些项目、需要多久，再决定做什么；不用一次安排太满。"
        )
    elif any(word in name for word in ("旧址", "遗址", "故居")):
        intro = f"{name}可以作为这次认识城市老地方的一站。"
        idea = (
            "如果对老地方感兴趣，可以把这里放进散步路线，留一点时间看看地点本身。"
            "想拍照的话，可以从现场允许停留的位置找找角度；是否能进入，以当天开放安排为准。"
        )
    elif any(word in category for word in ("咖啡", "茶饮")):
        intro = f"{name}是这次可以考虑的一处咖啡或茶饮去处。"
        idea = (
            "想给自己留一段歇歇的时间，可以先看看菜单，挑一杯合口味的饮品。"
            "也可以带本自己的书；到店后按座位情况安排，慢慢喝一会儿，不用把休息变成任务。"
        )
    elif any(word in category for word in ("图书", "书店")):
        intro = f"{name}可以作为这次找书、看看书的一站。"
        idea = (
            "可以先选一个最近感兴趣的主题，慢慢找找相关的书。"
            "碰到想读的内容就多停一会儿；是否能坐下阅读、借阅或购买，按现场规则安排。"
        )
    elif any(word in category for word in ("博物", "展览")):
        intro = f"{name}可以作为这次了解展览内容的一站。"
        idea = (
            "出发前先看看当天开放和展览安排，选自己感兴趣的部分慢慢看。"
            "不用追求一次逛完；如果想拍照或记点东西，按现场参观规则来就好。"
        )
    elif any(word in category for word in ("小吃", "小馆", "餐")):
        intro = f"{name}是这次可以考虑的一处吃东西的去处。"
        idea = (
            "可以先看看菜单和价格，挑一两样自己想尝的，再按饭量加点。"
            "如果路过时人比较多，可以根据等候情况调整安排，让这次出门留一点随意的空间。"
        )
    else:
        intro = f"{name}可以作为这次换条路线、走走看看的一站。"
        idea = (
            "可以把这里安排成一次轻松的停留，到现场再按自己的兴趣和体力决定怎么走。"
            "想多看看就多留一会儿，也可以和原本的出门计划顺路安排，不用赶着完成固定路线。"
        )
    sections = [f"📍 这次去哪里\n{title}\n{intro}", f"💡 可以怎么安排\n{idea}"]
    practical = [f"地点：{name}", f"地址：{address}"]
    label = schedule_label(event)
    if label:
        practical.append("本场时间：" + label)
    elif metadata.get("opening_hours"):
        practical.append("营业信息：" + plain_text(metadata["opening_hours"]))
    price = str(metadata.get("price_info") or "").strip()
    if metadata.get("kind") != "event" and re.fullmatch(r"\d+(?:\.\d+)?", price):
        practical.append(f"人均消费参考：约{price}元，具体项目价格以店内为准")
    practical.append(
        "出发前确认当天开放安排；涉及体验或参观时，也可以先问问是否需要预约。"
    )
    sections.append("🧭 出门小提示\n" + "\n".join(practical))
    return "\n\n".join(sections)
