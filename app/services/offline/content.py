"""Canonical presentation text and conservative place/source identity checks."""
from __future__ import annotations
import re
from difflib import SequenceMatcher
from urllib.parse import urlsplit, urlunsplit


def plain_text(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", text)
    text = re.sub(r"(?:https?://|www\.)[^\s<>，。；！？）)]+", "", text, flags=re.I)
    return re.sub(r"[ \t]+", " ", text).strip()


def canonical_url(value: object) -> str:
    try:
        p = urlsplit(str(value or ""))
        if p.scheme not in {"http", "https"} or not p.hostname or p.username:
            return ""
        return urlunsplit((p.scheme, p.netloc.lower(), p.path.rstrip("/"), p.query, ""))
    except ValueError:
        return ""


def normalized(value: object) -> str:
    return re.sub(r"[\W_]+", "", str(value or "")).casefold()


# Discovery articles are useful search input, but never an individual destination.
_COLLECTION_TITLE = re.compile(
    r"攻略|合集|排行榜|状元榜|扫街榜|十大|周末去哪|这些|那些|这几|那几|盘点|汇总|大全|"
    r"[0-9一二三四五六七八九十]+[个大处家]|(?:咖啡店|咖啡馆|景点|公园|小店).{0,6}(?:推荐|整理|指南)|"
    r"\b(?:the best|things to do|travel guide)\b", re.I
)


def is_collection_title(value: object) -> bool:
    return bool(_COLLECTION_TITLE.search(str(value or "")))


def concrete_place_name(value: object, city: object = "") -> bool:
    text = str(value or "").strip()
    if not 2 <= len(text) <= 60 or is_collection_title(text):
        return False
    if re.search(r"[？！?!]|门户网站|人民政府|高德地图|怎么玩|去哪玩|玩得开心|藏着|收藏这|打卡指南", text):
        return False
    if re.search(r"^(?:从.{1,30}(?:聊聊|看看)|带你|一起看|走进|探访|揭秘|探秘|寻访)", text):
        return False
    name = normalized(text)
    city_name = normalized(city).removesuffix("市")
    local_name = name.removeprefix(city_name).removeprefix("市") if city_name else name
    if name != local_name and local_name in {"博物馆", "图书馆", "公园"}:
        return True
    return local_name not in {
        "", "附近", "周边", "当前位置", "当前位置附近", "咖啡", "咖啡店", "咖啡馆",
        "茶馆", "公园", "博物馆", "图书馆", "景区", "景点", "小店", "美食", "餐厅",
    }


def place_source_matches(card: dict, title: str, content: str) -> bool:
    name = normalized(card.get("location_name"))
    city = normalized(card.get("city")).removesuffix("市")
    heading, body = normalized(title), normalized(content)
    # A listicle is a discovery source, never a place or a place's photo album.
    if is_collection_title(title) or not concrete_place_name(card.get("location_name"), card.get("city")):
        return False
    return bool(len(name) >= 2 and name in heading and city and city in heading + body)


def novel_fragment(text: str, previous: list[str]) -> bool:
    value = normalized(text)
    return bool(value) and all(SequenceMatcher(None, value, normalized(old)).ratio() < .82 for old in previous)
