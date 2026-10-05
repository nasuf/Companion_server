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


def place_source_matches(card: dict, title: str, content: str) -> bool:
    name = normalized(card.get("location_name"))
    city = normalized(card.get("city")).removesuffix("市")
    heading, body = normalized(title), normalized(content)
    # A listicle is a discovery source, never a place or a place's photo album.
    if re.search(r"攻略|合集|排行榜|十大|周末去哪|[0-9一二三四五六七八九十]+[个大处家]", title):
        return False
    return bool(len(name) >= 2 and name in heading and city and city in heading + body)


def novel_fragment(text: str, previous: list[str]) -> bool:
    value = normalized(text)
    return bool(value) and all(SequenceMatcher(None, value, normalized(old)).ratio() < .82 for old in previous)
