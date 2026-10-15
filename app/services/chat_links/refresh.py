"""Refresh only unversioned, unusable previews; historical versions stay pinned."""

from app.services.chat_links.extraction import extract_urls


def needs_refresh(link) -> bool:
    if link.platform != "微博":
        return False
    fields = (link.title, link.description, link.summary, link.content_text)
    if "Sina Visitor System" in " ".join(str(value or "") for value in fields):
        return True
    body = " ".join(str(value or "") for value in fields[1:]).strip()
    count = len(extract_urls(body))
    return count > 0 and len(body.replace(" ", "")) <= 120 + count * 80
