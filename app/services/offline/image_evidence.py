"""Page-level image evidence. Related-page advertisements are not place photos."""
from html.parser import HTMLParser
import re
from urllib.parse import urljoin
from app.services.offline.content import canonical_url, normalized

_IMAGE = re.compile(r'!\[([^\]]*)\]\(([^\s)]+)(?:\s+"[^"]*")?\)')
_NON_PHOTO = re.compile(r'logo|icon|二维码|广告|标识|标志|示意图|效果图|设计方案|规划图|头像|海报|订阅|吉祥物', re.I)
_SCENE = re.compile(r'外观|外景|实景|全景|内景|馆舍|建筑|大门|正门|入口|馆内|阅览|风景|湖面|绿道|步道|远景|近景|侧视|匾额|展厅|展陈|陈列|器展|精品展|庭院|花园|门头|店内|吧台|座位区|用餐区|书架|摊位|摊档')
_NAMED_VENUE = re.compile(r'[\u4e00-\u9fffA-Za-z]{2,24}(?:图书馆|博物馆|展览馆|公园|咖啡店|咖啡馆|书店)')


def _other_place_caption(caption: str, name: str) -> bool:
    # Generic scene captions are useful on a bound POI page, but an explicitly
    # named different venue (e.g. a nearby attraction) must not borrow its identity.
    own = normalized(name).replace('市', '')
    for match in _NAMED_VENUE.finditer(caption):
        other = normalized(match.group()).replace('市', '')
        if own not in other and other not in own:
            return True
    return False


def _visible_text(text: str) -> str:
    text = _IMAGE.sub('', text)
    return re.sub(r'\[([^\]]*)\]\([^)]*\)|\]\([^)]*\)', r'\1', text).strip()


def indexed_image_evidence(raw: str, name: str, images: list[dict], source_url: str = '') -> dict[str, str]:
    """Extract per-photo evidence without leaking captions between neighbouring images.

    A caller first verifies the page's place and city. Native image captions and
    the enclosing Markdown link title are evidence; a generic page heading isn't.
    Relative image links are resolved against that verified page, not a search URL.
    """
    result = {}
    descriptions = {canonical_url(urljoin(source_url, i.get('url') or '')): str(i.get('description') or '')
                    for i in images if i.get('description_source') in {'alt', 'caption'}}
    seen_markdown = set()
    previous = ''
    heading = ''
    for block in raw.split('\n\n'):
        matches = list(_IMAGE.finditer(block))
        if not matches:
            text = _visible_text(block)
            if text.startswith('#'):
                heading = text.lstrip('# ').strip()
            previous = text if len(text) <= 80 and not text.startswith('#') else ''
            continue
        for index, match in enumerate(matches):
            url = canonical_url(urljoin(source_url, match.group(2)))
            if not url:
                continue
            seen_markdown.add(url)
            alt = re.sub(r'^Image\s*\d*\s*:?\s*', '', match.group(1), flags=re.I)
            # [![Image N](photo)](album "展厅说明") is common in indexed museum pages.
            tail = block[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(block)]
            linked_caption = re.match(r'^\]\([^\s)]+(?:\s+"([^"]*)")?\)', tail)
            caption = (alt or descriptions.get(url, '') or
                       (linked_caption.group(1) if linked_caption else '') or '').strip()
            start = matches[index - 1].end() if index else 0
            before = _visible_text(block[start:match.start()])[-350:]
            nearby = previous + ' ' + before
            # The previous photo's caption must not reject the current photograph.
            if (_NON_PHOTO.search(caption) or _other_place_caption(caption, name) or _NON_PHOTO.search(heading)
                    or (not caption and _NON_PHOTO.search(nearby))):
                continue
            explicit_name = len(normalized(name)) >= 2 and normalized(name) in normalized(caption + nearby)
            if explicit_name or _SCENE.search(caption):
                result[url] = 'indexed_place_caption'
        previous = ''
    for url, caption in descriptions.items():
        if url and url not in seen_markdown and not _NON_PHOTO.search(caption) and not _other_place_caption(caption, name) and len(normalized(name)) >= 2 and normalized(name) in normalized(caption):
            result.setdefault(url, 'indexed_place_alt')
    return result


class PageImages(HTMLParser):
    def __init__(self, page_url: str, place_name: str):
        super().__init__()
        self.page_url = page_url
        self.name = normalized(place_name)
        self.images: dict[str, str] = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta' and (attrs.get('property') or attrs.get('name')) in {'og:image','twitter:image'}:
            url = canonical_url(urljoin(self.page_url, attrs.get('content') or ''))
            if url:
                self.images[url] = 'place_page_cover'
        if tag == 'img':
            alt = normalized((attrs.get('alt') or '') + (attrs.get('title') or ''))
            if (len(self.name) >= 2 and self.name in alt and not _NON_PHOTO.search(alt)
                    and not _other_place_caption(alt, self.name)):
                for key in ('src','data-src','data-original'):
                    if attrs.get(key):
                        url = canonical_url(urljoin(self.page_url, attrs[key]))
                        if url:
                            self.images[url] = 'place_image_caption'


async def page_image_evidence(client, url: str, name: str, public_url) -> dict[str, str]:
    try:
        for _ in range(3):
            if not await public_url(url):
                return {}
            async with client.stream('GET', url, headers={'accept':'text/html'}) as response:
                if response.is_redirect:
                    url = str(response.url.join(response.headers.get('location','')))
                    continue
                response.raise_for_status()
                if 'text/html' not in response.headers.get('content-type',''):
                    return {}
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > 1_500_000:
                        break
            parser = PageImages(url, name)
            parser.feed(bytes(data).decode(response.encoding or 'utf-8', errors='replace'))
            return parser.images
    except Exception:
        return {}
    return {}
