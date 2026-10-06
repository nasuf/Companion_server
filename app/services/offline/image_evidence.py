"""Page-level image evidence. Related-page advertisements are not place photos."""
from html.parser import HTMLParser
import re
from urllib.parse import urljoin
from app.services.offline.content import canonical_url, normalized

_IMAGE = re.compile(r'!\[([^\]]*)\]\((https?://[^\s)]+)(?:\s+"[^"]*")?\)')
_NON_PHOTO = re.compile(r'logo|icon|二维码|广告|标识|标志|示意图|效果图|设计方案|规划图|头像|海报|订阅|吉祥物', re.I)
_SCENE = re.compile(r'外观|外景|实景|全景|内景|馆舍|建筑|大门|正门|入口|馆内|阅览|风景|湖面|绿道|步道|远景|近景|侧视|匾额|展厅|庭院|花园')


def indexed_image_evidence(raw: str, name: str, images: list[dict]) -> dict[str, str]:
    """Use crawler-preserved captions, including captions outside the img alt.

    Caller must first verify the page's place/city. Page fetch availability is
    not evidence of image relevance; a 403 must not invalidate indexed captions.
    Header/footer links and a page's generic title alone are insufficient.
    """
    result = {}
    descriptions = {canonical_url(i.get('url')): str(i.get('description') or '')
                    for i in images if i.get('description_source') in {'alt', 'caption'}}
    seen_markdown = set()
    previous = ''
    heading = ''
    for block in raw.split('\n\n'):
        matches = list(_IMAGE.finditer(block))
        if not matches:
            text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', block).strip()
            if text.startswith('#'):
                heading = text.lstrip('# ').strip()
            previous = text[-150:]
            continue
        # Remove images and link destinations before looking for a caption.
        text = _IMAGE.sub('', block)
        text = re.sub(r'\[([^\]]*)\]\([^)]*\)|\]\([^)]*\)', r'\1', text).strip()
        nearby = (previous if len(previous) <= 80 and not previous.startswith('#') else '') + ' ' + text
        for match in matches:
            url = canonical_url(match.group(2))
            seen_markdown.add(url)
            alt = re.sub(r'^Image\s*\d*\s*:?\s*', '', match.group(1), flags=re.I)
            caption = (alt or descriptions.get(url, '')).strip()
            if _NON_PHOTO.search(caption) or _NON_PHOTO.search(heading) or _NON_PHOTO.search(nearby):
                continue
            explicit_name = len(normalized(name)) >= 2 and normalized(name) in normalized(caption + nearby)
            if explicit_name or (_SCENE.search(caption) and not _NON_PHOTO.search(nearby)):
                result[url] = 'indexed_place_caption'
        previous = ''
    # Some extraction formats omit markdown images but retain native img alt.
    for url, caption in descriptions.items():
        if url and url not in seen_markdown and not _NON_PHOTO.search(caption) and len(normalized(name)) >= 2 and normalized(name) in normalized(caption):
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
            if len(self.name) >= 2 and self.name in alt:
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
