"""Page-level image evidence. Related-page advertisements are not place photos."""
from html.parser import HTMLParser
from urllib.parse import urljoin
from app.services.offline.content import canonical_url, normalized


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
