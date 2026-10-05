import io
from unittest.mock import AsyncMock
import httpx
from PIL import Image
import pytest

from app.services.offline import activity_images as images, activity_generation as generation
from app.services.offline import activity_media_storage as storage
from app.services.offline.content import plain_text, place_source_matches
from app.services.offline.providers.search import _result_from_item, SearchResult

CARD={'location_name':'莲湖公园','city':'东莞','address':'桥头镇莲湖路','official_url':'https://example.com/lianhu'}


def source(urls, title='东莞桥头莲湖公园', content='东莞桥头镇莲湖公园的现场照片'):
    return _result_from_item({'title':title,'url':CARD['official_url'],'content':content,'images':urls})


def test_nested_tavily_images_keep_provenance_and_both_shapes():
    parsed=source(['https://img.example/a.jpg',{'url':'https://img.example/b.jpg','description':'公园湖边'}])
    assert len(parsed.images)==2
    assert all(i['source_url']==CARD['official_url'] for i in parsed.images)
    assert parsed.images[1]['description']=='公园湖边'


@pytest.mark.parametrize('title,content',[('西安莲湖公园','西安市莲湖区'),('东莞周末去哪玩十大公园','莲湖公园以及其他公园'),('东莞人民公园','莲湖公园在另一边')])
def test_rejects_other_city_listicle_or_other_place(title,content):
    assert not place_source_matches(CARD,title,content)
    assert not images._source_images(CARD,[source(['https://img.example/a.jpg'],title,content)])


def test_card_source_requires_exact_page_not_just_domain():
    card={**CARD,'official_url':'https://example.com/different'}
    assert not generation._card_is_source_backed(card,[{'url':CARD['official_url'],'title':'东莞莲湖公园','content':'东莞桥头'}])
    assert generation._card_is_source_backed(CARD,[{'url':CARD['official_url'],'title':'东莞莲湖公园','content':'东莞桥头'}])


async def test_gallery_refills_after_failed_or_duplicate_downloads_and_reuses_cache(monkeypatch,tmp_path):
    cache={}
    async def save(card,data):cache.update(images=data)
    monkeypatch.setattr(images.place_catalog,'load_place',AsyncMock(side_effect=lambda _:cache or None))
    monkeypatch.setattr(images.place_catalog,'save_place',save)
    monkeypatch.setattr(storage,'_MEDIA_DIR',tmp_path)
    monkeypatch.setattr(images,"page_image_evidence",AsyncMock(return_value={f"https://img.example/{i}":"place_image_caption" for i in ["bad",1,2,3,4]}))
    calls=[]
    async def download(client,url):
        calls.append(url)
        if url.endswith('bad'):return None
        n=int(url.rsplit('/',1)[1])
        # 1 and 2 are duplicate pictures, despite distinct URLs.
        n=1 if n==2 else n
        return bytes([n]),str(n)*64,{1:0,3:65535,4:281474976645120}[n]
    monkeypatch.setattr(images,'_download_image',download)
    search=AsyncMock(side_effect=[[source(['https://img.example/2','https://img.example/3'])],[source(['https://img.example/4'])]])
    monkeypatch.setattr(images,'tavily_place_images',search)
    card=dict(CARD)
    gallery=await images.persist_activity_images(user_id='u1',card=card,city='东莞',search_results=[source(['https://img.example/bad','https://img.example/1'])])
    assert len(gallery)==3 and search.await_count==2
    assert len(card['image_provenance'])==3
    assert all('source_url' in i for i in cache['images'])
    again=await images.persist_activity_images(user_id='another-user',card=dict(CARD),city='东莞',search_results=[])
    assert again==gallery and search.await_count==2
    assert all('another-user' not in url and 'u1' not in url for url in again)


async def test_no_valid_images_never_fills_with_llm_or_stock_urls(monkeypatch,tmp_path):
    monkeypatch.setattr(images.place_catalog,'load_place',AsyncMock(return_value=None))
    monkeypatch.setattr(images.place_catalog,'save_place',AsyncMock())
    monkeypatch.setattr(images,'tavily_place_images',AsyncMock(return_value=[]))
    download=AsyncMock();monkeypatch.setattr(images,'_download_image',download)
    card={**CARD,'image_urls':['https://unsplash.com/random.jpg']}
    assert await images.persist_activity_images(user_id='u',card=card,city='东莞',search_results=[])==[]
    download.assert_not_called()


async def test_image_download_validates_pixels_redirects_and_size(monkeypatch):
    raw=io.BytesIO();Image.new('RGB',(500,400),(30,90,150)).save(raw,format='PNG')
    monkeypatch.setattr(images,'_public_url',AsyncMock(side_effect=lambda u:'private' not in u))
    def handler(request):
        if request.url.path=='/redirect':return httpx.Response(302,headers={'location':'http://private/image'})
        if request.url.path=='/html':return httpx.Response(200,headers={'content-type':'text/html'},content=b'error')
        if request.url.path=='/broken':return httpx.Response(200,headers={'content-type':'image/png'},content=b'not an image')
        return httpx.Response(200,headers={'content-type':'image/png'},content=raw.getvalue())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await images._download_image(client,'https://img.example/good')
        for suffix in ('redirect','html','broken'):
            assert await images._download_image(client,'https://img.example/'+suffix) is None


def test_recommendation_text_removes_urls_but_preserves_readable_labels():
    assert plain_text('看看[活动介绍](https://example.com/page)，https://bad.example/link。')=='看看活动介绍，。'


def test_page_images_reject_ads_and_keep_place_captions_and_cover():
    from app.services.offline.image_evidence import PageImages
    parser=PageImages('https://example.com/park','莲湖公园')
    parser.feed('<meta property="og:image" content="/hero.jpg"><img alt="莲湖公园荷花" src="/flower.jpg"><img alt="相关酒店广告" src="/ad.jpg">')
    assert parser.images=={'https://example.com/hero.jpg':'place_page_cover','https://example.com/flower.jpg':'place_image_caption'}
