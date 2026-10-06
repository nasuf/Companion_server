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
    return _result_from_item({'title':title,'url':CARD['official_url'],'content':content,'images':urls,'raw_content':'\n\n'.join('![莲湖公园实景]('+ (u if isinstance(u,str) else u['url']) +')' for u in urls)})


def test_nested_tavily_images_keep_provenance_and_both_shapes():
    parsed=source(['https://img.example/a.jpg',{'url':'https://img.example/b.jpg','description':'公园湖边'}])
    assert len(parsed.images)==2
    assert all(i['source_url']==CARD['official_url'] for i in parsed.images)
    assert parsed.images[1]['description']=='公园湖边'


def test_bound_cafe_scene_captions_do_not_admit_other_named_venues():
    from app.services.offline.image_evidence import indexed_image_evidence
    raw = '''# 小岛咖啡馆

![门头](https://photo.example/front.jpg)

![店内](https://photo.example/inside.jpg)

![吧台](https://photo.example/bar.jpg)

![南京别处咖啡馆店内](https://photo.example/wrong.jpg)
'''
    found = indexed_image_evidence(raw, '小岛咖啡馆', [])
    assert set(found) == {f'https://photo.example/{name}.jpg' for name in ('front', 'inside', 'bar')}
    assert indexed_image_evidence('![南京图书馆阅览区](https://photo.example/wrong.jpg)', '镇江市图书馆', []) == {}


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


async def test_unverified_candidates_do_not_starve_later_verified_sources(monkeypatch,tmp_path):
    monkeypatch.setattr(images.place_catalog,'load_place',AsyncMock(return_value=None))
    monkeypatch.setattr(images.place_catalog,'save_place',AsyncMock())
    monkeypatch.setattr(storage,'_MEDIA_DIR',tmp_path)
    bad=SearchResult(title='东莞莲湖公园', url=CARD['official_url'], content='东莞莲湖公园', images=[{'url':f'https://blocked.example/{i}.jpg'} for i in range(50)])
    good=SearchResult(title='东莞莲湖公园实景',url='https://official.example/park',content='东莞莲湖公园',images=[{'url':'https://photo.example/park.jpg'}])
    async def evidence(client,url,*args):
        return {'https://photo.example/park.jpg':'place_image_caption'} if 'official' in url else {}
    monkeypatch.setattr(images,'page_image_evidence',evidence)
    download=AsyncMock(return_value=(b'image','c'*64,123))
    monkeypatch.setattr(images,'_download_image',download)
    monkeypatch.setattr(images,'tavily_place_images',AsyncMock(return_value=[]))
    gallery=await images.persist_activity_images(user_id='u',card=dict(CARD),city='东莞',search_results=[bad,good])
    assert len(gallery)==1
    download.assert_awaited_once()


def test_indexed_captions_survive_origin_block_and_reject_logo_renderings():
    from app.services.offline.image_evidence import indexed_image_evidence
    raw = """# 镇江市图书馆

镇江市图书馆

[![Image 3](https://cdn.example/front.jpg)](https://baike.example/library)[![Image 4](https://cdn.example/inside.jpg)](https://baike.example/library)

### 形象标识

![Image 6](https://cdn.example/brand.jpg)

### 新馆规划

镇江市图书馆效果图

![Image 7](https://cdn.example/render.jpg)
"""
    result = indexed_image_evidence(raw, '镇江市图书馆', [dict(url='https://cdn.example/brand.jpg', description='镇江市图书馆', description_source='alt')])
    assert set(result) == {'https://cdn.example/front.jpg', 'https://cdn.example/inside.jpg'}
    assert indexed_image_evidence('# 莲湖公园\n\n![Image 1](https://cdn.example/ad.jpg)', '莲湖公园', []) == {}


async def test_indexed_photos_do_not_require_origin_html(monkeypatch, tmp_path):
    monkeypatch.setattr(images.place_catalog, 'load_place', AsyncMock(return_value=None))
    monkeypatch.setattr(images.place_catalog, 'save_place', AsyncMock())
    monkeypatch.setattr(storage, '_MEDIA_DIR', tmp_path)
    fetch = AsyncMock(side_effect=AssertionError('indexed captions suffice even if origin returns 403'))
    monkeypatch.setattr(images, 'page_image_evidence', fetch)
    monkeypatch.setattr(images, '_download_image', AsyncMock(return_value=(b'image', 'd'*64, 42)))
    monkeypatch.setattr(images, 'tavily_place_images', AsyncMock(return_value=[]))
    assert len(await images.persist_activity_images(user_id='u', card=dict(CARD), city='东莞', search_results=[source(['https://photo.example/real.jpg'])])) == 1
    fetch.assert_not_called()


@pytest.mark.parametrize('name', ['镇江这些咖啡店', '镇江十大咖啡馆', '镇江市2025高德状元榜·美食', '在道滘玩得开心吗？_东莞市人民政府门户网站', '从老照片聊聊镇江的古董店'])
def test_collection_and_article_titles_are_not_place_identities(name):
    card = dict(location_name=name, city='镇江市', title=name, address=name)
    assert not generation._card_has_concrete_place(card, '镇江市')
    assert not generation._fallback_card('镇江市', [], [SearchResult(title=name, url='https://example.com/article', content='镇江市的地点汇总')])
    assert not place_source_matches(card, name, '镇江市')


def test_fallback_preserves_cafe_branch_name_and_rejects_listicle_first():
    article = SearchResult(title='镇江这些咖啡店，藏着整个春天！', url='https://example.com/list', content='镇江市咖啡店合集')
    cafe = SearchResult(title='库迪咖啡(临湖苑店) - 镇江市句容市经济开发区- 餐饮服务', url='https://example.com/place', content='库迪咖啡(临湖苑店)，镇江市句容市临湖苑商业B2幢110号')
    card = generation._fallback_card('镇江市', [], [article, cafe])
    assert card['location_name'] == '库迪咖啡(临湖苑店)'
    assert card['title'] == '库迪咖啡(临湖苑店)'
    assert generation._card_has_concrete_place(card, '镇江市')
    card['city'] = '镇江市'
    assert generation._card_is_source_backed(card, generation._sources([cafe]))


def test_indexed_museum_link_captions_do_not_share_plan_image_rejection():
    from app.services.offline.image_evidence import indexed_image_evidence
    raw = '''# 镇江博物馆

## 建筑格局

[![Image 3](https://cdn.example/plan.jpg)](https://example.com/album "镇江市博物馆平面示意图")镇江市博物馆平面示意图

## 展览陈列

青铜文化的展览。[![Image 4](https://cdn.example/bronze.jpg)](https://example.com/album "青铜器展")青铜器展
陶瓷文化的展览。[![Image 5](/photos/ceramic.jpg)](https://example.com/album "陶瓷器精品展")陶瓷器精品展
金银器的展览。[![Image 6](//cdn.example/silver.jpg)](https://example.com/album "金银器精品展")金银器精品展

### 相关广告

![镇江博物馆附近酒店](https://cdn.example/ad.jpg)
'''
    evidence = indexed_image_evidence(raw, '镇江博物馆', [], 'https://example.com/museum')
    assert set(evidence) == {'https://cdn.example/bronze.jpg', 'https://example.com/photos/ceramic.jpg', 'https://cdn.example/silver.jpg'}


def test_photo_context_does_not_borrow_the_next_places_caption():
    from app.services.offline.image_evidence import indexed_image_evidence
    raw = '# 莲湖公园\n\n![广告](https://example.com/ad.jpg)\n莲湖公园实景\n![莲湖公园大门](https://example.com/gate.jpg)'
    assert set(indexed_image_evidence(raw, '莲湖公园', [])) == {'https://example.com/gate.jpg'}


@pytest.mark.parametrize('options', [
    dict(activity_id=['one']),
    dict(refill_incomplete=True),
    dict(refill_incomplete=True, activity_id=['one'], fill_only=True),
])
async def test_gallery_repair_rejects_ambiguous_scope(options, monkeypatch):
    from scripts import repair_offline_galleries as repair
    from types import SimpleNamespace
    connect = AsyncMock()
    monkeypatch.setattr(repair, 'db', SimpleNamespace(connect=connect))
    with pytest.raises(ValueError):
        await repair.run(SimpleNamespace(**options))
    connect.assert_not_awaited()
