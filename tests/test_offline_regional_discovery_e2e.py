"""Web publish → real search adapter → authenticated HTTP → DB → local photos.

Search/model/image CDN responses are reproducible external fixtures. Application
discovery, verification, JPEG validation, persistence and serialization are real.
"""
import io
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from PIL import Image, ImageDraw

from app.api.public import offline
from app.services.offline import activity_generation as gen, activity_images as media
from app.services.offline import activity_service as service, repository as repo
from app.services.offline.activity_discovery import ACTIVITY_PLACE_CATEGORIES, ordered_categories, event_facts, source_address
from app.services.offline.chat_emit import build_activity_component_card
from app.services.offline.providers import search
from app.services.prompting import store
from tests.test_offline_quality_e2e import journey  # noqa: F401
from tests.test_proactive_activity_e2e import flow  # noqa: F401

RELEASE = json.loads((Path(__file__).parents[1] / 'scripts/prompt_releases/20261006_regional_discovery.json').read_text())


@pytest.fixture
async def discovery(journey, monkeypatch, tmp_path):
    j = journey
    entry = RELEASE['prompts'][0]
    before = await j.db.prompttemplate.find_unique(where={'key': entry['key']})
    history = await j.db.prompttemplateversion.find_many(where={'promptKey': entry['key']})
    result = await j.client.put('/admin-api/prompts/' + entry['key'], json={
        'content': entry['content'], 'expected_updated_at': before.updatedAt.isoformat(),
    })
    assert result.status_code == 200, result.text
    after = await j.db.prompttemplate.find_unique(where={'key': entry['key']})
    versions = await j.db.prompttemplateversion.find_many(where={'promptKey': entry['key']})
    added = [v for v in versions if v.id not in {v.id for v in history}]
    assert len(added) == 1 and added[0].changeType == 'manual_save'
    assert after.defaultContent == before.defaultContent and after.isEnabled == before.isEnabled
    assert entry['content'] == after.content == await j.redis.get(store._redis_key(entry['key']))
    assert str(await store.get_prompt_text(entry['key'])) == entry['content']
    stale = await j.client.put('/admin-api/prompts/' + entry['key'], json={
        'content': 'stale', 'expected_updated_at': before.updatedAt.isoformat(),
    })
    assert stale.status_code == 409
    ctx = dict(user_id=j.user, agent_id=j.agent, workspace_id=j.workspace,
               conversation_id=j.conversation, user_location_city='镇江市',
               user_location_region='京口区', agent_name='小伴', has_location=True)
    monkeypatch.setattr(repo, 'resolve_user_context', AsyncMock(return_value=ctx))
    monkeypatch.setattr(repo, 'list_user_tags', AsyncMock(return_value=[]))
    monkeypatch.setattr(repo, 'memory_brief', AsyncMock(return_value=''))
    monkeypatch.setattr(offline, 'is_activity_enabled', AsyncMock(return_value=True))
    monkeypatch.setattr(service, 'geocode_address', AsyncMock(return_value=None))
    monkeypatch.setattr(service, 'generate_activity_invite_message', AsyncMock(return_value='找到个地方，想看看吗'))
    monkeypatch.setattr(gen, '_recommendation_copy', AsyncMock(return_value=''))
    class SearchCache:
        async def get(self, key):
            return await j.redis.get(store._redis_key('search:' + key))
        async def set(self, key, value, **kwargs):
            return await j.redis.set(store._redis_key('search:' + key), value, **kwargs)
    monkeypatch.setattr(search, 'get_redis', AsyncMock(return_value=SearchCache()))
    monkeypatch.setattr(search.settings, 'tavily_api_key', 'external-fixture-key')
    monkeypatch.setattr(search.settings, 'tavily_search_endpoint', 'https://search.fixture.test/search')
    monkeypatch.setattr(media.storage, '_MEDIA_DIR', tmp_path)
    async def public(url):
        return httpx.URL(url).host in {'photo.fixture.test', 'source.fixture.test'}
    monkeypatch.setattr(media, '_public_url', public)
    j.queries, j.downloads, j.prompts = [], [], []
    j.rows, j.proposals = [], {'candidates': []}
    blobs = []
    for index in range(3):
        image = Image.new('RGB', (320, 240), 'white')
        draw = ImageDraw.Draw(image)
        for x in range(8):
            for y in range(8):
                if ((x * 11 + y * 7 + index * 17) % (3 + index)) == 0:
                    draw.rectangle((x * 40, y * 30, x * 40 + 35, y * 30 + 25), fill='black')
        stream = io.BytesIO(); image.save(stream, 'JPEG'); blobs.append(stream.getvalue())
    original = httpx.AsyncClient
    async def transport(request):
        if request.url.host == 'search.fixture.test':
            payload = json.loads(request.content)
            j.queries.append(payload['query'])
            assert payload['search_depth'] == 'basic' and payload['include_raw_content'] == 'markdown'
            rows = j.search_rows(payload['query']) if hasattr(j, 'search_rows') else j.rows
            return httpx.Response(200, json={'results': rows, 'images': ['https://photo.fixture.test/unbound.jpg']})
        if request.url.host == 'photo.fixture.test':
            j.downloads.append(str(request.url))
            try:
                index = int(request.url.path.rsplit('/', 1)[-1].split('.')[0])
            except ValueError:
                raise AssertionError('Unbound or unrelated picture must never be downloaded')
            return httpx.Response(200, content=blobs[index], headers={'content-type': 'image/jpeg'})
        return httpx.Response(404)
    class ExternalClient(original):
        def __init__(self, *args, **kwargs):
            kwargs['transport'] = httpx.MockTransport(transport)
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(httpx, 'AsyncClient', ExternalClient)
    async def model(_, prompt):
        j.prompts.append(prompt)
        assert 'candidates' in prompt and '搜索位置锚点：镇江市 京口区' in prompt
        assert '{sources_json}' not in prompt and '{search_anchor}' not in prompt
        return json.dumps(j.proposals, ensure_ascii=False)
    # The model transport is external; avoid cached HTTP client constructors
    # when replacing the search/image HTTP boundary in this fixture.
    monkeypatch.setattr(gen, 'get_chat_model', lambda: object())
    monkeypatch.setattr(gen, 'invoke_text', model)
    j.source = 'https://source.fixture.test/' + uuid4().hex
    def row(name, content='', *, images=True, title=None):
        heading = title or name
        raw = '# ' + heading + '\n' + content + '\n'
        if images:
            raw += '\n'.join(f'![{name}现场照片 {i}](https://photo.fixture.test/{i}.jpg)' for i in range(3))
        return dict(title=heading, url=j.source + '/' + name, content=content, raw_content=raw)
    j.row = row
    return j


@pytest.mark.parametrize('name,category', [
    ('巷口咖啡(江滨店)', '咖啡与茶饮'), ('Blue Bottle', '咖啡与茶饮'),
    ('青禾展览馆', '展览空间'), ('镇江博物馆', '展览与博物馆'),
    ('镇江市图书馆', '阅读与文化'), ('山间独立书店', '书店阅读'),
    ('南山公园', '公园与绿地'), ('老周苍蝇馆', '街头小吃'),
    ('阿婆生煎', '小吃与轻食'), ('小巷面馆', '餐饮小店'),
    ('陶趣工坊', '手作与小店'), ('桥畔唱片店', '唱片与旧物'),
    ('江滨夜市', '市集夜市'), ('小岛茶室', '茶饮小店'),
])
async def test_real_generation_supports_diverse_named_places_and_three_photos(discovery, name, category):
    j = discovery
    address = '镇江市京口区江滨路18号'
    j.rows = [j.row(name, f'名称：{name}\n地址：{address}\n类型：{category}。')]
    # A malformed/empty model pool still has source-grounded deterministic recovery.
    response = await j.client.post('/offline/activities/recommend')
    assert response.status_code == 200, response.text
    card = response.json()
    assert card['location_name'] == name and card['address'] == address
    assert card['category'] == category
    assert len(j.prompts) == 1
    assert len(card['image_urls']) == 3 and len(set(card['image_urls'])) == 3
    assert all('京口区' in query for query in j.queries[:6])
    assert 'https://photo.fixture.test/unbound.jpg' not in j.downloads
    detail = (await j.client.get('/offline/activities/' + card['id'])).json()
    assert detail['image_urls'] == card['image_urls'] and detail['summary'] == card['summary']
    component = build_activity_component_card(detail, status_label='待确定')
    assert component['title'] == card['title'] and component['payload']['image_url'] == card['image_urls'][0]
    for url in card['image_urls']:
        result = await j.client.get(url)
        assert result.status_code == 200 and result.headers['content-type'] == 'image/jpeg'
        assert Image.open(io.BytesIO(result.content)).size == (320, 240)


async def test_collection_multiple_leads_require_independent_local_branch_verification(discovery):
    j = discovery
    article = j.row('镇江这些咖啡店', '镇江市京口区\n## 阿婆生煎\n## Blue Bottle\n## 无影咖啡店', images=False)
    local = j.row('Blue Bottle', '名称：Blue Bottle\n地址：镇江市京口区江滨路18号\n类型：咖啡店')
    wrong_city = j.row('阿婆生煎', '地址：上海市黄浦区北京路1号')
    j.proposals = {'candidates': [
        {'location_name': name, 'official_url': article['url'], 'address': '模型编造的地址'}
        for name in ('阿婆生煎', 'Blue Bottle', '虚构不存在的店')
    ]}
    def rows(query):
        if query == '镇江市 阿婆生煎 地址': return [wrong_city]
        if query == '镇江市 Blue Bottle 地址': return [local]
        if query == '镇江市 无影咖啡店 地址': return []
        return [article]
    j.search_rows = rows
    response = await j.client.post('/offline/activities/recommend')
    assert response.status_code == 200, response.text
    card = response.json()
    assert card['location_name'] == 'Blue Bottle' and card['official_url'] == local['url']
    assert card['address'] == '镇江市京口区江滨路18号' and len(card['image_urls']) == 3
    assert not any('虚构不存在的店' in query for query in j.queries)
    assert {s['url'] for s in card['search_sources'] if s.get('url')} == {local['url'], *j.downloads}


@pytest.mark.parametrize('window,valid', [('current', True), ('expired', False), ('yearless', False), ('missing', False)])
async def test_temporary_event_cannot_be_recycled_as_permanent_place(discovery, window, valid):
    j = discovery
    now = datetime.now(UTC)
    first, last = now - timedelta(days=1), now + timedelta(days=1)
    if window == 'expired': first, last = now - timedelta(days=8), now - timedelta(days=7)
    date = f'{first:%Y年%m月%d日} 至 {last:%Y年%m月%d日}'
    if window == 'yearless': date = f'{first:%m月%d日} 至 {last:%m月%d日}'
    if window == 'missing': date = '本周末'
    j.rows = [j.row('青禾广场', f'镇江市京口区\n活动时间：{date}\n举办地点：青禾广场\n地址：镇江市京口区江滨路18号',
                    title='小岛秋日市集活动 - 青禾广场')]
    # A model may try to remove the dates; the source event identity wins.
    j.proposals = {'candidates': [{'location_name': '青禾广场', 'official_url': j.rows[0]['url'], 'starts_at': None, 'ends_at': None}]}
    response = await j.client.post('/offline/activities/recommend')
    if valid:
        assert response.status_code == 200, response.text
        card = response.json()
        assert card['category'] == '活动现场' and card['location_name'] == '青禾广场'
        assert card['starts_at'] and card['ends_at'] and len(card['image_urls']) == 3
        assert datetime.fromisoformat(card['expires_at']) <= datetime.fromisoformat(card['ends_at'])
    else:
        assert response.status_code == 503 and response.json()['detail']['reason'] == 'no_suitable_activity'
        assert not j.downloads
        assert await j.db.query_raw('SELECT id FROM offline_activity_recommendations WHERE user_id=$1', j.user) == []
        service.emit_activity_card.assert_not_awaited()


async def test_region_ranking_repeated_generation_and_source_failure(discovery):
    j = discovery
    # No photo evidence still produces a real place; never use unbound query images.
    j.rows = [j.row('城外咖啡店', '地址：镇江市丹徒区远山路1号', images=False),
              j.row('小巷面馆', '地址：镇江市京口区江滨路18号', images=False),
              j.row('山间书店', '地址：镇江市京口区读书路19号', images=False)]
    names = []
    for _ in range(3):
        response = await j.client.post('/offline/activities/recommend')
        assert response.status_code == 200, response.text
        names.append(response.json()['location_name'])
        assert response.json()['image_urls'] == []
    assert names[:2] == ['小巷面馆', '山间书店'] and len(set(names)) == 3
    response = await j.client.post('/offline/activities/recommend')
    assert response.status_code == 503


def test_exploration_reaches_every_category_and_balances_families():
    reached = set()
    for size in range(21):
        history = [{'location_name': '已推荐的其他场所'}] * size
        categories = ordered_categories(history, [], 'same-user')
        assert len({c.family for c in categories[:8]}) >= 6
        reached.update(c.name for c in categories[:8])
    assert reached == {c.name for c in ACTIVITY_PLACE_CATEGORIES}


def test_event_year_and_invalid_calendar_dates_are_not_guessed():
    from app.services.offline.providers.search import SearchResult
    now = datetime(2026, 10, 6, tzinfo=UTC)
    def event(window):
        return SearchResult(title='临时活动', url='https://example.test', content=f'时间：{window}\n地点：青禾广场')
    assert event_facts(event('2026年10月6日 至 2026年10月8日'), now)['ends_at'].startswith('2026-10-08')
    for window in ('10月6日 至 10月8日', '2026年2月30日', '2025年10月6日', '2027年10月6日'):
        assert event_facts(event(window), now) == {}
    assert event_facts(event('2026年10月6日 09:00-10:00'), now) == {}  # Already ended at 18:00 CST.
    assert event_facts(event('2026年10月6日-8日'), now) == {}  # Incomplete end date.
    permanent = SearchResult(title='青禾音乐厅', url='https://example.test', content='镇江市，演出和音乐会场所')
    assert event_facts(permanent, now) is None


async def test_directory_and_navigation_address_never_become_a_destination(discovery):
    j = discovery
    j.rows = [j.row('【镇江京口区百货商场企业名录】',
                    '地址：镇江市京口区九里街1号1幢第一层103室\n注册资本：1万（万元）\n黄页企业单位名录', images=False)]
    response = await j.client.post('/offline/activities/recommend')
    assert response.status_code == 503 and not j.downloads
    from app.services.offline.providers.search import SearchResult
    dirty = SearchResult(title='南景饭店', url='https://example.test',
                         content='地址：首页 » 欢迎光临 公司介绍\n地址：镇江市京口区江滨路18号')
    assert source_address(dirty) == '镇江市京口区江滨路18号'


async def test_model_and_cache_outage_keep_a_verified_source_destination(discovery, monkeypatch):
    j = discovery
    j.rows = [j.row('山间书店', '地址：镇江市京口区读书路19号')]
    monkeypatch.setattr(gen, 'invoke_text', AsyncMock(side_effect=RuntimeError('model unavailable')))
    monkeypatch.setattr(search, 'get_redis', AsyncMock(side_effect=RuntimeError('cache unavailable')))
    response = await j.client.post('/offline/activities/recommend')
    assert response.status_code == 200 and response.json()['location_name'] == '山间书店'
    assert len(response.json()['image_urls']) == 3


async def test_gallery_window_binds_the_destination_before_truncating_sources(discovery):
    from app.services.offline.providers.search import SearchResult
    j = discovery
    raw = j.row('小岛茶室', '地址：镇江市京口区江滨路18号')
    chosen = SearchResult(title=raw['title'], url=raw['url'], content=raw['content'], raw_content=raw['raw_content'])
    unrelated = [SearchResult(title=f'别处公园{i}', url=f'https://source.fixture.test/{i}', content='南京市') for i in range(9)]
    card = dict(location_name='小岛茶室', city='镇江市', address='镇江市京口区江滨路18号')
    urls = await media.persist_activity_images(user_id=j.user, card=card, city='镇江市', search_results=unrelated + [chosen])
    assert len(urls) == 3 and len(j.downloads) == 3


async def test_discovery_deadline_retains_a_completed_query_while_peer_is_slow(monkeypatch):
    from app.services.offline.providers.search import SearchResult
    monkeypatch.setattr(gen, '_DISCOVERY_BUDGET_S', .05)
    monkeypatch.setattr(gen, '_search_query_specs', lambda *args: [gen.SearchQuerySpec('fast'), gen.SearchQuerySpec('slow')])
    cancelled = []
    async def query(text, **kwargs):
        if text == 'fast':
            return [SearchResult(title='镇江博物馆', url='https://example.test/museum', content='镇江市')]
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
    monkeypatch.setattr(gen, 'tavily_search', query)
    filtered, all_results, _ = await gen._search_activity_candidates('镇江市', [], [], city='镇江市')
    assert filtered == all_results and filtered[0].title == '镇江博物馆' and cancelled


async def test_malformed_model_copy_cannot_corrupt_verified_facts_or_task(discovery):
    j = discovery
    j.rows = [j.row('青禾展览馆', '地址：镇江市京口区江滨路18号')]
    j.proposals = {'candidates': [{'location_name': '青禾展览馆', 'official_url': j.rows[0]['url'],
                                 'title': '去上海外滩逛街', 'summary': {'bad': 'object'},
                                 'address': '上海市', 'easter_egg_task': 'not an object'}]}
    response = await j.client.post('/offline/activities/recommend')
    assert response.status_code == 200, response.text
    card = response.json()
    assert card['title'] == '青禾展览馆' and card['address'] == '镇江市京口区江滨路18号'
    assert isinstance(card['summary'], str) and 'object' not in card['summary']
    task_rows = await j.db.query_raw('SELECT easter_egg_task FROM offline_activity_recommendations WHERE id=$1', card['id'])
    task = task_rows[0]['easter_egg_task']
    task = json.loads(task) if isinstance(task, str) else task
    assert isinstance(task, dict) and task['title'] == '小彩蛋任务'
