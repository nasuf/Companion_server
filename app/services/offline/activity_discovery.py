"""Region discovery policy and source facts, independent of model phrasing.

Categories are search intents, never a city/place database. History balances
exploration; explicit dates distinguish temporary events from permanent venues.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

from app.services.offline.content import concrete_place_name, normalized
from app.services.offline.providers.search import SearchResult


@dataclass(frozen=True)
class ActivityPlaceCategory:
    name: str
    keywords: tuple[str, ...]
    family: str
    suitable: str

    @property
    def query_hint(self) -> str:
        return ' '.join(self.keywords)


ACTIVITY_PLACE_CATEGORIES = (
    ActivityPlaceCategory('阅读与文化', ('图书馆', '城市书房'), '文化', '阅读、看看书架'),
    ActivityPlaceCategory('书店阅读', ('书店', '旧书店', '独立书店'), '文化', '翻书、逛书架'),
    ActivityPlaceCategory('展览与博物馆', ('博物馆', '纪念馆', '非遗馆'), '文化', '看展、留意展品'),
    ActivityPlaceCategory('展览空间', ('展览馆', '美术馆', '艺术馆', '画廊'), '文化', '看展、看看作品'),
    ActivityPlaceCategory('咖啡与茶饮', ('咖啡馆', '咖啡店', '咖啡', 'coffee'), '饮食', '喝一杯、坐一会儿'),
    ActivityPlaceCategory('茶饮小店', ('茶馆', '茶室', '茶饮店'), '饮食', '喝茶、歇一会儿'),
    ActivityPlaceCategory('小吃与轻食', ('小吃', '生煎', '锅盖面', '馄饨', '米粉'), '饮食', '尝尝小吃、随意逛逛'),
    ActivityPlaceCategory('街头小吃', ('苍蝇馆', '路边摊', '大排档', '夜宵', '档口'), '饮食', '尝尝本地味道'),
    ActivityPlaceCategory('餐饮小店', ('面馆', '饭馆', '小馆', '饭店', '餐厅', '火锅', '烧烤', '私房菜', '小炒'), '饮食', '吃顿饭、尝尝家常菜'),
    ActivityPlaceCategory('烘焙甜品', ('面包店', '甜品店', '烘焙店', '蛋糕店'), '饮食', '尝尝甜点、喝点东西'),
    ActivityPlaceCategory('公园与绿地', ('公园', '植物园', '湿地公园', '花园'), '户外', '散步、看看绿意'),
    ActivityPlaceCategory('水边散步', ('滨江步道', '码头', '湖边', '江边'), '户外', '沿水边走走'),
    ActivityPlaceCategory('山与轻户外', ('森林公园', '郊野公园', '古道', '观景台'), '户外', '轻徒步、看看风景'),
    ActivityPlaceCategory('街区与市集', ('老街', '古街', '步行街', '文旅街区'), '城市', '逛街、看看街景'),
    ActivityPlaceCategory('市集夜市', ('夜市', '市集', '菜市场'), '城市', '逛摊位、看看日常'),
    ActivityPlaceCategory('城市观察', ('老建筑', '老厂房', '创意园', '渡口'), '城市', '走走看看、拍细节'),
    ActivityPlaceCategory('安静角落', ('书院', '寺庙', '社区中心', '市民中心'), '城市', '安静走走、看看建筑'),
    ActivityPlaceCategory('手作与小店', ('陶艺', '手作店', '工坊', '画室'), '体验', '看看手作、尝试体验'),
    ActivityPlaceCategory('花艺文创', ('花店', '文创店', '杂货店', '花艺'), '体验', '逛小店、看看设计'),
    ActivityPlaceCategory('唱片与旧物', ('唱片店', '中古店', '旧物店', '古玩店'), '体验', '翻翻旧物、看看收藏'),
    ActivityPlaceCategory('商铺与零售', ('杂货铺', '百货', '商铺'), '城市', '随意逛逛'),
    ActivityPlaceCategory('室内避雨', ('文化中心', '商场', '购物中心'), '城市', '室内逛逛、坐一会儿'),
    ActivityPlaceCategory('演出与电影', ('剧场', '影院', '音乐厅', 'livehouse'), '文娱', '看电影、看看演出'),
    ActivityPlaceCategory('休闲社交', ('桌游', '台球', '小酒馆', 'KTV'), '文娱', '轻松玩一会儿'),
    ActivityPlaceCategory('轻运动', ('体育公园', '球场', '骑行绿道', '健身步道'), '运动', '活动一下、慢慢走'),
    ActivityPlaceCategory('亲子与乐园', ('动物园', '科普馆', '水族馆', '游乐场'), '体验', '看看动物、逛逛展馆'),
    ActivityPlaceCategory('活动现场', ('临时活动', '展览', '音乐节', '市集活动', '展会', '公开讲座'), '限时', '看看现场、感受氛围'),
)


def category_for(text: str) -> ActivityPlaceCategory | None:
    value = normalized(text)
    matches = [(len(normalized(k)), c) for c in ACTIVITY_PLACE_CATEGORIES
               for k in c.keywords if normalized(k) in value]
    return max(matches, key=lambda item: item[0])[1] if matches else None


def ordered_categories(recent: list[dict], tags: list[str], seed: str) -> list[ActivityPlaceCategory]:
    categories = list(ACTIVITY_PLACE_CATEGORIES)
    offset = (int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) if seed else 0)
    offset %= len(categories)
    categories = categories[offset:] + categories[:offset]
    counts: dict[str, int] = {}
    for item in recent[:20]:
        category = category_for(' '.join(str(item.get(k) or '') for k in ('category', 'location_name', 'title')))
        if category:
            counts[category.name] = counts.get(category.name, 0) + 1
    interest = normalized(' '.join(tags))
    families: dict[str, list[ActivityPlaceCategory]] = {}
    for category in categories:
        families.setdefault(category.family, []).append(category)
    for family, items in families.items():
        # Rotate each family independently so even families with many intents
        # get full coverage while history is capped at twenty recommendations.
        step = len(recent) % len(items)
        items = items[step:] + items[:step]
        items.sort(key=lambda c: (counts.get(c.name, 0),
                                 not any(normalized(k) in interest for k in c.keywords)))
        families[family] = items
    groups = list(families.values())
    step = len(recent) % len(groups)
    groups = groups[step:] + groups[:step]
    groups.sort(key=lambda items: (
        counts.get(items[0].name, 0),
        not any(normalized(k) in interest for k in items[0].keywords)))
    output: list[ActivityPlaceCategory] = []
    for position in range(max(map(len, groups))):
        output.extend(items[position] for items in groups if position < len(items))
    return output


ADDRESS_RE = re.compile(r'(?<!注册)(?:地址为|地址[：:]|位置[：:])\s*(?:\*\*)?([^。；;\n]{4,100})')
VENUE_RE = re.compile(r'(?:举办地点|活动地点|场馆|场地|地点|店名|名称)[：:]\s*(?:\*\*)?([^\n。；;，,]{2,60})')
_EVENT_RE = re.compile(r'音乐节|市集活动|展会|巡展|临时活动|演出|音乐会|讲座')
_EVENT_WINDOW_RE = re.compile(r'(?:活动时间|展期|举办时间|演出时间|市集时间)[：:]')
_DATE_RE = re.compile(r'(?:(?P<year>20\d{2})[年./-])?(?P<month>\d{1,2})[月./-](?P<day>\d{1,2})(?:日|号)?')
_TIME_ZONE = timezone(timedelta(hours=8))


def source_text(result: SearchResult) -> str:
    return result.content + '\n' + result.raw_content[:12000]


def source_address(result: SearchResult) -> str:
    for match in ADDRESS_RE.finditer(source_text(result)):
        value = match.group(1).strip().rstrip('* ')
        if re.search(r'首页|欢迎光临|公司介绍|企业名录|导航|注册资本|成立时间', value):
            continue
        if re.search(r'路|街|巷|弄|号|镇|村|大厦|广场|大楼|street|road', value, re.I):
            return value
    return ''


def event_facts(result: SearchResult, now: datetime | None = None) -> dict | None:
    """None=permanent place; {}=event with unverifiable/expired dates or venue.

    Require a labelled window with an explicit year. Never guess a year from a
    search timestamp or recycle an expired event into an evergreen place card.
    """
    text = source_text(result)
    if not (_EVENT_RE.search(result.title) or _EVENT_WINDOW_RE.search(text)):
        return None
    window = re.search(r'(?:活动时间|展期|举办时间|演出时间|市集时间|日期|时间)[：:]\s*(?:\*\*)?([^\n。；;]{1,100})', text)
    venue = VENUE_RE.search(text)
    if not window or not venue or not concrete_place_name(venue.group(1).strip().rstrip('* ')):
        return {}
    dates = list(_DATE_RE.finditer(window.group(1)))
    if not dates or not dates[0]['year'] or len(dates) > 2:
        return {}
    first, last = dates[0], dates[-1]
    # "6日-8日" has no second month token. Require the full end date rather
    # than silently interpreting this range as a one-day event.
    if len(dates) == 1 and re.search(r'[-—~～至到]\s*\d{1,2}(?:日|号)', window.group(1)[first.end():]):
        return {}
    try:
        start = datetime(int(first['year']), int(first['month']), int(first['day']), tzinfo=_TIME_ZONE)
        end = datetime(int(last['year'] or first['year']), int(last['month']), int(last['day']), tzinfo=_TIME_ZONE)
        end += timedelta(days=1, seconds=-1)
        clocks = re.findall(r'(?<!\d)(\d{1,2}):(\d{2})(?!\d)', window.group(1))
        if clocks:
            if len(clocks) != 2:
                return {}
            start = start.replace(hour=int(clocks[0][0]), minute=int(clocks[0][1]))
            end = end.replace(hour=int(clocks[1][0]), minute=int(clocks[1][1]), second=0)
    except ValueError:
        return {}
    current = now or datetime.now(UTC)
    if end <= current or end < start or start > current + timedelta(days=30):
        return {}
    return dict(location_name=venue.group(1).strip().rstrip('* '), starts_at=start.isoformat(), ends_at=end.isoformat())


def article_place_names(result: SearchResult) -> list[str]:
    """Only explicit names/section headings become independent verification leads."""
    text = source_text(result)
    names = [m.group(1).strip() for m in VENUE_RE.finditer(text)]
    names += [m.group(1).strip() for m in re.finditer(
        r'(?:^|\n)\s*(?:#{1,4}\s+|\d{1,2}[.、]\s*|【)([^\n】]{2,60})', text)]
    return list(dict.fromkeys(name for name in names if concrete_place_name(name)))[:8]
