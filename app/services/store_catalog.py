"""Canonical mall catalog: exchange SKUs plus pack (bundle) offers.

Prices come from 商城-兑换清单.xlsx / 商城-礼包清单.xlsx.
Member price is charged when the wallet is currently VIP; otherwise list price.
Flutter `store_data.dart` must stay in lockstep with this module.
"""

from __future__ import annotations

from dataclasses import dataclass

GIFT_SUBCATEGORIES: tuple[str, ...] = (
    "奢享",
    "数码",
    "生活",
    "美食",
    "配饰",
    "饮品",
    "饰品",
    "鲜花",
)

MUSIC_COUPON_KIND = "music_hour_coupon"
MUSIC_COUPON_VALID_DAYS = 30


@dataclass(frozen=True)
class ExchangeProduct:
    product_kind: str
    title: str
    member_price: int
    list_price: int
    category: str  # gift | blind | outfit
    subcategory: str | None = None
    contents: str | None = None
    asset_key: str | None = None

    def price_for(self, is_vip: bool) -> int:
        return self.member_price if is_vip else self.list_price

    def to_public_dict(self, *, is_vip: bool) -> dict:
        return {
            "product_kind": self.product_kind,
            "title": self.title,
            "member_price": self.member_price,
            "list_price": self.list_price,
            "price": self.price_for(is_vip),
            "category": self.category,
            "subcategory": self.subcategory,
            "contents": self.contents,
        }


@dataclass(frozen=True)
class BundleTier:
    tier_id: str
    label: str
    ticket_price: int
    grant_amount: int


MUSIC_BUNDLE_TIERS: tuple[BundleTier, ...] = (
    BundleTier("1", "1张", 10, 1),
    BundleTier("5", "5张", 45, 5),
    BundleTier("10", "10张", 80, 10),
)

GAME_BUNDLE_TIERS: tuple[BundleTier, ...] = (
    BundleTier("100", "100点", 20, 100),
    BundleTier("200", "200点", 35, 200),
    BundleTier("500", "500点", 80, 500),
)

VIP_TRIAL_YUAN = 1
VIP_TRIAL_DAYS = 30

_PRODUCTS: tuple[ExchangeProduct, ...] = (
    ExchangeProduct("gift_1", "美式咖啡", 18, 25, "gift", "饮品", asset_key="1"),
    ExchangeProduct("gift_2", "单支玫瑰", 18, 25, "gift", "鲜花", asset_key="2"),
    ExchangeProduct("gift_3", "拿铁咖啡", 28, 40, "gift", "饮品", asset_key="3"),
    ExchangeProduct("gift_4", "卡布奇诺", 38, 55, "gift", "饮品", asset_key="4"),
    ExchangeProduct("gift_5", "毛绒小熊", 58, 85, "gift", "生活", asset_key="5"),
    ExchangeProduct("gift_6", "抹茶拿铁", 68, 95, "gift", "饮品", asset_key="6"),
    ExchangeProduct("gift_7", "肥宅快乐水", 68, 95, "gift", "饮品", asset_key="7"),
    ExchangeProduct("gift_8", "冰红茶", 78, 110, "gift", "饮品", asset_key="8"),
    ExchangeProduct("gift_9", "蜜桃乌龙茶", 88, 125, "gift", "饮品", asset_key="9"),
    ExchangeProduct("gift_10", "银质项链", 88, 125, "gift", "饰品", asset_key="10"),
    ExchangeProduct("gift_11", "玩偶挂件", 108, 155, "gift", "生活", asset_key="11"),
    ExchangeProduct("gift_12", "柠檬冰茶", 118, 170, "gift", "饮品", asset_key="12"),
    ExchangeProduct("gift_13", "布朗尼蛋糕", 128, 185, "gift", "美食", asset_key="13"),
    ExchangeProduct("gift_14", "巧克力曲奇", 138, 195, "gift", "美食", asset_key="14"),
    ExchangeProduct("gift_15", "芝士蛋糕", 148, 210, "gift", "美食", asset_key="15"),
    ExchangeProduct("gift_16", "特调鸡尾酒", 158, 225, "gift", "饮品", asset_key="16"),
    ExchangeProduct("gift_17", "精酿啤酒", 168, 240, "gift", "饮品", asset_key="17"),
    ExchangeProduct("gift_18", "提拉米苏", 168, 240, "gift", "美食", asset_key="18"),
    ExchangeProduct("gift_19", "奶油蛋糕", 178, 255, "gift", "美食", asset_key="19"),
    ExchangeProduct("gift_20", "银质手链", 178, 255, "gift", "饰品", asset_key="20"),
    ExchangeProduct("gift_21", "威士忌", 188, 270, "gift", "饮品", asset_key="21"),
    ExchangeProduct("gift_22", "马卡龙", 188, 270, "gift", "美食", asset_key="22"),
    ExchangeProduct("gift_23", "冰淇淋", 218, 310, "gift", "美食", asset_key="23"),
    ExchangeProduct("gift_24", "香薰蜡烛", 228, 325, "gift", "生活", asset_key="24"),
    ExchangeProduct("gift_25", "花香膏", 248, 355, "gift", "生活", asset_key="25"),
    ExchangeProduct("gift_26", "墨镜", 288, 410, "gift", "配饰", asset_key="26"),
    ExchangeProduct("gift_27", "八音盒", 288, 410, "gift", "生活", asset_key="27"),
    ExchangeProduct("gift_28", "玫瑰花束", 288, 410, "gift", "鲜花", asset_key="28"),
    ExchangeProduct("gift_29", "手摇咖啡机", 288, 410, "gift", "生活", asset_key="29"),
    ExchangeProduct("gift_30", "多肉盆栽", 308, 440, "gift", "饰品", asset_key="30"),
    ExchangeProduct("gift_31", "水晶手串", 328, 470, "gift", "饰品", asset_key="31"),
    ExchangeProduct("gift_32", "头戴耳机", 328, 470, "gift", "数码", asset_key="32"),
    ExchangeProduct("gift_33", "康乃馨花束", 328, 470, "gift", "鲜花", asset_key="33"),
    ExchangeProduct("gift_34", "针织围巾", 388, 555, "gift", "配饰", asset_key="34"),
    ExchangeProduct("gift_35", "郁金香花束", 388, 555, "gift", "鲜花", asset_key="35"),
    ExchangeProduct("gift_36", "宝石耳钉", 418, 595, "gift", "饰品", asset_key="36"),
    ExchangeProduct("gift_37", "满天星花束", 428, 610, "gift", "鲜花", asset_key="37"),
    ExchangeProduct("gift_38", "银质戒指", 468, 670, "gift", "饰品", asset_key="38"),
    ExchangeProduct("gift_39", "经典香水", 488, 695, "gift", "生活", asset_key="39"),
    ExchangeProduct("gift_40", "水晶吊坠", 488, 695, "gift", "饰品", asset_key="40"),
    ExchangeProduct("gift_41", "帆布包", 488, 695, "gift", "配饰", asset_key="41"),
    ExchangeProduct("gift_42", "珍珠耳钉", 548, 785, "gift", "饰品", asset_key="42"),
    ExchangeProduct("gift_43", "爱心手链", 588, 840, "gift", "饰品", asset_key="43"),
    ExchangeProduct("gift_44", "蓝牙音箱", 588, 840, "gift", "数码", asset_key="44"),
    ExchangeProduct("gift_45", "小香风手提包", 688, 985, "gift", "配饰", asset_key="45"),
    ExchangeProduct("gift_46", "复古双肩包", 688, 985, "gift", "配饰", asset_key="46"),
    ExchangeProduct("gift_47", "滑板", 688, 985, "gift", "生活", asset_key="47"),
    ExchangeProduct("gift_48", "鼠标", 688, 985, "gift", "数码", asset_key="48"),
    ExchangeProduct("gift_49", "球鞋", 788, 1125, "gift", "配饰", asset_key="49"),
    ExchangeProduct("gift_50", "蓝牙耳机", 888, 1270, "gift", "数码", asset_key="50"),
    ExchangeProduct("gift_51", "毛绒大玩偶", 1588, 2270, "gift", "生活", asset_key="51"),
    ExchangeProduct("gift_52", "智能手环", 1888, 2700, "gift", "数码", asset_key="52"),
    ExchangeProduct("gift_53", "精致下午茶", 1888, 2700, "gift", "美食", asset_key="53"),
    ExchangeProduct("gift_54", "豪华大餐一份", 2888, 4200, "gift", "美食", asset_key="54"),
    ExchangeProduct("gift_55", "游戏机", 3888, 5555, "gift", "数码", asset_key="55"),
    ExchangeProduct("gift_56", "钻石戒指", 4888, 7000, "gift", "饰品", asset_key="56"),
    ExchangeProduct("gift_57", "钻石项链", 5888, 8500, "gift", "饰品", asset_key="57"),
    ExchangeProduct("gift_58", "电脑", 6888, 9900, "gift", "数码", asset_key="58"),
    ExchangeProduct("gift_60", "定制礼服", 6888, 9900, "gift", "配饰", asset_key="60"),
    ExchangeProduct("gift_61", "单反相机", 7888, 12000, "gift", "数码", asset_key="61"),
    ExchangeProduct("gift_62", "名牌包包", 7888, 12000, "gift", "配饰", asset_key="62"),
    ExchangeProduct("gift_59", "9999朵玫瑰", 8888, 13000, "gift", "鲜花", asset_key="59"),
    ExchangeProduct("gift_63", "机械腕表", 9888, 15000, "gift", "饰品", asset_key="63"),
    ExchangeProduct("gift_64", "定制烟火秀", 18888, 28000, "gift", "奢享", asset_key="64"),
    ExchangeProduct("gift_65", "花束海洋", 28888, 42000, "gift", "奢享", asset_key="65"),
    ExchangeProduct("gift_66", "全城大屏联动", 38888, 55555, "gift", "奢享", asset_key="66"),
    ExchangeProduct("gift_67", "漫天花瓣雨", 48888, 70000, "gift", "奢享", asset_key="67"),
    ExchangeProduct("gift_68", "无人机灯光秀", 68888, 100000, "gift", "奢享", asset_key="68"),
    ExchangeProduct("gift_69", "星系命名", 99999, 150000, "gift", "奢享", asset_key="69"),
    ExchangeProduct("gift_70", "环球奢华游", 666666, 1000000, "gift", "奢享", asset_key="70"),
    ExchangeProduct("blind_milk_tea", "奶茶盲盒", 18, 25, "blind", contents="珍珠奶茶一杯、波霸奶茶一杯、杨枝甘露一杯、芋泥波波一杯、蜜桃乌龙一杯、桂花乌龙一杯、茉香奶绿一杯、伯爵鲜奶一杯、黑糖珍珠鲜奶一杯、椰奶芋圆一杯、抹茶红豆一杯、焦糖海盐一杯、玫瑰荔枝一杯、蓝莓酸奶一杯、芒果椰奶一杯、草莓脏脏茶一杯、青柠薄荷一杯、百香果双响炮一杯、芝士葡萄一杯、蜜瓜椰奶一杯", asset_key="milk_tea"),
    ExchangeProduct("blind_cola", "可乐盲盒", 28, 40, "blind", contents="经典可乐一杯、樱桃可乐一杯、香草可乐一杯、青柠可乐一杯、姜汁可乐一杯、肉桂可乐一杯、咖啡可乐一杯、草莓可乐一杯、椰子可乐一杯、蜂蜜可乐一杯", asset_key="cola"),
    ExchangeProduct("blind_coffee", "咖啡盲盒", 38, 55, "blind", contents="意式浓缩一杯、美式咖啡一杯、拿铁一杯、卡布奇诺一杯、摩卡一杯、冷萃咖啡一杯、澳白一杯、手冲瑰夏一杯、手冲耶加雪菲一杯、手冲曼特宁一杯、抹茶拿铁一杯、焦糖玛奇朵一杯、海盐拿铁一杯、燕麦拿铁一杯、桂花拿铁一杯、榛果拿铁一杯、香草拿铁一杯、南瓜拿铁一杯、椰青美式一杯、爱尔兰咖啡一杯", asset_key="coffee"),
    ExchangeProduct("blind_alcohol", "酒精盲盒", 48, 70, "blind", contents="莫吉托一杯、金汤力一杯、威士忌酸一杯、长岛冰茶一杯、桑格利亚一杯、热红酒一杯、玛格丽特一杯、白俄罗斯一杯、迈泰一杯、贝里尼一杯、浑浊IPA一杯、世涛一杯、比利时三料一杯、水果酸啤一杯、小麦白啤一杯、英式波特一杯、琥珀艾尔一杯、法兰德斯红一杯、赛松一杯、皮尔森一杯", asset_key="alcohol"),
    ExchangeProduct("blind_dessert", "甜点盲盒", 108, 155, "blind", contents="提拉米苏一份、抹茶千层一份、红丝绒蛋糕一份、蓝莓芝士一份、芒果慕斯一份、黑森林蛋糕一份、蒙布朗栗子一份、草莓奶油蛋糕一份、巧克力熔岩一份、椰子冻一份、焦糖布丁一份、半熟芝士一份、可露丽一份、费南雪一份、玛德琳一份、马卡龙一份、蝴蝶酥一份、千层酥一份、闪电泡芙一份、歌剧院蛋糕一份", asset_key="dessert"),
    ExchangeProduct("blind_flower", "鲜花盲盒", 228, 330, "blind", contents="红玫瑰花束一束、粉玫瑰花束一束、白百合花束一束、郁金香花束一束、向日葵花束一束、蓝色绣球花束一束、紫色桔梗花束一束、洋甘菊花束一束、薰衣草花束一束、蝴蝶兰花束一束、白色雏菊花束一束、粉色康乃馨花束一束、香槟玫瑰花束一束、尤加利叶花束一束、雪柳一束、铃兰花束一束、小苍兰花束一束、鸢尾花花束一束、银莲花花束一束、帝王花花束一束", asset_key="flower"),
    ExchangeProduct("blind_food", "美食盲盒", 288, 420, "blind", contents="安格斯牛排券一张、和牛汉堡券一张、秘制炸鸡券一张、麻辣火锅券一张、寿司拼盘券一张、意大利面券一张、披萨券一张、烤羊排券一张、海鲜大餐券一张、泰式冬阴功券一张、日式拉面券一张、越南河粉券一张、西班牙海鲜饭券一张、法式焗蜗牛券一张、烤鸭券一张、钵钵鸡券一张、小龙虾券一张、烤肉拼盘券一张、生蚝刺身券一张、佛跳墙券一张", asset_key="food"),
    ExchangeProduct("outfit_capsule", "胶囊皮肤", 188, 265, "outfit", asset_key="capsule"),
    ExchangeProduct("outfit_chat_frame", "聊天框皮肤", 288, 405, "outfit", asset_key="chat_frame"),
    ExchangeProduct("outfit_bubble", "聊天气泡", 388, 550, "outfit", asset_key="bubble"),
    ExchangeProduct("outfit_backdrop", "聊天背景", 588, 825, "outfit", asset_key="backdrop"),
    ExchangeProduct("outfit_theme", "主题皮肤", 688, 960, "outfit", asset_key="theme"),
    ExchangeProduct("outfit_stationery", "信纸皮肤", 888, 1240, "outfit", asset_key="stationery"),
    ExchangeProduct("outfit_checkin", "打卡页面皮肤", 1888, 2640, "outfit", asset_key="checkin"),
)

EXCHANGE_PRODUCTS: dict[str, ExchangeProduct] = {
    product.product_kind: product for product in _PRODUCTS
}


def music_tier(tier_id: str) -> BundleTier | None:
    for tier in MUSIC_BUNDLE_TIERS:
        if tier.tier_id == str(tier_id):
            return tier
    return None


def game_tier(tier_id: str) -> BundleTier | None:
    for tier in GAME_BUNDLE_TIERS:
        if tier.tier_id == str(tier_id):
            return tier
    return None


def catalog_payload(*, is_vip: bool, vip_trial_available: bool) -> dict:
    return {
        "is_vip": is_vip,
        "vip_trial_available": vip_trial_available,
        "products": [item.to_public_dict(is_vip=is_vip) for item in _PRODUCTS],
        "bundles": {
            "music": {
                "kind": "music_coupon",
                "title": "音乐畅听券",
                "currency": "ticket",
                "tiers": [
                    {
                        "tier_id": t.tier_id,
                        "label": t.label,
                        "ticket_price": t.ticket_price,
                        "grant_amount": t.grant_amount,
                    }
                    for t in MUSIC_BUNDLE_TIERS
                ],
            },
            "game": {
                "kind": "game_points",
                "title": "游戏积分券",
                "currency": "ticket",
                "tiers": [
                    {
                        "tier_id": t.tier_id,
                        "label": t.label,
                        "ticket_price": t.ticket_price,
                        "grant_amount": t.grant_amount,
                    }
                    for t in GAME_BUNDLE_TIERS
                ],
            },
            "vip_trial": {
                "kind": "vip_trial",
                "title": "月度 VIP 体验",
                "currency": "cny",
                "yuan_price": VIP_TRIAL_YUAN,
                "available": vip_trial_available,
            },
        },
    }
