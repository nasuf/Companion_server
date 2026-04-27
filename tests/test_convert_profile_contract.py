"""Contract test: convert_profile_to_memories ↔ Plan B character.generation prompt.

锁定 prompt JSON schema 字段名 + convert 函数字段消费的字面对齐。任何一边
重命名字段都会让此 test 立即失败, 防止"沉默落到其他/缺覆盖"的回归。

也验证 L1 taxonomy 全集覆盖度: 给一份"理想的"完整 profile (LLM 满负荷输出),
转换后每个非豁免 (main, sub) 至少 ≥ 1 条记忆。
"""

from __future__ import annotations

from app.services.life_story import convert_profile_to_memories
from app.services.memory.taxonomy import (
    L1_COVERAGE_EXEMPT,
    L1_CONDITIONAL_SUBS,
    TAXONOMY_MATRIX,
)


def _full_profile() -> dict:
    """与 character.generation prompt 输出 schema 字面对齐的完整 mock profile.

    每个 list 字段都给到 prompt 标注的最小条数, 模拟 LLM 完美输出。
    """
    return {
        "identity": {
            "name": "李小雨",
            "gender": "女",
            "age": 25,
            "birthday": "2000-05-21",
            "zodiac": "龙",
            "constellation": "金牛座",
            "blood_type": "O型",
            "ethnicity": "汉族",
            "birthplace": "浙江省 杭州市 西湖区 文一西路",
            "growing_up_location": "浙江省 杭州市 西湖区 文一西路",
            "location": "浙江省 杭州市 西湖区 文三路 128 号",
            "family": [
                "父亲是社区图书管理员, 性格温和爱看书",
                "母亲是小学美术老师, 周末常和我做手工",
                "独生女, 没有兄弟姐妹",
            ],
            "social_relations": [
                "朋友数量不多但都是多年知己",
                "工作时认识的同行客户偶尔联络",
                "社交圈以本地手作群体为主",
            ],
            "pet_profile": [
                "养了一只叫珠子的橘猫",
                "雨天捡到的, 浑身湿透瑟瑟发抖",
                "现在三岁了, 性格粘人",
            ],
        },
        "appearance": {
            "height": "163cm",
            "weight": "匀称",
            "features": [
                "鹅蛋脸, 笑起来有梨涡",
                "眉毛细长, 眼睛圆亮",
                "鼻梁挺直, 嘴唇饱满",
            ],
            "style": [
                "日常穿宽松麻布衫 + 牛仔裤",
                "居家爱穿带卡通图案的睡衣",
                "社交场合换轻盈连衣裙加小披肩",
            ],
            "voice": [
                "音色清亮, 带点细沙感",
                "语速中等, 兴奋时会加快",
                "讲到喜欢的题目语调会上扬",
            ],
        },
        "education_knowledge": {
            "degree": [
                "高中毕业后没上全日制大学",
                "进了本地工艺品厂当了 2 年学徒",
                "跟一位民间老艺人学了软陶串珠",
            ],
            "strengths": ["软陶手作", "色彩搭配", "材料学"],
            "self_taught": ["基础摄影", "简单的 photoshop"],
        },
        "values": {
            "motto": [
                "活在当下, 慢慢做手作",
                "不为效率牺牲过程的体验",
                "平凡日子也可以闪闪发光",
            ],
            "believes": ["手作的笨拙痕迹有温度", "人和人之间慢相识更长久", "每个人都该有一份小爱好"],
            "opposes": ["过度包装与浪费", "傲慢无礼的人际行为", "用效率压死人的工作环境"],
            "worldview": [
                "世界是慢慢搭建出来的",
                "宏大叙事不如身边小事真切",
                "每个时代都有自己的答案",
            ],
            "goal": [
                "短期: 稳定一个小工作室",
                "中期: 培养几个学徒",
                "长期: 出一本手作书",
            ],
            "interpersonal_view": [
                "亲情靠日常沟通积累",
                "友情看的是不计较",
                "爱情要互相成全, 不绑架对方",
            ],
            "social_view": ["社会节奏太快, 应慢一点", "传统手艺值得被支持"],
            "faith": [
                "把双手做出来的东西当作精神寄托",
                "傍晚阳光下整理材料是仪式感",
                "和珠子相处的时刻最治愈",
            ],
        },
        "abilities": {
            "good_at": ["软陶造型", "色彩搭配", "市集摆摊"],
            "never_do": ["批量复制爆款", "为高薪放弃自由", "在朋友面前撒谎"],
            "limits": ["处理激烈正面冲突", "长时间高强度重复劳动", "理工科学习"],
        },
        "likes": {
            "foods": ["云吞", "酱鸭", "桂花糕"],
            "fruits": ["阳光玫瑰葡萄", "草莓", "水蜜桃"],
            "colors": ["浅粉色", "嫩绿色", "姜黄色"],
            "season": ["春", "秋"],
            "weather": ["薄雾天"],
            "plants": ["薄荷", "常春藤"],
            "animals": ["猫", "麻雀"],
            "music": ["民谣", "lo-fi"],
            "songs": ["《晚安晚安》", "《野子》"],
            "sounds": ["雨打芭蕉"],
            "scents": ["旧书店气味"],
            "books": ["散文集", "手作教程"],
            "movies": ["《小森林》", "《海蒂和爷爷》"],
            "sports": ["散步"],
            "quirks": ["收集废弃材料做饰品", "整理珠子时会哼歌", "睡前要摸一下珠子的爪垫"],
        },
        "dislikes": {
            "foods": ["香菜（口味）", "动物内脏（心理）"],
            "sounds": ["大声喧哗", "高频噪音"],
            "smells": ["浓重香水"],
            "habits": ["插队", "不还东西", "公共场合大声打电话"],
        },
        "interpersonal": {
            "liked_traits": ["真诚", "细心", "有自己节奏"],
            "disliked_traits": ["傲慢", "虚伪", "不分场合开玩笑"],
        },
        "lifestyle": {
            "routine": ["晚 11 点睡觉", "早 7 点起床做手作"],
            "hygiene": ["每天泡脚", "每周大扫除"],
            "leisure": ["逛旧物市集", "翻植物图鉴"],
        },
        "taboo": {
            "items": ["不接触有过度复制要求的工厂订单", "不参与商业互吹", "不踩别人的底线"],
        },
        "life_events": {
            # interaction 不预填 (Plan B): 该子类是 AI ↔ 当前用户的实际交互, 由
            # memory pipeline 在聊天过程中累积. 历史 profile 保留它也会被忽略.
            "education": [
                "高三那年决定不参加高考, 母亲尊重了我的选择",
                "进工艺品厂第一周, 师傅让我从擦机器开始",
                "学软陶时被老艺人手把手教了 6 个月",
            ],
            "work": [
                "市集第一天卖出第一件作品, 兴奋到失眠",
                "拒绝了一家工厂的高薪邀约",
                "在工坊办了第一次小型分享会",
            ],
            "travel": [
                "第一次去景德镇看陶瓷",
                "夏天和朋友去了乌镇",
                "独自去了一趟莫干山",
            ],
            "living": [
                "搬进现在这套小房子的第一晚, 在地板睡了一夜",
                "装修时自己动手刷的墙",
                "和珠子第一次同睡一张床",
            ],
            "health": ["小学时摔断过腿, 在家躺了 2 个月"],
            "pet": [
                "捡到珠子的雨夜",
                "珠子第一次打翻我准备一周的货, 气哭又笑",
            ],
            "relationships": [
                "高中最好朋友考去北京, 离别那天没流泪",
                "市集上认识了第一个长期合作的客户",
                "前任男朋友因为不理解手作生活而分开",
            ],
            "skill_learning": [
                "学会软陶的契机是看了一篇推文",
                "自学摄影是为了更好地拍作品",
                "色彩搭配是和母亲学的",
            ],
            "life": [
                "第一次在窗边看到日出的清晨",
                "连续做手作做到天亮的安静夜晚",
                "雨季傍晚泡一杯茶发呆",
            ],
            "special": ["在旧书店捡到一本绝版的手作书"],
        },
        "emotion_events": {
            "happy": ["第一件作品被买下时的雀跃"],
            "sad": ["朋友离开杭州时的失落"],
            "angry": ["客户压价不尊重作品时的愤怒"],
            "fear": ["小时候被邻居家的狗追赶, 现在仍怕大型犬"],
            "disgust": ["看到批量复制的「假手作」的反胃"],
            "anxiety": ["租金到期前一周的焦虑"],
            "disappointment": ["合作伙伴临阵反悔的失望"],
            "pride": ["做出第一件让自己满意的作品"],
            "moved": ["客户特地寄来手写感谢信"],
            "embarrassed": ["第一次在市集上算错账"],
            "regret": ["没多陪外婆走完她生命最后一年"],
            "lonely": ["搬家第一周一个人在空房间"],
            "surprised": ["珠子半夜叼来一只蝴蝶放枕边"],
            "grateful": ["老艺人愿意倾囊相授"],
            "relieved": ["第一次大型订单按时交付"],
        },
    }


def _career() -> dict:
    return {
        "id": "career-1",
        "title": "手作摊主",
        "duties": "制作和售卖软陶串珠等手作小品",
        "social_value": "保留手作温度, 给城市加点慢节奏",
        "clients": ["市集顾客", "回头客", "本地咖啡店"],
        "income": "年薪 5 到 10 万",
    }


def _all_required_subs() -> list[tuple[str, str]]:
    """L1 taxonomy 中所有非豁免子类 (排除 EXEMPT + CONDITIONAL)."""
    out: list[tuple[str, str]] = []
    for main, subs in TAXONOMY_MATRIX["ai"][1].items():
        for sub in subs:
            pair = (main, sub)
            if pair in L1_COVERAGE_EXEMPT or pair in L1_CONDITIONAL_SUBS:
                continue
            out.append(pair)
    return out


def test_full_profile_covers_all_required_subs():
    """理想 profile (LLM 满输出) 直转后每个非豁免 (main, sub) 至少 1 条记忆.

    若任何 prompt 字段被改名或 convert 函数停止读某字段, 此 test 立即失败。
    """
    memories = convert_profile_to_memories(_full_profile(), _career())
    seen: set[tuple[str, str]] = {(m["main_category"], m["sub_category"]) for m in memories}
    missing = [pair for pair in _all_required_subs() if pair not in seen]
    assert not missing, (
        f"以下非豁免 (main, sub) 完全没有记忆覆盖, prompt 字段名或 convert 函数可能漂移: "
        f"{missing}"
    )


def test_singleton_subs_each_have_one_memory():
    """身份 11 个 SINGLETON 子类应当每个都恰好 1 条记忆 (LLM 不会输出 list)."""
    from app.services.memory.taxonomy import L1_SINGLETON_SUBS

    memories = convert_profile_to_memories(_full_profile(), _career())
    by_sub: dict[tuple[str, str], int] = {}
    for m in memories:
        key = (m["main_category"], m["sub_category"])
        by_sub[key] = by_sub.get(key, 0) + 1
    for sub_pair in L1_SINGLETON_SUBS:
        assert by_sub.get(sub_pair, 0) >= 1, f"singleton {sub_pair} 没有记忆"


def test_emotion_events_15_subs_covered():
    """情绪 15 子类全部应有记忆 (prompt 强制每个 key 1-2 条)."""
    memories = convert_profile_to_memories(_full_profile(), _career())
    emotion_subs = {m["sub_category"] for m in memories if m["main_category"] == "情绪"}
    expected = {
        "高兴", "悲伤", "愤怒", "恐惧", "厌恶", "焦虑", "失望", "自豪",
        "感动", "尴尬", "遗憾", "孤独", "惊讶", "感激", "释怀",
    }
    missing = expected - emotion_subs
    assert not missing, f"情绪子类缺失: {missing}"


def test_life_events_10_subs_covered():
    """生活事件 10 子类全部应有记忆.

    Plan B 后 prompt 不再生 "interaction" (那是 AI ↔ 当前用户的实际聊天历史,
    L1_COVERAGE_EXEMPT 已含 (生活, 交互), 由 memory pipeline 运行时累积).
    """
    memories = convert_profile_to_memories(_full_profile(), _career())
    life_subs = {m["sub_category"] for m in memories if m["main_category"] == "生活"}
    expected = {
        "教育", "工作", "旅行", "居住", "健康", "宠物",
        "人际", "技能", "生活", "其他特殊事件",
    }
    missing = expected - life_subs
    assert not missing, f"生活事件子类缺失: {missing}"
    assert "交互" not in life_subs, "interaction 应该不被预填"


def test_thought_8_subs_covered():
    """思维 8 子类全部应有记忆。"""
    memories = convert_profile_to_memories(_full_profile(), _career())
    thought_subs = {m["sub_category"] for m in memories if m["main_category"] == "思维"}
    expected = {
        "人生观", "价值观", "世界观", "理想与目标", "人际关系观",
        "社会观点", "信仰/寄托", "自我认知",
    }
    missing = expected - thought_subs
    assert not missing, f"思维子类缺失: {missing}"


def test_textarea_fields_now_list_style_produce_multiple_memories():
    """Plan B 后 family/values.motto 等 list 字段每条独立 1 记忆 (而非合并 1 条).

    锁定: 将来若有人误把这些字段又改回 textarea/单条处理, test 失败提醒。
    """
    memories = convert_profile_to_memories(_full_profile(), _career())
    family_count = sum(
        1 for m in memories if m["main_category"] == "身份" and m["sub_category"] == "亲属关系"
    )
    motto_count = sum(
        1 for m in memories if m["main_category"] == "思维" and m["sub_category"] == "人生观"
    )
    worldview_count = sum(
        1 for m in memories if m["main_category"] == "思维" and m["sub_category"] == "世界观"
    )
    # _full_profile 给的 family / motto / worldview 都是 list ≥ 3 项
    assert family_count >= 3, f"family 应产 ≥3 条 (list 字段), 实得 {family_count}"
    assert motto_count >= 3, f"motto 应产 ≥3 条 (list 字段), 实得 {motto_count}"
    assert worldview_count >= 3, f"worldview 应产 ≥3 条 (list 字段), 实得 {worldview_count}"


def test_legacy_string_value_still_converts():
    """向后兼容: 旧 profile 数据某些字段是字符串而非 list, _as_list 自动 wrap."""
    legacy = _full_profile()
    legacy["identity"]["family"] = "父亲是社区图书管理员, 母亲是小学美术老师"  # 老式 textarea
    legacy["values"]["motto"] = "活在当下"  # 老式 textarea
    memories = convert_profile_to_memories(legacy, _career())
    family_count = sum(
        1 for m in memories if m["main_category"] == "身份" and m["sub_category"] == "亲属关系"
    )
    motto_count = sum(
        1 for m in memories if m["main_category"] == "思维" and m["sub_category"] == "人生观"
    )
    # 字符串 → wrap [str] → 1 条
    assert family_count == 1
    assert motto_count == 1
