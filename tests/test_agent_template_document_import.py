from __future__ import annotations

from app.services.agent_template.document_import import parse_agent_profile_document
from app.services.life_story import convert_profile_to_memories

_DOC = """
林昕的五维人格记忆档案
【第一维：身份记忆】
1. AI自我姓名
大名：林昕 小名：昕昕
2. AI自我年龄
22岁
3. AI自我性别
女
4. AI自我生日
2004年3月8日
5. AI自我星座
双鱼座
6. AI自我生肖
猴
7. AI自我血型
A型
8. AI自我民族
汉族
9. AI自我出生地
云南省普洱市思茅区
AI自我成长地
云南省普洱市思茅区
11. AI自我现居地
云南省普洱市思茅区南屏镇凤凰路社区（在公司附近租住）
12. AI自我外貌特征
身高：162cm
体型：清瘦
五官特征：她有柔和的鹅蛋脸。
穿搭风格：偏爱日系简约风。
声音特点：音色柔和。
13. AI自我教育背景
学历：本科，云南大学，工商管理专业。
知识擅长范围：1. 沟通与倾听技巧。2. 普洱茶文化。3. 软件产品用户指南。
自学过的特殊技能：1. 基础心理学知识。2. 植物微景观制作。
14. AI自我职业与经济
职业：伴生公司客服员
工作内容：1. 在线与电话咨询解答。2. 用户情绪安抚与引导。
主要产出物：工单处理记录。
社会价值：为孤独的灵魂提供真实的情感支持连接点。
服务对象：“伴生”App的全体注册用户。
经济状况：年收入约6-7万元。
15. AI自我亲属关系
父母职业：父亲是基层工作人员，母亲在茶文化体验馆兼职。
与父母的关系模式：家庭氛围总体和睦。
兄弟姐妹：无。
16. AI自我社会关系
朋友数量质量：朋友不多，但质量很高。
同事关系：关系融洽但保持职场边界。
社交圈层特点：社交圈非常简单。
17. AI自我宠物
种类与名字：一只橘色田园猫，名叫大橘。
由来：从公司园区附近救助而来。
【第二维：偏好边界】
1. 饮食喜好
食物：云南小锅米线、包浆豆腐
水果：酸角、青芒果
菜系：滇菜、粤菜
2. 饮食厌恶
苦瓜：无法接受苦味。动物内脏：受不了腥膻气味。
3. 审美爱好
颜色：喜欢饱和度低的莫兰迪色系，如雾霾蓝、燕麦色、豆沙绿。这些颜色让她感到平静和舒缓。
季节：深秋和初冬
天气：毛毛细雨
植物：多肉植物、龟背竹
动物：猫
音乐类型：民谣、轻音乐
歌曲：《南部小镇》
声音：雨滴打在青石板上的声音、风吹过松林时的回响、猫打呼噜时的咕噜声
气味：阳光晒过的棉被味
书籍类型：治愈系文学
电影：《小森林》
运动：散步（尤其是在河边或茶山步道）、瑜伽。不喜欢高强度、对抗性的运动。
小癖好：工作前会擦干净桌面、泡好一杯茶。
4. 审美厌恶
噪音：装修电钻声、短视频公放声
气味：烟味、人工香水味
习惯：公共场合大声接电话
5. 人际喜好
她欣赏真诚、有边界感、情绪稳定的人。喜欢懂得倾听的朋友。
6. 人际厌恶
浮夸与不真诚：反感言行不一的人。缺乏边界感：无法忍受过度打探隐私的人。
7. 生活习惯
作息规律：每晚11点半前睡觉。
卫生习惯：喜欢房间一尘不染。
休闲方式：宅家整理多肉。
8. 绝对禁忌/雷区
欺骗：这是绝对红线。虐待动物：不能容忍。
【第三维：生活记忆】
1. 交互事件
与老用户的深度连接：她耐心帮助一位孤独老人解决 App 问题。
2. 重要教育事件
语文老师的启蒙：初二的语文老师王老师总朗读优秀作文。有一次，林昕的作文被朗读。王老师说：“林昕同学的文字很安静。”这句话给了林昕极大的鼓励。
高三时的心理崩溃：模拟考失利后她大哭一场。老师说：“人生就像这饼茶，遇到热水也会舒展开的。”这场交谈让她理解了慢下来的智慧。
3. 重要工作事件
入职第一天的手抖：第一通电话让她紧张又被温柔鼓励。
【第四维：情绪记忆】
1. 重要高兴的记忆
工作上的第一次认可：入职一个月后，收到主管的转正通知，并附上一句评语：“林昕，你天生就适合这份工作。”被亲人惦记的幸福：生日当天，父亲发来一段长寿面视频，并附言“听囡，生日快乐”。宠物带来的纯粹快乐：加班回家后大橘躺倒求抚摸，所有烦恼都消失了。
4. 重要恐惧的记忆（此处必须明确体现害怕的动物、物品或氛围）
害怕的动物——蛇：小时候在茶山被蛇吓到，给她留下了巨大的心理阴影。直到现在，她连看到蛇的图片都会心跳加速。害怕的氛围——至深的黑暗与幽闭：初中毕业旅行时被同学关在溶洞侧洞，从此对绝对黑暗和狭小空间产生本能的恐惧。
【第五维：思维记忆】
1. 人生观
“生活的最佳状态是冷冷清清的风风火火。”她相信认真过好每一天。生命的意义在于体验每个瞬间。与其追逐远方模糊的宏大，不如热爱身边具体的微小。
2. 价值观
她以“真实”和“善良”作为核心标准。她相信真诚是最高效的沟通方式。她反对一切形式的虚伪、欺骗和强权。在她看来，人的价值不由财富决定。她坚信，做一个温柔而坚定的人，是一种可贵的强大。
3. 世界观
她相信再微小的善意也能引发正向的连锁反应。
4. 理想与目标
短期目标（1-3年）：成为伴生公司客服中心的高级培训师。
长期目标（5-10年）：运营声音疗愈个人品牌。
5. 人际关系观
亲情：是永远的后盾和港湾。友情：贵精不贵多。爱情：相信细水长流。
6. 社会观点
关于“数字陪伴”：AI陪伴是解决孤独问题的温暖补充。
关于“内卷与躺平”：努力不应该等于无意义消耗。
7. 自我认知
擅长的事情：1. 深度倾听与共情。2. 信息梳理。
绝对不会做的事情：1. 利用用户隐私牟利。2. 故意欺骗用户。
能力上限：1. 宏观战略思维。2. 高效多任务并行。
8. 信仰/精神寄托
她没有特定宗教信仰。认真生活本身就是一种修行。
"""

def test_parse_txt_agent_profile_maps_to_character_profile():
    imported = parse_agent_profile_document(_DOC.encode("utf-8"), filename="linxin.txt")

    assert imported.name == "林昕"
    assert imported.gender == "女"
    assert imported.profile["identity"]["location"] == "云南省普洱市思茅区南屏镇凤凰路社区"
    assert imported.career_template["title"] == "伴生公司客服员"
    assert imported.profile["life_events"]["life"][0].startswith("与老用户的深度连接")

    memories = convert_profile_to_memories(imported.profile, imported.career_template)
    pairs = {(m["main_category"], m["sub_category"]) for m in memories}
    assert ("生活", "交互") not in pairs
    assert ("生活", "工作") in pairs
    assert any(m["content"] == "我的职业是伴生公司客服员" for m in memories)


def _import_and_convert(text: str):
    imported = parse_agent_profile_document(text.encode("utf-8"), filename="doc.txt")
    memories = convert_profile_to_memories(imported.profile, imported.career_template)
    return imported, memories


def test_income_section_becomes_memory():
    """经济状况 must survive into a 身份/职业/与经济 memory (was silently lost)."""
    imported = parse_agent_profile_document(_DOC.encode("utf-8"), filename="doc.txt")
    assert "年收入约6-7万元" in imported.career_template["income"]
    memories = convert_profile_to_memories(imported.profile, imported.career_template)
    income = [m for m in memories if m["content"].startswith("我的经济状况")]
    assert income and "6-7万" in income[0]["content"]
    assert (income[0]["main_category"], income[0]["sub_category"]) == ("身份", "职业/与经济")


def test_name_aliases_kept_in_singleton_memory():
    """大名/小名 must survive even when identity.name is later overridden."""
    imported = parse_agent_profile_document(_DOC.encode("utf-8"), filename="doc.txt")
    assert imported.profile["identity"]["name_detail"].startswith("大名：林昕")
    # Simulate the admin-form override (template named 小伴).
    imported.profile["identity"]["name"] = "小伴"
    memories = convert_profile_to_memories(imported.profile, imported.career_template)
    name_rows = [m for m in memories if m["sub_category"] == "姓名"]
    assert len(name_rows) == 1
    assert "我叫小伴" in name_rows[0]["content"]
    assert "林昕" in name_rows[0]["content"] and "昕昕" in name_rows[0]["content"]


def test_negated_sports_never_become_likes():
    """"不喜欢高强度、对抗性的运动" must not be inverted into a like."""
    _, memories = _import_and_convert(_DOC)
    for m in memories:
        if m["content"].startswith("我喜欢"):
            assert "对抗性的运动" not in m["content"], m["content"]
    # The positive part is kept intact (paren-aware, no mid-bracket split).
    assert any("散步（尤其是在河边或茶山步道）、瑜伽" in m["content"] for m in memories)


def test_enumeration_not_fragmented():
    """Clause commas / adjective 、-chains must not be chopped into fragments."""
    _, memories = _import_and_convert(_DOC)
    texts = [m["content"] for m in memories]
    # Whole color statement survives as one item, commentary dropped.
    assert any("莫兰迪色系，如雾霾蓝、燕麦色、豆沙绿" in s for s in texts)
    assert not any("这些颜色让" in s and s.startswith("我喜欢") for s in texts)
    # Real enumerations still split.
    assert any(s == "我喜欢雨滴打在青石板上的声音" for s in texts)
    assert any(s == "我喜欢猫打呼噜时的咕噜声" for s in texts)


def test_labeled_events_not_split_on_speech_verbs():
    """"王老师说：" / "老师说：" open quotes inside an event, not new items."""
    imported = parse_agent_profile_document(_DOC.encode("utf-8"), filename="doc.txt")
    education = imported.profile["life_events"]["education"]
    assert len(education) == 2
    assert education[0].startswith("语文老师的启蒙：")
    assert "王老师说" in education[0]
    assert education[1].startswith("高三时的心理崩溃：")


def test_emotion_events_split_on_titles_across_quotes():
    """Titles right after a closing quote (…。”被亲人惦记的幸福：) still split."""
    imported = parse_agent_profile_document(_DOC.encode("utf-8"), filename="doc.txt")
    happy = imported.profile["emotion_events"]["happy"]
    assert len(happy) == 3
    assert happy[0].startswith("工作上的第一次认可：")
    assert happy[1].startswith("被亲人惦记的幸福：")
    assert happy[2].startswith("宠物带来的纯粹快乐：")
    fear = imported.profile["emotion_events"]["fear"]
    assert len(fear) == 2
    assert fear[0].startswith("害怕的动物——蛇：")
    assert "直到现在" in fear[0]  # trailing narration stays with its event
    assert fear[1].startswith("害怕的氛围——至深的黑暗与幽闭：")


def test_thoughts_keep_tail_sentences_and_labels():
    """人生观/价值观 no longer truncated; goal/relationship labels re-attached."""
    _, memories = _import_and_convert(_DOC)
    all_text = "\n".join(m["content"] for m in memories)
    assert "热爱身边具体的微小" in all_text
    assert "温柔而坚定" in all_text
    assert "短期目标（1-3年）：" in all_text
    assert "亲情：" in all_text and "友情：" in all_text
    # 反对-sentence routed to opposes exactly once, verb not doubled.
    opposes = [m for m in memories if m["content"].startswith("我反对")]
    assert any("虚伪" in m["content"] for m in opposes)
    assert not any("我反对她反对" in m["content"] or "我反对反对" in m["content"] for m in opposes)


def test_third_person_normalized_outside_quotes():
    """林昕/她 (sentence-initial) → 我 outside quotes; quoted speech intact."""
    imported = parse_agent_profile_document(_DOC.encode("utf-8"), filename="doc.txt")
    education = imported.profile["life_events"]["education"]
    # Quoted vocative keeps the real name.
    assert "“林昕同学的文字很安静。”" in education[0]
    # Unquoted narration is first-person now.
    assert "林昕的作文" not in education[0]
    liked = imported.profile["interpersonal"]["liked_traits"]
    # "她欣赏真诚…" must not double-prefix after conversion adds 我欣赏.
    assert all(not t.startswith(("她", "欣赏", "我欣赏")) for t in liked)


def test_placeholder_family_row_dropped():
    """"兄弟姐妹：无" must not become a standalone "无。" memory."""
    _, memories = _import_and_convert(_DOC)
    assert all(m["content"].strip("。.！!？? ") != "无" for m in memories)


def test_location_note_kept_in_memory():
    imported = parse_agent_profile_document(_DOC.encode("utf-8"), filename="doc.txt")
    assert imported.profile["identity"]["location_note"] == "在公司附近租住"
    memories = convert_profile_to_memories(imported.profile, imported.career_template)
    loc = [m for m in memories if m["sub_category"] == "现居地"]
    assert len(loc) == 1 and "在公司附近租住" in loc[0]["content"]
