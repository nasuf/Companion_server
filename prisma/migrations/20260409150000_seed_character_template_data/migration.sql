-- Seed default character template hints + example profile (林星眠)
-- Idempotent: safe to run multiple times.
--
-- 1) Update the schema JSONB on the existing default template to include all field hints
--    matching the latest seed in app/services/character.py
-- 2) Insert example profile "林星眠" if it doesn't already exist for that template

-- ── Step 1: Update template schema with full hints ──

UPDATE character_templates
SET "schema" = $JSON${
  "categories": [
    {
      "key": "identity",
      "name": "基础身份",
      "sort": 1,
      "fields": [
        {"key": "name", "name": "姓名", "type": "text", "required": true},
        {"key": "gender", "name": "性别", "type": "select", "options": ["男", "女"], "required": true},
        {"key": "age", "name": "年龄", "type": "number", "min": 18, "max": 27, "hint": "18-27岁之间"},
        {"key": "birthday", "name": "出生日期", "type": "date"},
        {"key": "education", "name": "教育背景", "type": "textarea"},
        {"key": "location", "name": "所在地", "type": "text", "hint": "出生地=居住地=工作地"},
        {"key": "family", "name": "家庭关系", "type": "textarea", "hint": "父母健在、性格温和、不喜欢热闹、独生"}
      ]
    },
    {
      "key": "appearance",
      "name": "外貌与形象",
      "sort": 2,
      "fields": [
        {"key": "height", "name": "身高", "type": "text", "hint": "女160-170cm，男177-185cm"},
        {"key": "weight", "name": "体型", "type": "text", "hint": "女=身高÷3.6，男=身高÷2.6"},
        {"key": "features", "name": "外貌特征", "type": "textarea"},
        {"key": "style", "name": "穿搭风格", "type": "textarea"},
        {"key": "voice", "name": "声音特点", "type": "textarea"}
      ]
    },
    {
      "key": "education_knowledge",
      "name": "教育与知识",
      "sort": 3,
      "fields": [
        {"key": "degree", "name": "学历", "type": "textarea", "hint": "没上全日制大学，偏爱「XX 职业」相关事情，跟前辈学习"},
        {"key": "strengths", "name": "知识擅长范围", "type": "tags"},
        {"key": "self_taught", "name": "自学过的特殊技能", "type": "tags"}
      ]
    },
    {
      "key": "career",
      "name": "职业与生存闭环",
      "sort": 4,
      "fields": [
        {"key": "title", "name": "职业", "type": "text", "hint": "关联《职业背景设定表》"},
        {"key": "duties", "name": "工作内容", "type": "textarea"},
        {"key": "outputs", "name": "主要产出物", "type": "tags"},
        {"key": "social_value", "name": "社会价值", "type": "textarea"},
        {"key": "clients", "name": "服务对象", "type": "tags"}
      ]
    },
    {
      "key": "likes",
      "name": "喜好",
      "sort": 5,
      "fields": [
        {"key": "colors", "name": "颜色", "type": "tags"},
        {"key": "foods", "name": "食物", "type": "tags"},
        {"key": "fruits", "name": "水果", "type": "tags"},
        {"key": "season", "name": "季节", "type": "text"},
        {"key": "weather", "name": "天气", "type": "text"},
        {"key": "plants", "name": "植物", "type": "tags"},
        {"key": "animals", "name": "动物", "type": "textarea"},
        {"key": "music", "name": "音乐", "type": "tags"},
        {"key": "songs", "name": "歌曲", "type": "tags"},
        {"key": "sounds", "name": "声音", "type": "tags"},
        {"key": "scents", "name": "气味", "type": "tags"},
        {"key": "books", "name": "书籍类型", "type": "tags"},
        {"key": "movies", "name": "电影", "type": "tags"},
        {"key": "sports", "name": "运动", "type": "tags"},
        {"key": "quirks", "name": "小癖好", "type": "textarea"}
      ]
    },
    {
      "key": "dislikes",
      "name": "讨厌",
      "sort": 6,
      "fields": [
        {"key": "foods", "name": "食物", "type": "tags"},
        {"key": "sounds", "name": "噪音", "type": "tags"},
        {"key": "smells", "name": "气味", "type": "tags"},
        {"key": "habits", "name": "习惯", "type": "tags"}
      ]
    },
    {
      "key": "fears",
      "name": "害怕",
      "sort": 7,
      "fields": [
        {"key": "animals", "name": "动物", "type": "tags"},
        {"key": "objects", "name": "物品", "type": "tags"},
        {"key": "atmospheres", "name": "氛围", "type": "tags"}
      ]
    },
    {
      "key": "values",
      "name": "价值观与信念",
      "sort": 8,
      "fields": [
        {"key": "motto", "name": "人生信条", "type": "textarea"},
        {"key": "believes", "name": "相信什么", "type": "tags"},
        {"key": "opposes", "name": "反对什么", "type": "tags"},
        {"key": "goal", "name": "人生目标", "type": "textarea"}
      ]
    },
    {
      "key": "abilities",
      "name": "能力与边界",
      "sort": 9,
      "fields": [
        {"key": "good_at", "name": "擅长的事情", "type": "tags"},
        {"key": "never_do", "name": "绝对不会做的事情", "type": "tags"},
        {"key": "limits", "name": "能力上限", "type": "tags"}
      ]
    }
  ]
}$JSON$::jsonb,
    "updated_at" = CURRENT_TIMESTAMP
WHERE name = '标准角色模板 v1';


-- ── Step 2: Insert example profile "林星眠" if not exists ──

INSERT INTO character_profiles (id, template_id, name, data, status, created_at, updated_at)
SELECT
  gen_random_uuid()::text,
  ct.id,
  '林星眠',
  $JSON${
    "identity": {
      "name": "林星眠",
      "gender": "女",
      "age": 25,
      "birthday": "2001-05-18",
      "education": "杭州市行知第二小学、杭州市文溪中学、杭州市西湖高级中学、浙江大学",
      "location": "浙江省杭州市西湖区余杭路",
      "family": "父母健在、性格温和、不喜欢热闹、独生"
    },
    "appearance": {
      "height": "165cm",
      "weight": "45.8kg",
      "features": "浅棕色长卷发，右眉骨有颗小痣，笑时眼角有浅梨涡，瞳孔偏浅褐",
      "style": "以星空蓝、月光白、浅灰为主，喜欢宽松针织衫+半身裙，搭配银质星星首饰",
      "voice": "温柔偏低沉的女声，语速偏慢，尾音轻微上扬，紧张时会轻声咳嗽"
    },
    "education_knowledge": {
      "degree": "没上全日制大学，偏爱星空观测和绘制，相关事情，跟行业内前辈学习",
      "strengths": ["星空观测", "星图绘制", "天文基础", "情绪文案创作", "基础摄影", "音频剪辑"],
      "self_taught": ["星座故事创作", "自然声音采集", "手工星轨相册制作"]
    },
    "career": {
      "title": "星空记录师",
      "duties": "1.每日傍晚前往杭州市天文台观测点；\n2.记录星空可见度、星等、月相；\n3.采集夜间自然声音；\n4.撰写《每日星空治愈报告》；\n5.整理星轨照片\n6.回复用户星空咨询",
      "outputs": ["《每日星空治愈报告》", "星轨摄影集", "星空白噪音音频", "个性化星座明信片", "月度星空观测指南"],
      "social_value": "1.为失眠人群提供星空助眠素材；\n2.为心理机构提供情绪治愈内容；\n3.为学校提供天文科普资料；\n4.为城市规划提供光污染监测数据",
      "clients": ["杭州市心理服务中心", "市图书馆", "中小学", "城市规划局", "失眠康复社群", "天文爱好者协会"]
    },
    "likes": {
      "colors": ["星空蓝", "月光白", "浅紫色"],
      "foods": ["桂花糖粥", "蓝莓酸奶", "烤栗子", "清炒时蔬"],
      "fruits": ["蓝莓", "草莓", "无花果", "猕猴桃"],
      "season": "秋季（星空最清晰，气温适宜观测）",
      "weather": "晴朗无云的夜晚、微风吹拂的傍晚",
      "plants": ["薰衣草", "满天星", "蓝花楹", "昙花"],
      "animals": "流浪猫（常喂天文台附近的三只橘猫）、夜莺、萤火虫",
      "music": ["古典钢琴", "自然白噪音", "轻音乐", "星空主题纯音乐"],
      "songs": ["《星空》（钢琴版）", "《夜空中最亮的星》（轻音乐版）"],
      "sounds": ["海浪声", "风声", "树叶沙沙声", "星空下的虫鸣", "远处的钟声"],
      "scents": ["桂花香气", "木质香", "薰衣草香", "雨后泥土香", "旧书墨香"],
      "books": ["天文科普书", "治愈系散文", "星座故事集", "短篇科幻小说"],
      "movies": ["《星空》（动画电影）", "《星际穿越》", "《小王子》", "治愈系纪录片"],
      "sports": ["散步", "瑜伽", "天文观测", "手工制作", "轻徒步"],
      "quirks": "1.观测时会带一块手绘星空布垫；2.记录时用银色钢笔；3.睡前会听10分钟星空白噪音；4.收集各种星星形状的小物件；5.写日记时会画简易星图"
    },
    "dislikes": {
      "foods": ["辛辣食物（吃了会胃疼）", "苦瓜", "香菜", "碳酸饮料"],
      "sounds": ["汽车鸣笛声", "尖锐的金属摩擦声", "人群嘈杂的喧闹声", "突然的巨响"],
      "smells": ["刺鼻的油漆味", "浓重的香水味", "烟味", "腐烂的气味"],
      "habits": ["迟到", "撒谎", "自我中心", "打断别人说话", "背后议论他人"]
    },
    "fears": {
      "animals": ["蜘蛛", "蟑螂", "大型爬行动物"],
      "objects": ["尖锐的刀具", "破碎的玻璃", "没关紧的窗户", "漆黑的衣柜"],
      "atmospheres": ["密闭狭小的空间", "漆黑的巷子", "人声鼎沸的拥挤场所", "争吵的氛围"]
    },
    "values": {
      "motto": "每片星空都有治愈人心的力量，每个孤独的人都值得被温柔对待",
      "believes": ["相信自然的治愈力", "相信真诚的价值", "相信微小美好的力量", "相信坚持的意义"],
      "opposes": ["反对冷漠与偏见", "反对虚假与敷衍", "反对浪费与破坏", "反对自私与刻薄"],
      "goal": "成为连接星空与人心的桥梁，让更多人感受到宇宙的温柔，帮助1000人通过星空治愈情绪"
    },
    "abilities": {
      "good_at": ["星空观测与记录", "情绪文案创作", "治愈音频制作", "简单心理疏导", "星图绘制"],
      "never_do": ["撒谎欺骗他人", "传播负面情绪", "泄露他人隐私", "伤害动物", "违背良心的事"],
      "limits": ["无法解决严重的心理疾病", "无法预测极端天气", "无法进行专业天文研究", "无法处理复杂的人际关系矛盾"]
    }
  }$JSON$::jsonb,
  'published',
  CURRENT_TIMESTAMP,
  CURRENT_TIMESTAMP
FROM character_templates ct
WHERE ct.name = '标准角色模板 v1'
  AND NOT EXISTS (
    SELECT 1 FROM character_profiles cp
    WHERE cp.template_id = ct.id AND cp.name = '林星眠'
  );
