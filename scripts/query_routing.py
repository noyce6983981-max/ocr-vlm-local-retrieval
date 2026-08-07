"""Deterministic query routing for OCR, visual, and metadata evidence."""

from __future__ import annotations

import logging
import re
from functools import lru_cache


TEXT_EVIDENCE_TERMS = (
    "提到",
    "出现",
    "写着",
    "包含",
    "记录",
    "表格",
    "编号",
    "多少",
    "成绩",
    "课程",
    "公式",
    "关键词",
    "日期",
    "地址",
)

VISUAL_FORM_TERMS = (
    "图片",
    "照片",
    "画面",
    "截图",
    "海报",
    "自然图像",
    "自然图片",
    "自然照片",
    "风景",
    "人物照片",
    "动物照片",
    "建筑照片",
    "课件",
    "幻灯片",
    "演示文稿",
)

VISUAL_STRUCTURE_TERMS = (
    "广告牌",
    "街景",
    "白板",
    "流程图",
    "柱状图",
    "折线图",
    "图表",
    "仪表盘",
    "菜单栏",
    "工具栏",
    "搜索框",
    "定位标记",
    "商店招牌",
    "路牌",
    "印章",
    "签字栏",
    "日期栏",
    "复选框",
    "表格边框",
    "填写栏",
    "倾斜拍摄",
    "阴影遮挡",
    "三栏排版",
    "登录界面",
    "错误提示",
    "弹窗",
    "命令行窗口",
    "手写",
    "包装盒",
    "菜单",
    "价目表",
    "通知截图",
    "全家福",
    "购物清单",
    "自行车",
    "放风筝",
    "帽子",
    "帐篷",
)

STRONG_LITERAL_TEXT_TERMS = (
    "提到",
    "写着",
    "写有",
    "文字",
    "英文",
    "代码",
    "姓名",
    "地址",
    "日期",
    "时间",
    "地点",
    "金额",
    "标题",
    "关键词",
    "课程",
    "记录",
    "计划",
)

TEMPORAL_PERSONAL_TERMS = (
    "我去年",
    "我上个月",
    "我上周",
    "我昨天",
    "我昨晚",
    "去年",
    "上个月",
    "上周",
    "昨天",
    "昨晚",
)

TOPIC_QUERY_NOISE = (
    "请帮我",
    "请查找",
    "查找",
    "搜索",
    "找一下",
    "找一张",
    "找一页",
    "找",
    "标题里面",
    "标题里",
    "标题中",
    "哪一页",
    "那一页",
    "文档",
    "文件",
    "资料",
    "页面",
    "图片",
    "照片",
    "截图",
    "课件",
    "演示页面",
    "演示文稿",
    "幻灯片",
    "查到",
    "带有",
    "包含",
    "提到",
    "介绍",
    "列出",
    "展示",
    "显示",
    "写有",
    "写着",
)

VISUAL_DISCOVERY_TERMS = (
    "蓝色",
    "红色",
    "绿色",
    "黄色",
    "黑色",
    "白色",
    "紫色",
    "粉色",
    "橙色",
    "灰色",
    "棕色",
    "青色",
    "美丽",
    "漂亮",
    "好看",
    "风景",
    "森林",
    "山林",
    "山景",
    "山水",
    "山水画",
    "风景画",
    "自然",
    "感觉",
    "类似",
    "天空",
    "白云",
    "蓝天",
    "山峰",
    "雪山",
    "湖泊",
    "海边",
    "海面",
    "沙滩",
    "树木",
    "日出",
    "日落",
    "夕阳",
    "倒影",
    "动物",
    "建筑",
    "车辆",
    "帆船",
    "花朵",
    "树林",
    "草原",
    "田野",
    "河流",
    "瀑布",
    "沙漠",
    "峡谷",
    "大海",
    "海洋",
    "海岸",
    "雪景",
    "城市夜景",
    "夜景",
    "公园",
    "花园",
    "草地",
    "猫",
    "狗",
    "鸟",
    "跑车",
    "火车",
    "轮船",
    "肖像",
    "天地",
)

COLOR_QUERY_ALIASES = {
    "blue": ("蓝", "蓝色"),
    "red": ("红", "红色"),
    "green": ("绿", "绿色"),
    "yellow": ("黄", "黄色"),
    "black": ("黑", "黑色"),
    "white": ("白", "白色"),
    "purple": ("紫", "紫色"),
    "pink": ("粉", "粉色", "粉红色"),
    "orange": ("橙", "橙色"),
    "gray": ("灰", "灰色", "灰白色"),
    "brown": ("棕", "棕色", "褐色"),
    "cyan": ("青", "青色", "青蓝色"),
}

COLOR_QUERY_EXPANSIONS = {
    "blue": "以蓝色为主要色调，具有大面积明显蓝色区域的真实图片",
    "red": "以红色为主要色调，具有大面积明显红色区域的真实图片",
    "green": "以绿色为主要色调，具有大面积明显绿色区域的真实图片",
    "yellow": "以黄色为主要色调，具有大面积明显黄色区域的真实图片",
    "black": "以黑色为主要色调，具有大面积明显黑色区域的真实图片",
    "white": "以白色为主要色调，具有大面积明显白色区域的真实图片",
    "purple": "以紫色为主要色调，具有大面积明显紫色区域的真实图片",
    "pink": "以粉色为主要色调，具有大面积明显粉色区域的真实图片",
    "orange": "以橙色为主要色调，具有大面积明显橙色区域的真实图片",
    "gray": "以灰色为主要色调，具有大面积明显灰色区域的真实图片",
    "brown": "以棕色为主要色调，具有大面积明显棕色区域的真实图片",
    "cyan": "以青色或青蓝色为主要色调的真实图片",
}

TECHNICAL_TOPIC_MARKERS = (
    "研究",
    "工程",
    "算法",
    "模型",
    "系统",
    "方法",
    "理论",
    "数据",
    "课程",
    "论文",
    "语言处理",
    "图像处理",
    "识别",
    "检测",
    "分类",
    "分析",
    "训练",
    "生态学",
    "建筑学",
    "动物学",
)

QUESTION_TERMS = (
    "哪张",
    "哪份",
    "哪些",
    "什么",
    "谁",
    "是否",
    "多少",
    "怎么",
    "怎样",
    "为何",
    "为什么",
    "如何",
    "哪里",
    "何时",
    "有没有",
)

COMMON_SINGLE_SURNAMES = frozenset(
    "赵钱孙李周吴郑王冯陈蒋沈韩杨朱秦许何吕施张孔曹严华金魏陶姜"
    "谢邹喻柏窦章云苏潘葛范彭鲁韦昌马苗方俞任袁柳鲍史唐费廉岑"
    "薛雷贺倪汤滕殷罗毕郝邬安常乐于傅皮卞齐康伍余顾孟平黄和"
    "穆萧尹姚邵汪祁毛禹狄米贝明臧计伏成戴宋茅庞熊纪舒屈项祝"
    "董梁杜阮蓝闵席季麻强贾路娄江童颜郭梅盛林刁钟徐邱骆高夏"
    "蔡田樊胡凌霍虞万柯管卢莫房裘解应宗丁宣邓郁单杭洪包诸左"
    "石崔吉龚程邢裴陆荣翁羊惠甄曲封储靳段富巫乌焦巴牧山谷车"
    "侯全班仰秋仲伊宫宁栾甘厉祖武符刘景詹束龙叶幸司黎薄宿白"
    "怀蒲鄂索赖卓屠蒙池乔谭贡劳姬申扶冉桑桂牛寿通边燕冀浦尚"
    "农温庄晏柴瞿阎慕连茹习艾鱼容向古易慎戈廖耿满弘匡国文寇"
    "广聂晁冷辛简饶曾沙关游权盖桓欧沃利蔚越师巩敖融"
)

COMMON_COMPOUND_SURNAMES = (
    "欧阳",
    "司马",
    "上官",
    "诸葛",
    "东方",
    "皇甫",
    "尉迟",
    "公孙",
    "慕容",
    "司徒",
    "司空",
    "令狐",
    "轩辕",
    "宇文",
    "长孙",
    "南宫",
    "夏侯",
)

ENTITY_QUERY_PREFIXES = ("请搜索", "搜索", "请查找", "查找", "查一下", "找一下", "找", "查")
ENTITY_QUERY_SUFFIXES = (
    "的相关资料",
    "相关资料",
    "的资料",
    "的信息",
    "的照片",
    "的图片",
    "是谁",
    "资料",
    "照片",
    "图片",
)

PERSON_CONTEXT_TERMS = ("姓名", "人名", "这个人", "人物", "是谁")
NON_PERSON_POS_TAGS = frozenset(
    {"n", "ns", "nt", "nz", "v", "vn", "a", "ad", "an", "l", "i", "s", "f", "t", "m", "eng"}
)
NON_PERSON_TERM_SUFFIXES = (
    "工程",
    "系统",
    "模型",
    "算法",
    "数据",
    "网络",
    "技术",
    "方法",
    "理论",
    "数学",
    "力学",
    "化学",
    "物理",
    "林业",
    "农业",
    "工业",
    "程式",
    "马达",
    "性能",
    "质量",
    "水平",
    "荣耀",
)


@lru_cache(maxsize=512)
def lexical_person_name_decision(candidate: str) -> bool | None:
    """Use lightweight Chinese lexical evidence before surname heuristics."""
    try:
        import jieba
        import jieba.posseg as posseg
    except ImportError:
        return None
    jieba.setLogLevel(logging.WARNING)
    tokens = [(token.word, token.flag) for token in posseg.cut(candidate)]
    if len(tokens) != 1 or tokens[0][0] != candidate:
        return None
    flag = tokens[0][1]
    if flag.startswith("nr"):
        return True
    if flag in NON_PERSON_POS_TAGS:
        return False
    return None


def extract_strict_entity_term(query: str) -> str | None:
    """Extract a short Chinese personal name that needs exact evidence."""
    normalized_query = "".join(query.split()).strip(
        "，。！？?：:；;、\"'“”‘’"
    )
    contextual_match = re.search(
        r"(?:叫|名为|姓名(?:是|为)|人名(?:是|为))"
        r"([\u3400-\u9fff]{2,4}?)"
        r"(?=的人|这个人|是谁|[，。！？?：:；;、]|$)",
        normalized_query,
    )
    candidate = (
        contextual_match.group(1)
        if contextual_match
        else normalized_query
    )
    for prefix in ENTITY_QUERY_PREFIXES:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    for suffix in ENTITY_QUERY_SUFFIXES:
        if candidate.endswith(suffix):
            candidate = candidate[: -len(suffix)]
            break
    if not re.fullmatch(r"[\u3400-\u9fff]{2,4}", candidate):
        return None
    has_person_context = any(
        term in normalized_query for term in PERSON_CONTEXT_TERMS
    )
    # Known visual and technical topics must win over the weak signal that
    # their first character also happens to be a Chinese surname. Explicit
    # person wording such as “是谁” can still override this ambiguity.
    has_visual_topic = any(
        len(term) >= 2 and term in candidate
        for term in VISUAL_DISCOVERY_TERMS
    )
    has_domain_topic_suffix = candidate.endswith(
        NON_PERSON_TERM_SUFFIXES
    )
    if (
        not has_person_context
        and (has_visual_topic or has_domain_topic_suffix)
    ):
        return None
    if has_person_context and (
        candidate.startswith(COMMON_COMPOUND_SURNAMES)
        or candidate[0] in COMMON_SINGLE_SURNAMES
    ):
        return candidate
    lexical_decision = lexical_person_name_decision(candidate)
    if lexical_decision is True:
        return candidate
    if lexical_decision is False:
        return None
    if candidate.startswith(COMMON_COMPOUND_SURNAMES):
        return candidate
    if candidate[0] in COMMON_SINGLE_SURNAMES:
        return candidate
    return None


def classify_query_form(query: str) -> str:
    """Classify by evidence need using intent, lexical, and form signals.

    The order is deliberate: explicit text evidence beats visual wording,
    known visual subjects beat weak surname heuristics, and only then does a
    short lexical entity become an exact lookup. This mirrors production
    search pipelines that route by intent instead of applying one threshold
    to every query.
    """
    normalized = " ".join(query.split())
    has_visual_structure = any(
        term in normalized for term in VISUAL_STRUCTURE_TERMS
    )
    if has_visual_structure:
        has_literal_text = any(
            term in normalized for term in STRONG_LITERAL_TEXT_TERMS
        )
        return "mixed" if has_literal_text else "visual_metadata"
    if any(term in normalized for term in TEXT_EVIDENCE_TERMS):
        return "text_evidence"
    if extract_quoted_terms(normalized) and any(
        term in normalized for term in VISUAL_FORM_TERMS
    ):
        return "visual_metadata"
    if is_visual_discovery_query(normalized):
        return "visual_discovery"
    if extract_strict_entity_term(normalized):
        return "entity_exact"
    if any(term in normalized for term in VISUAL_FORM_TERMS):
        return "visual_metadata"
    if is_keyword_discovery_query(normalized):
        return "topic_discovery"
    return "mixed"


def extract_topic_evidence_terms(query: str) -> list[str]:
    """Extract conservative literal phrases for topic-result boosting.

    These phrases are evidence hints, never standalone proof that a compound
    query is fully satisfied.
    """
    normalized = "".join(query.split()).strip(
        "，。！？?：:；;、\"'“”‘’"
    )
    for noise in sorted(TOPIC_QUERY_NOISE, key=len, reverse=True):
        normalized = normalized.replace(noise, " ")
    pieces = re.split(r"(?:以及|并且|同时|和|与|或|、|的|里|中)", normalized)
    generic = {"一张", "一页", "一个", "学术", "技术", "正式", "旧"}
    result: list[str] = []
    for piece in pieces:
        cleaned = piece.strip()
        cleaned = re.sub(r"^(?:有|含|带)", "", cleaned)
        if len(cleaned) < 2 or cleaned in generic:
            continue
        if cleaned not in result:
            result.append(cleaned)
        if cleaned.endswith("列表") and len(cleaned) > 4:
            core = cleaned[: -len("列表")]
            if core not in result:
                result.append(core)
    return result


def extract_temporal_personal_terms(query: str) -> list[str]:
    """Return user-relative time constraints that need indexed evidence."""
    normalized = "".join(query.split())
    return [term for term in TEMPORAL_PERSONAL_TERMS if term in normalized]


def extract_quoted_terms(query: str) -> list[str]:
    """Extract explicit literal evidence enclosed in Chinese or ASCII quotes."""
    result: list[str] = []
    for chinese, ascii_value in re.findall(
        r"“([^”]+)”|\"([^\"]+)\"", query
    ):
        value = (chinese or ascii_value).strip()
        if len(value) >= 2 and value not in result:
            result.append(value)
    return result


def detect_color_intent(query: str) -> str | None:
    """Return one unambiguous requested color without matching words like 蓝牙."""
    normalized = "".join(query.split()).strip(
        "，。！？?：:；;、\"'“”‘’"
    )
    matches: set[str] = set()
    for color_name, aliases in COLOR_QUERY_ALIASES.items():
        for alias in aliases:
            if len(alias) >= 2 and alias in normalized:
                matches.add(color_name)
            elif normalized == alias:
                matches.add(color_name)
    return next(iter(matches)) if len(matches) == 1 else None


def is_pure_color_query(query: str) -> bool:
    """Distinguish '蓝' from a compound constraint such as '蓝色汽车'."""
    color_intent = detect_color_intent(query)
    if not color_intent:
        return False
    remainder = "".join(query.split()).strip(
        "，。！？?：:；;、\"'“”‘’"
    )
    for filler in (
        "请",
        "帮我",
        "查找",
        "搜索",
        "找",
        "一张",
        "图片",
        "照片",
        "画面",
        "颜色",
        "色调",
        "为主",
        "主要",
        "的",
    ):
        remainder = remainder.replace(filler, "")
    for alias in sorted(
        COLOR_QUERY_ALIASES[color_intent], key=len, reverse=True
    ):
        remainder = remainder.replace(alias, "")
    return not remainder


def infer_retrieval_route(query: str) -> str:
    """Backward-compatible public entry point for query-form routing."""
    return classify_query_form(query)


def is_visual_discovery_query(query: str) -> bool:
    """Return true for visual topics that should blend related imagery."""
    normalized = " ".join(query.split())
    if any(term in normalized for term in TEXT_EVIDENCE_TERMS):
        return False
    if extract_strict_entity_term(normalized):
        return False
    if detect_color_intent(normalized):
        return True
    has_visual_form = any(
        term in normalized for term in VISUAL_FORM_TERMS
    )
    if (
        any(term in normalized for term in TECHNICAL_TOPIC_MARKERS)
        and not has_visual_form
    ):
        return False
    if any(term in normalized for term in VISUAL_DISCOVERY_TERMS):
        return True
    return False


def is_exploratory_query(query: str) -> bool:
    """Compatibility alias; there is no standalone exploration mode."""
    return is_visual_discovery_query(query)


def is_keyword_discovery_query(query: str) -> bool:
    """Detect a browse-style keyword query that should favor recall.

    A short topic keyword should not share the no-result policy used by a
    factual question or exact entity lookup. This lightweight classifier keeps
    that distinction deterministic and local.
    """
    normalized = " ".join(query.split()).strip(
        "，。！？?：:；;、\"'“”‘’"
    )
    if not normalized or len(normalized) > 16:
        return False
    if any(term in normalized for term in TEXT_EVIDENCE_TERMS):
        return False
    if extract_strict_entity_term(normalized):
        return False
    if any(term in normalized for term in QUESTION_TERMS):
        return False
    return bool(
        re.fullmatch(
            r"[\u3400-\u9fffA-Za-z0-9+#.\-\s]+",
            normalized,
        )
    )


def expand_visual_query(query: str) -> str:
    """Add concrete visual semantics to otherwise underspecified queries."""
    normalized = " ".join(query.split())
    if not is_visual_discovery_query(normalized):
        return normalized
    color_intent = detect_color_intent(normalized)
    if color_intent:
        if is_pure_color_query(normalized):
            return COLOR_QUERY_EXPANSIONS[color_intent]
        return (
            f"{normalized}，主体内容必须匹配，同时要求"
            f"{COLOR_QUERY_EXPANSIONS[color_intent]}"
        )
    if "天地" in normalized:
        return (
            "广阔天空与大地同框的自然风景，天空、地面、"
            "山川或原野清晰可见"
        )
    if "美丽" in normalized or "漂亮" in normalized or "好看" in normalized:
        return (
            "美丽的自然风景，青山、森林、湖泊、花朵和"
            "令人愉悦的真实画面"
        )
    if "森林" in normalized:
        return "茂密的绿色森林，树木、山林和自然风景的真实照片"
    return f"{normalized}，与该主题直接相关、主体清晰的真实图片"
