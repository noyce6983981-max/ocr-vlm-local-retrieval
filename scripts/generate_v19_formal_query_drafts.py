from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/evaluation/v19/formal"
METHOD_LOCK = OUTPUT_DIR / "method_lock.json"


@dataclass(frozen=True, slots=True)
class FamilySpec:
    split: str
    route: str
    slug: str
    queries: tuple[str, str, str, str]


def spec(
    split: str,
    route: str,
    slug: str,
    *queries: str,
) -> FamilySpec:
    if len(queries) != 4:
        raise ValueError(f"family {slug} must contain exactly four paraphrases")
    return FamilySpec(split, route, slug, tuple(queries))  # type: ignore[arg-type]


FAMILIES = (
    spec(
        "calibration",
        "text_evidence",
        "approved_budget",
        "这份方案最后核准的项目经费总额是多少？",
        "帮我查出材料里正式批下来的预算合计。",
        "文件明确核定了多少项目资金？",
        "我需要文中最终批准的经费总数。",
    ),
    spec(
        "calibration",
        "text_evidence",
        "acceptance_date",
        "报告把最终验收日期定在了哪一天？",
        "请确认材料中写明的项目验收时间。",
        "文件约定什么时候完成最终验收？",
        "帮我找出正文里登记的验收日期。",
    ),
    spec(
        "calibration",
        "text_evidence",
        "contact_phone",
        "申请材料中留下的联系人电话是什么？",
        "帮我查一下表里登记的联系电话。",
        "文件给出的项目联系人手机号是多少？",
        "我需要材料中明确填写的联络号码。",
    ),
    spec(
        "calibration",
        "text_evidence",
        "table_third_rank",
        "统计表中排名第三的是哪家机构？",
        "请告诉我表格第三位对应的单位名称。",
        "哪家单位在材料的排序表里排第三？",
        "帮我读取榜单第三行的机构。",
    ),
    spec(
        "calibration",
        "text_evidence",
        "exact_delivery_phrase",
        "哪份文件原文写了“分批交付”？",
        "帮我定位明确出现“分批交付”四个字的页面。",
        "材料里哪里提到了“分批交付”？",
        "我想找包含原句“分批交付”的那一页。",
    ),
    spec(
        "holdout",
        "text_evidence",
        "invoice_suffix",
        "发票号码最后四位具体是多少？",
        "帮我核对票据编号的末四位数字。",
        "材料中登记的发票号以哪四个数字结尾？",
        "请读取发票编号最后四位。",
    ),
    spec(
        "holdout",
        "text_evidence",
        "device_model",
        "附件中登记的设备型号是什么？",
        "帮我确认清单写明的机器型号。",
        "采购材料里列出的具体产品型号是哪一个？",
        "我需要表格中填写的设备型号信息。",
    ),
    spec(
        "holdout",
        "text_evidence",
        "submission_deadline",
        "通知要求最晚什么时候提交纸质材料？",
        "帮我查出文件规定的纸质版截止时间。",
        "材料里写的提交期限是哪一天？",
        "请确认正文明确要求的最晚报送日期。",
    ),
    spec(
        "holdout",
        "text_evidence",
        "reported_growth_rate",
        "报告最后公布的同比增长率是多少？",
        "请读取结论部分给出的同比增幅。",
        "材料明确写出的年度增长百分比是多少？",
        "帮我核对正文中的同比增长数据。",
    ),
    spec(
        "holdout",
        "text_evidence",
        "caption_device_name",
        "图下注释把这套装置称为什么？",
        "请读取图片下方给出的设备名称。",
        "材料中的图注写明这是什么装置？",
        "帮我找出插图说明里的装置名字。",
    ),
    spec(
        "calibration",
        "visual_discovery",
        "red_umbrella_rain",
        "想找雨夜街头撑着红伞的行人画面。",
        "有没有红色雨伞出现在湿漉漉街道上的照片？",
        "帮我翻出那张下雨时有人举红伞过街的图。",
        "我记得一幅夜间街景里最显眼的是红雨伞。",
    ),
    spec(
        "calibration",
        "visual_discovery",
        "deer_snow_forest",
        "找一只鹿穿过积雪树林的照片。",
        "哪张图里雪地森林中有鹿在行走？",
        "我想看冬天林间出现一头鹿的画面。",
        "帮我找白雪覆盖的树林和棕色鹿同框的图片。",
    ),
    spec(
        "calibration",
        "visual_discovery",
        "blue_kayak_lake",
        "有没有蓝色皮划艇漂在碧绿湖面上的画面？",
        "找一张清澈湖水里有蓝色小艇的照片。",
        "帮我翻出那幅蓝色独木舟停在绿色湖中的图。",
        "我需要蓝艇与翠绿色湖水同框的自然照片。",
    ),
    spec(
        "calibration",
        "visual_discovery",
        "chef_wok_flame",
        "找厨师颠锅时火焰腾起的照片。",
        "有没有厨房里师傅炒菜冒出大火的画面？",
        "我记得一张图中厨师正把锅抛起，旁边有火苗。",
        "帮我寻找餐馆后厨颠炒锅并出现火焰的图片。",
    ),
    spec(
        "calibration",
        "visual_discovery",
        "child_kite_grass",
        "想看孩子在草地上跑着放风筝的照片。",
        "哪幅画面里小朋友牵着风筝穿过绿色草坪？",
        "帮我找儿童在户外草地放飞风筝的图。",
        "有没有一个孩子奔跑、空中挂着风筝的场景？",
    ),
    spec(
        "holdout",
        "visual_discovery",
        "lighthouse_storm",
        "找暴风浪花拍打灯塔的照片。",
        "有没有乌云下海浪冲向白色灯塔的画面？",
        "我想看风暴海岸上灯塔被巨浪包围的图。",
        "帮我翻出那张天气阴沉、灯塔旁浪很高的图片。",
    ),
    spec(
        "holdout",
        "visual_discovery",
        "black_horse_beach",
        "找黑马在海滩边奔跑的照片。",
        "哪张画面里一匹深色马沿着沙滩跑？",
        "我想看黑色骏马和海岸同时出现的图片。",
        "帮我找到海边沙地上正在奔跑的黑马。",
    ),
    spec(
        "holdout",
        "visual_discovery",
        "cyclist_mountain_turn",
        "找穿黄色骑行服的人在山路转弯的画面。",
        "有没有黄衣自行车手压弯经过盘山路的照片？",
        "我记得一张图中骑手穿黄衫驶过山间弯道。",
        "帮我寻找山路急弯与黄色骑行者同框的图片。",
    ),
    spec(
        "holdout",
        "visual_discovery",
        "balloons_desert_sunrise",
        "想找日出时沙漠上空飘着热气球的照片。",
        "哪张图能看到晨光、荒漠和多只热气球？",
        "帮我翻出热气球在金色沙地上方升起的画面。",
        "有没有沙漠清晨天空中出现彩色热气球的图片？",
    ),
    spec(
        "holdout",
        "visual_discovery",
        "cat_rain_window",
        "找一只猫坐在雨天窗台上的照片。",
        "哪幅画面里猫望着布满雨滴的窗户？",
        "我想看窗外下雨、猫趴在窗边的图片。",
        "帮我找到猫和带雨珠玻璃同框的室内场景。",
    ),
    spec(
        "calibration",
        "visual_metadata",
        "cover_bottom_logo",
        "找标题竖排居中、右下角放有校徽的封面。",
        "哪张封面把主标题竖着排在中间，徽标放在底部右侧？",
        "帮我定位中央竖排标题配右下角圆形标志的首页。",
        "我记得封面中线是竖标题，右下方还有一个校徽。",
    ),
    spec(
        "calibration",
        "visual_metadata",
        "table_frozen_column",
        "找第一列固定、表头使用深蓝底色的数据表。",
        "哪张表格左侧首列保持不动，上方标题栏是蓝色？",
        "帮我定位带冻结首列和深蓝表头的统计表。",
        "我需要版式上首列独立固定、顶部蓝色的表格页。",
    ),
    spec(
        "calibration",
        "visual_metadata",
        "dashboard_filter_map",
        "找左侧是筛选面板、右侧占大半区域为地图的仪表盘。",
        "哪张界面把过滤条件放在左栏，把地图放在右边？",
        "帮我定位左窄右宽、右侧展示地图的后台页面。",
        "我记得一个数据面板左边有筛选器，右边是一张大地图。",
    ),
    spec(
        "calibration",
        "visual_metadata",
        "two_column_footnotes",
        "找正文分双栏且页脚列有三条脚注的页面。",
        "哪一页上半部是左右两栏，底部还有三条脚注？",
        "帮我定位双栏正文配三项页脚注释的文档页。",
        "我需要版面为两列、页尾集中放脚注的那一页。",
    ),
    spec(
        "calibration",
        "visual_metadata",
        "four_images_row",
        "找四张图片横向排成一整行的幻灯片。",
        "哪页演示文稿把四幅小图并排放置？",
        "帮我定位页面中部横排四张等宽图片的课件。",
        "我记得有张幻灯片只有一行四个图块。",
    ),
    spec(
        "holdout",
        "visual_metadata",
        "right_checkbox_form",
        "找最右侧整列都是复选框的表单。",
        "哪张表把勾选项统一放在右边一列？",
        "帮我定位右端竖排多个选择框的登记表。",
        "我需要版式上最右栏用于打勾的表格页面。",
    ),
    spec(
        "holdout",
        "visual_metadata",
        "horizontal_flowchart",
        "找从左往右依次展开的五节点流程图。",
        "哪一页画着横向连接的五个步骤框？",
        "帮我定位箭头由左指向右的五阶段流程图。",
        "我记得一张图把五个流程节点水平排成一行。",
    ),
    spec(
        "holdout",
        "visual_metadata",
        "equation_numbered_notes",
        "找中央是一条大公式、右侧排着三条编号说明的页面。",
        "哪一页把主要公式放中间，并在右边列出1到3的注释？",
        "帮我定位中心公式配右侧编号解释的文档页。",
        "我需要公式居中、三项说明竖排在右边的版式。",
    ),
    spec(
        "holdout",
        "visual_metadata",
        "timeline_top_third",
        "找横向时间轴位于页面上方三分之一处的幻灯片。",
        "哪张课件把时间线放在顶部，下方留给说明文字？",
        "帮我定位上部横排节点、下部为正文的时间轴页。",
        "我记得一页的时间轴靠上摆放，下面还有大块内容。",
    ),
    spec(
        "holdout",
        "visual_metadata",
        "chart_legend_below",
        "找图表在左、图例单独横排在下方的页面。",
        "哪一页左侧放主图，底部另有一排颜色图例？",
        "帮我定位图形主体靠左且图例置于页面下缘的版式。",
        "我需要图表左对齐、说明色块横放在底部的那页。",
    ),
    spec(
        "calibration",
        "entity_exact",
        "person_liang_siyuan",
        "查一下梁思远。",
        "材料里有没有梁思远？",
        "帮我定位梁思远的记录。",
        "梁思远出现在哪些页面？",
    ),
    spec(
        "calibration",
        "entity_exact",
        "person_gu_qinghe",
        "帮我找顾清禾。",
        "看看文件中是否提到顾清禾。",
        "顾清禾对应的资料有哪些？",
        "我需要检索顾清禾这个名字。",
    ),
    spec(
        "calibration",
        "entity_exact",
        "person_tang_yichen",
        "检索唐亦辰。",
        "唐亦辰在哪份名单里？",
        "请定位材料中的唐亦辰。",
        "库里能查到唐亦辰吗？",
    ),
    spec(
        "calibration",
        "entity_exact",
        "org_yuanlan_tech",
        "查找远岚科技。",
        "哪些材料提到了远岚科技？",
        "帮我定位远岚科技的相关记录。",
        "检索实体“远岚科技”。",
    ),
    spec(
        "calibration",
        "entity_exact",
        "product_zx410",
        "查一下ZX-410。",
        "材料中有没有ZX-410？",
        "帮我定位型号ZX-410对应的记录。",
        "检索精确实体“ZX-410”。",
    ),
    spec(
        "holdout",
        "entity_exact",
        "person_ye_jianing",
        "找一下叶嘉宁。",
        "叶嘉宁被哪些文件提到？",
        "请检索叶嘉宁的记录。",
        "名单里能查到叶嘉宁吗？",
    ),
    spec(
        "holdout",
        "entity_exact",
        "person_su_jingxing",
        "查苏景行。",
        "帮我找出出现苏景行的页面。",
        "材料里有没有苏景行？",
        "定位苏景行对应的资料。",
    ),
    spec(
        "holdout",
        "entity_exact",
        "person_cheng_ruoan",
        "检索程若安。",
        "请查看程若安在哪份材料中。",
        "程若安这个名字有记录吗？",
        "帮我定位程若安。",
    ),
    spec(
        "holdout",
        "entity_exact",
        "project_xinghe_two",
        "查找“星河二号”。",
        "哪些记录属于星河二号？",
        "帮我定位实体“星河二号”。",
        "材料中是否出现星河二号？",
    ),
    spec(
        "holdout",
        "entity_exact",
        "lab_beichen",
        "检索北辰实验室。",
        "帮我找北辰实验室的相关记录。",
        "哪些文件提到了北辰实验室？",
        "定位实体“北辰实验室”。",
    ),
    spec(
        "calibration",
        "topic_discovery",
        "marine_carbon_sink",
        "整理一下海洋碳汇方向的资料。",
        "我想了解资料库里关于蓝碳研究的内容。",
        "有哪些材料讨论海洋生态系统固碳？",
        "帮我浏览海洋碳汇相关研究。",
    ),
    spec(
        "calibration",
        "topic_discovery",
        "battery_recycling",
        "找些动力电池回收方面的资料。",
        "库里有哪些内容涉及废旧电池循环利用？",
        "我想浏览电池回收产业相关文档。",
        "帮我汇总退役电池再利用方向的材料。",
    ),
    spec(
        "calibration",
        "topic_discovery",
        "smart_water",
        "看看智慧水务方向都有什么资料。",
        "帮我浏览城市供水数字化相关内容。",
        "资料库覆盖了哪些智能水务研究？",
        "我想了解水务监测与调度方面的材料。",
    ),
    spec(
        "calibration",
        "topic_discovery",
        "assistive_robotics",
        "找一些康复辅助机器人领域的文档。",
        "库里有没有面向残障辅助的机器人研究？",
        "帮我整理辅助机器人方向的资料。",
        "我想浏览康复机器人相关项目。",
    ),
    spec(
        "calibration",
        "topic_discovery",
        "precision_agriculture",
        "汇总精准农业方面的资料。",
        "有哪些文档讨论农业生产的精细化管理？",
        "帮我浏览智能农作与精准施肥相关内容。",
        "我想了解精准农业方向的研究材料。",
    ),
    spec(
        "holdout",
        "topic_discovery",
        "industrial_digital_twin",
        "找些工业数字孪生方向的资料。",
        "库里有哪些内容讨论工厂数字孪生？",
        "帮我浏览制造业虚实映射相关研究。",
        "我想了解工业数字孪生的项目与论文。",
    ),
    spec(
        "holdout",
        "topic_discovery",
        "thermal_storage",
        "整理热能储存方面的材料。",
        "有哪些资料涉及相变储热或蓄热系统？",
        "帮我浏览热储能技术相关文档。",
        "我想看看资料库对热能存储的覆盖情况。",
    ),
    spec(
        "holdout",
        "topic_discovery",
        "privacy_computing",
        "找一下隐私计算方向的研究资料。",
        "库里有哪些联邦学习与安全计算内容？",
        "帮我汇总数据可用不可见相关文档。",
        "我想浏览隐私保护计算方面的材料。",
    ),
    spec(
        "holdout",
        "topic_discovery",
        "rural_ecommerce",
        "有哪些资料讨论农村电商发展？",
        "帮我整理县域电商与农产品上行相关内容。",
        "我想看看乡村电子商务方面的案例。",
        "浏览一下农村电商主题的文档。",
    ),
    spec(
        "holdout",
        "topic_discovery",
        "disaster_logistics",
        "找些应急物流体系相关资料。",
        "库里有哪些内容研究灾后物资调度？",
        "帮我浏览救灾供应链方向的材料。",
        "我想了解应急物资运输与配送方面的研究。",
    ),
    spec(
        "calibration",
        "mixed",
        "worker_repair_sign",
        "找戴黄色安全帽并举着“检修中”牌子的工人照片。",
        "哪张图里黄帽工人手中的牌子写着“检修中”？",
        "帮我定位人物戴黄安全帽、标牌文字为“检修中”的画面。",
        "我记得一张工地照片中有人举着写有“检修中”的牌子。",
    ),
    spec(
        "calibration",
        "mixed",
        "second_prize_silver_seal",
        "找正文写着二等奖并带银色圆形印章的证书。",
        "哪份证书同时出现“二等奖”文字和银色圆章？",
        "帮我定位获奖等级为二等奖、右下方有银色印章的页面。",
        "我需要文字内容是二等奖且视觉上带银色圆章的证书。",
    ),
    spec(
        "calibration",
        "mixed",
        "sync_failure_bar_chart",
        "找顶部显示“同步失败”、下方有橙色柱状图的界面截图。",
        "哪张截图同时包含“同步失败”提示和橙色柱形图？",
        "帮我定位错误文字为同步失败、底部图表是橙色柱状图的页面。",
        "我记得一张后台界面上方报同步失败，下面放着橙色柱图。",
    ),
    spec(
        "calibration",
        "mixed",
        "chengdu_star_boundary",
        "找用星号标出成都并写有“规划边界”的地图。",
        "哪张地图同时把成都标成星号、旁边标注规划边界？",
        "帮我定位成都位置带星形符号且文字提到规划边界的页面。",
        "我需要地图上成都有星号，并能读到“规划边界”字样。",
    ),
    spec(
        "calibration",
        "mixed",
        "blue_bottle_acetone",
        "找实验台上蓝色瓶子标签写着“丙酮”的照片。",
        "哪张图里蓝瓶的标签文字是丙酮？",
        "帮我定位蓝色试剂瓶与“丙酮”标签同时出现的画面。",
        "我记得实验室照片中有个蓝瓶，上面写着丙酮。",
    ),
    spec(
        "holdout",
        "mixed",
        "red_truck_a17",
        "找车身编号为A17的红色卡车照片。",
        "哪张图里的红卡车侧面写着A17？",
        "帮我定位红色货车与编号A17同时出现的画面。",
        "我记得一辆红卡车，车门文字是A17。",
    ),
    spec(
        "holdout",
        "mixed",
        "timeline_trial_production",
        "找第三个节点写着“试生产”且用蓝色高亮的时间轴。",
        "哪页时间线的第三阶段标注试生产，并显示为蓝色？",
        "帮我定位第三节点文字为“试生产”、节点底色为蓝色的页面。",
        "我需要时间轴第三项既写试生产又采用蓝色强调。",
    ),
    spec(
        "holdout",
        "mixed",
        "accepted_green_stamp",
        "找正文写着“已验收”且右下角盖有绿色印章的表单。",
        "哪份表格同时出现已验收文字和右下方绿章？",
        "帮我定位状态为“已验收”、底部右侧有绿色印章的页面。",
        "我记得一张验收表写着已验收，右下角还有绿色章。",
    ),
    spec(
        "holdout",
        "mixed",
        "pie_chart_2024q3",
        "找左侧放饼图且标题写着“2024年第三季度”的页面。",
        "哪张报告页同时有左边饼图和2024年第三季度标题？",
        "帮我定位标题文字为“2024年第三季度”、饼图位于左侧的版面。",
        "我需要上方标题写2024年第三季度、左边是饼图的那页。",
    ),
    spec(
        "holdout",
        "mixed",
        "jersey_eight_trophy",
        "找穿八号白色球衣并举着奖杯的运动员照片。",
        "哪张比赛图中白衣球员号码为8，手里还举着奖杯？",
        "帮我定位球衣是白色、号码写8且人物举杯的画面。",
        "我记得一名白色八号球员正在把奖杯举过头顶。",
    ),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    family_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    route_split_index: Counter[tuple[str, str]] = Counter()
    for family in FAMILIES:
        key = (family.split, family.route)
        route_split_index[key] += 1
        family_id = (
            f"v19f_{family.split[:3]}_{family.route}_"
            f"{route_split_index[key]:02d}_{family.slug}"
        )
        query_ids: list[str] = []
        for paraphrase_index, query_text in enumerate(family.queries, start=1):
            query_id = f"{family_id}_p{paraphrase_index}"
            query_ids.append(query_id)
            query_rows.append(
                {
                    "query_id": query_id,
                    "family_id": family_id,
                    "split": family.split,
                    "gold_route": family.route,
                    "query_text": query_text,
                    "status": "codex_draft_pending_human_review",
                }
            )
        family_rows.append(
            {
                "family_id": family_id,
                "split": family.split,
                "proposed_route": family.route,
                "scenario": family.slug,
                "query_ids": query_ids,
                "queries": list(family.queries),
                "status": "codex_draft_pending_human_review",
            }
        )
    return family_rows, query_rows


def validate_rows(
    families: list[dict[str, Any]], queries: list[dict[str, Any]]
) -> None:
    if len(families) != 60 or len(queries) != 240:
        raise ValueError("formal draft must contain 60 families and 240 queries")
    if len({row["family_id"] for row in families}) != 60:
        raise ValueError("family IDs must be unique")
    if len({row["query_id"] for row in queries}) != 240:
        raise ValueError("query IDs must be unique")
    if len({row["query_text"] for row in queries}) != 240:
        raise ValueError("query texts must be unique")
    counts = Counter((row["split"], row["gold_route"]) for row in queries)
    expected = {
        (split, route): 20
        for split in ("calibration", "holdout")
        for route in (
            "text_evidence",
            "visual_discovery",
            "visual_metadata",
            "entity_exact",
            "topic_discovery",
            "mixed",
        )
    }
    if counts != expected:
        raise ValueError(f"formal route balance mismatch: {counts}")
    family_split = {row["family_id"]: row["split"] for row in families}
    if any(family_split[row["family_id"]] != row["split"] for row in queries):
        raise ValueError("a paraphrase family crosses splits")


def main() -> int:
    if not METHOD_LOCK.is_file():
        raise FileNotFoundError("V19 method lock must exist before authoring")
    families, queries = build_rows()
    validate_rows(families, queries)
    family_path = OUTPUT_DIR / "query_families_draft.jsonl"
    calibration_path = OUTPUT_DIR / "calibration_queries_draft.csv"
    holdout_path = OUTPUT_DIR / "holdout_queries_draft.csv"
    receipt_path = OUTPUT_DIR / "authoring_receipt.json"
    write_jsonl_atomic(family_path, families)
    write_csv_atomic(
        calibration_path,
        [row for row in queries if row["split"] == "calibration"],
    )
    write_csv_atomic(
        holdout_path,
        [row for row in queries if row["split"] == "holdout"],
    )
    receipt = {
        "schema_version": 1,
        "study_id": "v19-local-llm-structured-intent-routing",
        "status": "codex_draft_pending_human_review",
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "family_count": len(families),
        "query_count": len(queries),
        "calibration_query_count": 120,
        "holdout_query_count": 120,
        "queries_per_family": 4,
        "families_sha256": sha256_file(family_path),
        "calibration_sha256": sha256_file(calibration_path),
        "holdout_sha256": sha256_file(holdout_path),
        "eligible_for_final_claim": False,
    }
    write_json_atomic(receipt_path, receipt)
    print("V19 formal drafts generated: 60 families, 240 queries")
    print(f"Wrote {family_path}")
    print(f"Wrote {calibration_path}")
    print(f"Wrote {holdout_path}")
    print(f"Wrote {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
