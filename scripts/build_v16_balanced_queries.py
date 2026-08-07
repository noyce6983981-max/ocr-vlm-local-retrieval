"""Build the sealed, route-balanced V16 calibration and holdout queries.

The builder reads only library source metadata and OCR.  It never calls the
retrieval system.  This preserves the rule that query wording and labels are
frozen before product results are observed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from query_routing import infer_retrieval_route  # noqa: E402


MANIFEST_PATH = PROJECT_ROOT / "outputs/user_library/manifest.jsonl"
OLD_QUERY_FILES = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_retrieval_queries_formal.csv",
    PROJECT_ROOT / "data/evaluation/blind_study_v1/formal_queries.csv",
)
ROUTES = (
    "text_evidence",
    "visual_metadata",
    "visual_discovery",
    "mixed",
    "topic_discovery",
)


TEXT_SPECS = {
    "calibration": (
        ("user_aaed48df68b4", "舱单传输人", "理货报告提交人"),
        ("user_def2c7f7f795", "个人职业生涯规划", "向往的职位"),
        (
            "user_ffd23198339f",
            "一般纳税人转为小规模纳税人登记表",
            "统一社会信用代码",
        ),
        (
            "user_2022a42a27de",
            "技术难题解决需求征集表",
            "生产工艺技术简介",
        ),
        ("user_4d002433ea11", "公务员登记表", "现工作单位"),
        ("user_79803fac6565", "血清总胆固醇", "甘油三酯"),
    ),
    "holdout": (
        (
            "user_2b3df31ffed0",
            "教师专业技术职务晋升申报表",
            "环境科学与工程",
        ),
        (
            "user_6b11d8b81a20",
            "西安住房公积金管理中心",
            "协议缴款账户信息变更事项",
        ),
        (
            "user_f87449cd9ae7",
            "高等学校学生及家庭情况调查表",
            "学生本人基本情况",
        ),
        ("user_2f69d67d73c9", "人员岗位变动申请表", "变动性质"),
        ("user_20dd434e832b", "领取养老金方式确认表", "养老发放方式"),
        ("user_7614134029c5", "文件更改申请表", "更改前内容"),
    ),
}


VISUAL_CLASS_SPECS = {
    "calibration": {
        "visual_metadata": (
            ("umbrella", "找一张雨伞照片"),
            ("camera", "找一张照相机照片"),
            ("cellphone", "找一张手机照片"),
            ("chair", "找一张椅子照片"),
            ("beaver", "找一张海狸照片"),
            ("ibis", "找一张鹮类照片"),
        ),
        "visual_discovery": (
            ("butterfly", "找一张蝴蝶与花朵相关的动物照片"),
            ("dolphin", "找一张海豚动物照片"),
            ("elephant", "找一张大象动物照片"),
            ("kangaroo", "找一张袋鼠动物照片"),
            ("panda", "找一张熊猫动物照片"),
            ("sunflower", "找一张向日葵花朵照片"),
        ),
    },
    "holdout": {
        "visual_metadata": (
            ("bonsai", "找一张盆景照片"),
            ("joshua_tree", "找一张约书亚树照片"),
            ("water_lilly", "找一张睡莲照片"),
            ("helicopter", "找一张直升机照片"),
            ("ferry", "找一张渡轮照片"),
            ("minaret", "找一张宣礼塔照片"),
        ),
        "visual_discovery": (
            ("crab", "找一张螃蟹动物照片"),
            ("flamingo", "找一张火烈鸟动物照片"),
            ("llama", "找一张羊驼动物照片"),
            ("sea_horse", "找一张海马动物照片"),
            ("lotus", "找一张荷花花朵照片"),
            ("schooner", "找一张帆船照片"),
        ),
    },
}


MIXED_SPECS = {
    "calibration": (
        ("user_5f2ea52e35f8", "路牌", "金华兰溪", "常乐寺"),
        ("user_8265b534d98d", "路牌", "陇海快速路", "兴华南街方向"),
        ("user_6e1d4748eb40", "路牌", "北城村", "Beichengcun"),
        ("user_67a89907c7fa", "路牌", "店子村", "Dianzicun"),
        ("user_95ed509aeea4", "路牌", "五环路", "机场高速"),
        ("user_f662856aef3a", "路牌", "G311", "92"),
    ),
    "holdout": (
        ("user_e383e5f2550f", "路牌", "九鼎路", "祥盛路"),
        ("user_b1324a30f679", "路牌", "国家高速", "G6"),
        ("user_da2359a343cf", "广告牌", "EQUA KOLA", ""),
        ("user_e3badcef453b", "街景", "NISSAN", "GENISS"),
        ("user_306cfc3d5ef9", "街景", "HELLO", "HOLA"),
        ("user_11d5e9ddd927", "街景", "Biergarten", "BAYERN"),
    ),
}


TOPIC_SPECS = {
    "calibration": (
        ("user_3dad250e1a63", "机构其他身份信息"),
        ("user_439015dba4e5", "因身体健康原因申请缓考"),
        ("user_01a7f9439cdd", "人力资源需求申请表"),
        ("user_33f8d26f865f", "遗传资源来源披露登记表"),
        ("user_4ae807fd90e7", "个人出境信息登记表"),
        ("user_a0fb13d57a81", "招聘人员信息登记表"),
    ),
    "holdout": (
        ("user_e96b28a97d00", "学生社会实践登记表"),
        ("user_e0d4fbe34c32", "教师教学发展中心审核意见"),
        ("user_18405cbb1225", "流动人员人事档案管理"),
        ("user_8e74ced766a4", "预防接种"),
        ("user_b9d275c6fec4", "英语四六级证书"),
        ("user_f789d3bba98b", "企业国有产权受让登记表"),
    ),
}


NO_ANSWER_SPECS = {
    "calibration": {
        "text_evidence": (
            (
                "找写着“QCM-8842量子咖啡机验收通过”的表格",
                "QCM-8842量子咖啡机验收通过",
            ),
            (
                "哪份文档包含“火星温室第七码头”和“极光电池保修单”？",
                "火星温室第七码头",
            ),
            (
                "查找记录“月球酒店早餐券编号LUNA-731”的页面",
                "月球酒店早餐券编号LUNA-731",
            ),
            (
                "找标题里包含“海底邮局年度企鹅会议纪要”的文档",
                "海底邮局年度企鹅会议纪要",
            ),
        ),
        "visual_metadata": (
            ("找一张三只企鹅围着打印机的照片", "三只企鹅围着打印机"),
            ("找一张茶壶顶着键盘的照片", "茶壶顶着键盘"),
            ("找一张六台打字机堆成拱门的照片", "六台打字机堆成拱门"),
            ("找一张潜水员坐在办公室沙发上的照片", "潜水员坐在办公室沙发"),
        ),
        "visual_discovery": (
            ("找紫色热气球飞越火山熔岩的照片", "紫色热气球飞越火山熔岩"),
            ("找蓝色鲸鱼游过沙漠峡谷的照片", "蓝色鲸鱼游过沙漠峡谷"),
            ("找绿色跑车停在雪山顶峰的照片", "绿色跑车停在雪山顶峰"),
            ("找粉色帆船穿过瀑布的照片", "粉色帆船穿过瀑布"),
        ),
        "mixed": (
            ("找写着“月球车维修站”的霓虹商店招牌", "月球车维修站"),
            ("找菜单栏里写着“火星烤鱼配送”的应用截图", "火星烤鱼配送"),
            ("找路牌上写着“企鹅高速出口99”的街景照片", "企鹅高速出口99"),
            ("找登录界面里写着“木星农场管理员”的截图", "木星农场管理员"),
        ),
        "topic_discovery": (
            ("深海量子农业", "深海量子农业"),
            ("火星考古快递", "火星考古快递"),
            ("月球咖啡机维修", "月球咖啡机维修"),
            ("极光潜艇保险", "极光潜艇保险"),
        ),
    },
    "holdout": {
        "text_evidence": (
            (
                "哪份表格写着“星际快递柜ZX-904”和“木星站签收”？",
                "星际快递柜ZX-904",
            ),
            (
                "查找包含“量子雨伞维修协议QRA-662”的文件",
                "量子雨伞维修协议QRA-662",
            ),
            (
                "找记录“沙漠潜艇年度保养完成”的页面",
                "沙漠潜艇年度保养完成",
            ),
            (
                "哪份通知写着“火山冰箱暂停营业至2099年”？",
                "火山冰箱暂停营业至2099年",
            ),
        ),
        "visual_metadata": (
            ("找一张机器人举着竹篮的照片", "机器人举着竹篮"),
            ("找一张蜗牛趴在扫描仪上的照片", "蜗牛趴在扫描仪"),
            ("找一张钢琴装进透明冰箱的照片", "钢琴装进透明冰箱"),
            ("找一张钟表挂在热气球篮筐里的照片", "钟表挂在热气球篮筐"),
        ),
        "visual_discovery": (
            ("找橙色大象站在城市夜景屋顶的照片", "橙色大象站在城市夜景屋顶"),
            ("找红色火车驶过海底珊瑚森林的照片", "红色火车驶过海底珊瑚森林"),
            ("找黄色海豚跃过沙漠金字塔的照片", "黄色海豚跃过沙漠金字塔"),
            ("找青色直升机降落在冰川湖泊上的照片", "青色直升机降落在冰川湖泊"),
        ),
        "mixed": (
            ("找广告牌上写着“深海咖啡机场”的街景照片", "深海咖啡机场"),
            ("找错误提示中写着“量子雨伞连接失败”的截图", "量子雨伞连接失败"),
            ("找包装盒上写着“火山冰淇淋发动机”的照片", "火山冰淇淋发动机"),
            ("找价目表里写着“月球石每公斤三元”的照片", "月球石每公斤三元"),
        ),
        "topic_discovery": (
            ("木星园林会计", "木星园林会计"),
            ("火山企鹅物流", "火山企鹅物流"),
            ("量子陶瓷导航", "量子陶瓷导航"),
            ("星际雨伞金融", "星际雨伞金融"),
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_ocr_texts(
    manifest: list[dict[str, Any]], library_dir: Path
) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in manifest:
        item_id = item["item_id"]
        override = library_dir / "ocr/overrides" / f"{item_id}.json"
        baseline = library_dir / "ocr/json" / f"{item_id}.json"
        source = override if override.is_file() else baseline
        payload = json.loads(source.read_text(encoding="utf-8"))
        result[item_id] = normalize("\n".join(payload.get("rec_texts", [])))
    return result


def old_target_ids(paths: tuple[Path, ...]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                values = ";".join(
                    (
                        row.get("expected_item_id", ""),
                        row.get("relevant_item_ids", ""),
                    )
                )
                result.update(
                    value.strip()
                    for value in re.split(r"[;,|]", values)
                    if value.strip()
                )
    return result


def phrase_matches(
    phrases: tuple[str, ...], ocr_texts: dict[str, str]
) -> list[str]:
    normalized = tuple(normalize(phrase) for phrase in phrases if phrase)
    return sorted(
        item_id
        for item_id, text in ocr_texts.items()
        if all(phrase in text for phrase in normalized)
    )


def make_row(
    *,
    split: str,
    route: str,
    query: str,
    relevant_ids: list[str],
    expected_item_id: str,
    evidence: str,
    generation_method: str,
    serial: int,
) -> dict[str, str]:
    is_no_answer = not relevant_ids
    prefix = "cal" if split == "calibration" else "hold"
    return {
        "query_id": f"v16_{prefix}_{serial:03d}",
        "query": query,
        "query_type": route,
        "route": route,
        "is_no_answer": str(is_no_answer),
        "expected_item_id": expected_item_id,
        "relevant_item_ids": ";".join(relevant_ids),
        "split": split,
        "review_status": "frozen_ai_source_verified",
        "label_source": (
            "closed_taxonomy_negative"
            if is_no_answer
            else "source_metadata_or_ocr"
        ),
        "generation_method": generation_method,
        "evidence": evidence,
        "notes": (
            "visual negative requires pooled-candidate confirmation"
            if is_no_answer and route.startswith("visual")
            else ""
        ),
    }


def build_split_rows(
    split: str,
    manifest: list[dict[str, Any]],
    ocr_texts: dict[str, str],
    previous_targets: set[str],
) -> list[dict[str, str]]:
    by_id = {row["item_id"]: row for row in manifest}
    rows: list[dict[str, str]] = []

    for target, first, second in TEXT_SPECS[split]:
        phrases = (first, second)
        relevant = phrase_matches(phrases, ocr_texts)
        if target not in relevant:
            raise ValueError(f"Text evidence missing from {target}: {phrases}")
        query = f"哪份资料中同时出现“{first}”和“{second}”？"
        rows.append(
            make_row(
                split=split,
                route="text_evidence",
                query=query,
                relevant_ids=relevant,
                expected_item_id=target,
                evidence=" | ".join(phrases),
                generation_method="two_exact_ocr_phrases",
                serial=len(rows) + 1,
            )
        )

    for route in ("visual_metadata", "visual_discovery"):
        for category_tag, query in VISUAL_CLASS_SPECS[split][route]:
            relevant = sorted(
                row["item_id"]
                for row in manifest
                if category_tag in (row.get("source_tags") or [])
            )
            if not relevant:
                raise ValueError(f"No class members found for {category_tag}")
            rows.append(
                make_row(
                    split=split,
                    route=route,
                    query=query,
                    relevant_ids=relevant,
                    expected_item_id=relevant[0],
                    evidence=f"caltech101 class={category_tag}",
                    generation_method="closed_visual_taxonomy",
                    serial=len(rows) + 1,
                )
            )

    for target, structure, first, second in MIXED_SPECS[split]:
        phrases = tuple(value for value in (first, second) if value)
        relevant = phrase_matches(phrases, ocr_texts)
        if target not in relevant:
            raise ValueError(f"Mixed evidence missing from {target}: {phrases}")
        quoted = "和".join(f"“{value}”" for value in phrases)
        query = f"找一张{structure}照片，画面文字写着{quoted}"
        rows.append(
            make_row(
                split=split,
                route="mixed",
                query=query,
                relevant_ids=relevant,
                expected_item_id=target,
                evidence=" | ".join(phrases),
                generation_method="visual_structure_plus_exact_ocr",
                serial=len(rows) + 1,
            )
        )

    for target, topic in TOPIC_SPECS[split]:
        relevant = phrase_matches((topic,), ocr_texts)
        if target not in relevant:
            raise ValueError(f"Topic evidence missing from {target}: {topic}")
        rows.append(
            make_row(
                split=split,
                route="topic_discovery",
                query=topic,
                relevant_ids=relevant,
                expected_item_id=target,
                evidence=topic,
                generation_method="unique_source_topic_phrase",
                serial=len(rows) + 1,
            )
        )

    for route in ROUTES:
        for query, absence_anchor in NO_ANSWER_SPECS[split][route]:
            if any(normalize(absence_anchor) in text for text in ocr_texts.values()):
                raise ValueError(f"Negative anchor occurs in OCR: {absence_anchor}")
            rows.append(
                make_row(
                    split=split,
                    route=route,
                    query=query,
                    relevant_ids=[],
                    expected_item_id="",
                    evidence=f"absent anchor={absence_anchor}",
                    generation_method="closed_taxonomy_compositional_negative",
                    serial=len(rows) + 1,
                )
            )

    for row in rows:
        actual_route = infer_retrieval_route(row["query"])
        if actual_route != row["route"]:
            raise ValueError(
                f"Route mismatch {row['query_id']}: "
                f"expected={row['route']} actual={actual_route} "
                f"query={row['query']}"
            )
        relevant = set(row["relevant_item_ids"].split(";")) - {""}
        overlap = relevant & previous_targets
        if overlap:
            raise ValueError(
                f"New query reuses old targets {row['query_id']}: {sorted(overlap)}"
            )
        unknown = relevant - set(by_id)
        if unknown:
            raise ValueError(f"Unknown relevant IDs: {sorted(unknown)}")

    counts = Counter((row["route"], row["is_no_answer"]) for row in rows)
    expected = {
        (route, is_no_answer): count
        for route in ROUTES
        for is_no_answer, count in (("False", 6), ("True", 4))
    }
    if counts != expected:
        raise ValueError(f"Unbalanced split {split}: {counts}")
    return rows


def query_fingerprint(rows: list[dict[str, str]]) -> str:
    canonical = [
        {
            "query_id": row["query_id"],
            "query": row["query"],
            "query_type": row["query_type"],
            "is_no_answer": row["is_no_answer"] == "True",
            "relevant_item_ids": sorted(
                item_id
                for item_id in row["relevant_item_ids"].split(";")
                if item_id
            ),
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/evaluation/v16"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else PROJECT_ROOT / args.output_root
    )
    manifest = read_manifest(MANIFEST_PATH)
    ocr_texts = load_ocr_texts(manifest, MANIFEST_PATH.parent)
    previous_targets = old_target_ids(OLD_QUERY_FILES)
    split_rows = {
        split: build_split_rows(split, manifest, ocr_texts, previous_targets)
        for split in ("calibration", "holdout")
    }

    target_sets = {
        split: {
            item_id
            for row in rows
            for item_id in row["relevant_item_ids"].split(";")
            if item_id
        }
        for split, rows in split_rows.items()
    }
    target_overlap = target_sets["calibration"] & target_sets["holdout"]
    if target_overlap:
        raise ValueError(
            "Calibration and holdout target overlap: "
            + ", ".join(sorted(target_overlap))
        )

    created_at = datetime.now(timezone.utc).isoformat()
    for split, rows in split_rows.items():
        split_dir = output_root / split
        query_path = split_dir / "frozen_queries.csv"
        protocol_path = split_dir / "dataset_protocol.json"
        write_csv(query_path, rows)
        protocol = {
            "schema_version": 1,
            "name": f"v16_route_balanced_{split}",
            "status": "frozen_before_retrieval",
            "created_at": created_at,
            "query_file": query_path.relative_to(PROJECT_ROOT).as_posix(),
            "query_count": len(rows),
            "answerable_count": sum(
                row["is_no_answer"] == "False" for row in rows
            ),
            "no_answer_count": sum(
                row["is_no_answer"] == "True" for row in rows
            ),
            "route_counts": dict(Counter(row["route"] for row in rows)),
            "query_set_sha256": query_fingerprint(rows),
            "query_file_sha256": sha256_file(query_path),
            "library_manifest_sha256": sha256_file(MANIFEST_PATH),
            "old_query_file_sha256": {
                path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
                for path in OLD_QUERY_FILES
            },
            "old_relevant_target_count": len(previous_targets),
            "old_target_overlap_count": 0,
            "other_v16_split_target_overlap_count": 0,
            "authoring_inputs": [
                "library manifest metadata",
                "source taxonomy tags",
                "OCR text",
            ],
            "authoring_prohibited_inputs": [
                "live_search rankings",
                "retrieval scores",
                "accept/reject decisions",
            ],
            "label_limitations": [
                "Labels are AI source-verified, not human adjudicated.",
                "Compositional visual no-answer probes require pooled-candidate confirmation.",
                "The holdout may be run only after the selected V16 config is locked.",
            ],
        }
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{split}: queries={len(rows)} "
            f"fingerprint={protocol['query_set_sha256']}"
        )
        print(query_path)
        print(protocol_path)


if __name__ == "__main__":
    main()
