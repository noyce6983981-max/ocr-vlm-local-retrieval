"""Combine diagnostic ranks and metadata into a 30-query review queue."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_200_query_candidates.csv"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/user_library/manifest.jsonl"),
    )
    parser.add_argument(
        "--text-report",
        type=Path,
        default=Path(
            "outputs/evaluation/public200_diagnostic/text/report.json"
        ),
    )
    parser.add_argument(
        "--visual-report",
        type=Path,
        default=Path(
            "outputs/evaluation/public200_diagnostic/visual/report.json"
        ),
    )
    parser.add_argument(
        "--fusion-report",
        type=Path,
        default=Path(
            "outputs/evaluation/public200_diagnostic/fusion/report.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/evaluation/"
            "public_dataset_200_query_diagnostic_results.csv"
        ),
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=Path(
            "data/evaluation/"
            "public_dataset_200_query_review_queue_30.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "data/evaluation/"
            "public_dataset_200_query_diagnostic_summary.json"
        ),
    )
    parser.add_argument("--review-count", type=int, default=30)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rank_lookup(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["query_id"]: row for row in report["results"]}


def fusion_lookup(
    report: dict[str, Any], method: str
) -> dict[str, dict[str, Any]]:
    return {
        row["query_id"]: row
        for row in report[method]["results"]
    }


def priority_score(
    query: dict[str, Any],
    text_rank: int | None,
    visual_rank: int | None,
    adaptive_rank: int | None,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons = []
    flags = {
        value
        for value in str(query.get("ambiguity_flags", "")).split(",")
        if value
    }
    if not query.get("query"):
        score += 100
        reasons.append("blank_query")
    if "privacy_query_requires_manual" in flags:
        score += 80
        reasons.append("privacy")
    if "category_needs_confirmation" in flags:
        score += 8
        reasons.append("category")
    if "visual_description_needs_confirmation" in flags:
        score += 12
        reasons.append("visual_description")
    if "evidence_not_unique" in flags:
        score += 10
        reasons.append("nonunique_evidence")
    if (
        text_rank is not None
        and visual_rank is not None
        and text_rank > 3
        and visual_rank > 3
    ):
        score += 30
        reasons.append("both_branches_fail")
    if adaptive_rank is not None and adaptive_rank > 1:
        score += min(adaptive_rank - 1, 20) * 2
        reasons.append("adaptive_not_top1")
    return score, reasons


def main() -> None:
    args = parse_args()
    with project_path(args.queries).open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        queries = [
            row
            for row in csv.DictReader(handle)
            if row["split"] in {"validation", "test"}
        ]
    manifest = [
        json.loads(line)
        for line in project_path(args.manifest)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    filename_by_id = {
        row["item_id"]: row.get("source_file_name", row["item_id"])
        for row in manifest
    }
    text_report = json.loads(
        project_path(args.text_report).read_text(encoding="utf-8")
    )
    visual_report = json.loads(
        project_path(args.visual_report).read_text(encoding="utf-8")
    )
    fusion_report = json.loads(
        project_path(args.fusion_report).read_text(encoding="utf-8")
    )
    text_by_id = rank_lookup(text_report)
    visual_by_id = rank_lookup(visual_report)
    fixed_by_id = fusion_lookup(fusion_report, "fixed_fusion")
    adaptive_by_id = fusion_lookup(
        fusion_report, "quality_adaptive_fusion"
    )

    rows = []
    for query in queries:
        query_id = query["query_id"]
        text = text_by_id.get(query_id)
        visual = visual_by_id.get(query_id)
        fixed = fixed_by_id.get(query_id)
        adaptive = adaptive_by_id.get(query_id)
        text_rank = text.get("expected_rank") if text else None
        visual_rank = visual.get("expected_rank") if visual else None
        adaptive_rank = (
            adaptive.get("expected_rank") if adaptive else None
        )
        score, reasons = priority_score(
            query, text_rank, visual_rank, adaptive_rank
        )
        text_top1 = (
            text["top_3"][0]["item_id"]
            if text and text.get("top_3")
            else ""
        )
        visual_top1 = (
            visual["top_3"][0]["item_id"]
            if visual and visual.get("top_3")
            else ""
        )
        adaptive_top1 = (
            adaptive["top_3_item_ids"][0]
            if adaptive and adaptive.get("top_3_item_ids")
            else ""
        )
        rows.append(
            {
                **query,
                "diagnostic_text_rank": text_rank or "",
                "diagnostic_visual_rank": visual_rank or "",
                "diagnostic_fixed_rank": (
                    fixed.get("expected_rank") if fixed else ""
                ),
                "diagnostic_adaptive_rank": adaptive_rank or "",
                "text_top1_filename": filename_by_id.get(
                    text_top1, text_top1
                ),
                "visual_top1_filename": filename_by_id.get(
                    visual_top1, visual_top1
                ),
                "adaptive_top1_filename": filename_by_id.get(
                    adaptive_top1, adaptive_top1
                ),
                "diagnostic_priority_score": round(score, 3),
                "diagnostic_review_reasons": ",".join(reasons),
            }
        )
    source_queries_by_group: dict[str, list[str]] = {}
    for row in rows:
        group_key = (
            row.get("relevant_item_ids") or row["expected_item_id"]
        )
        source_queries_by_group.setdefault(group_key, []).append(
            row["query_id"]
        )
    for row in rows:
        group_key = (
            row.get("relevant_item_ids") or row["expected_item_id"]
        )
        row["relevant_group_size"] = len(
            [value for value in group_key.split(";") if value]
        )
        row["source_query_ids_for_group"] = ";".join(
            source_queries_by_group[group_key]
        )
    rows.sort(
        key=lambda row: (
            -float(row["diagnostic_priority_score"]),
            row["query_id"],
        )
    )
    review_rows = []
    seen_groups: set[str] = set()
    for row in rows:
        group_key = (
            row.get("relevant_item_ids") or row["expected_item_id"]
        )
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)
        review_rows.append(row)
        if len(review_rows) >= args.review_count:
            break

    output_path = project_path(args.output)
    review_path = project_path(args.review_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for path, output_rows in (
        (output_path, rows),
        (review_path, review_rows),
    ):
        with path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(output_rows[0])
            )
            writer.writeheader()
            writer.writerows(output_rows)

    summary = {
        "status": "diagnostic_unreviewed",
        "validation_test_candidates": len(rows),
        "review_queue_pages": len(review_rows),
        "review_queue_unique_relevance_groups": len(review_rows),
        "blank_queries": sum(not row["query"] for row in rows),
        "text_candidate_metrics": text_report["overall"],
        "visual_candidate_metrics": visual_report["overall"],
        "fixed_candidate_metrics": fusion_report["fixed_fusion"][
            "metrics"
        ],
        "adaptive_candidate_metrics": fusion_report[
            "quality_adaptive_fusion"
        ]["metrics"],
        "warning": (
            "These metrics diagnose query candidates and must not be "
            "reported as formal test results."
        ),
    }
    project_path(args.summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
