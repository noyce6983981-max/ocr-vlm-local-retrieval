"""Optional V19 ColQwen2 Top-10 retrieval with complete condition evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.gating.candidate_verification import (  # noqa: E402
    load_ocr_lines,
)
from ocr_vlm_retrieval.gating.literal_evidence_v19_2 import (  # noqa: E402
    select_v19_2_literal_candidate,
    v19_2_override_eligibility,
)
from ocr_vlm_retrieval.gating.ocr_literals_v19_2 import (  # noqa: E402
    complete_explicit_evidence_item_ids,
    extract_v19_2_literal_groups,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from ocr_vlm_retrieval.runtime.late_interaction import (  # noqa: E402
    exclusive_process_lock,
    load_embedding,
)
from ocr_vlm_retrieval.runtime.v19_condition_runtime import (  # noqa: E402
    alias_fallback_payload,
    build_condition_ranking,
)
from scripts import live_search  # noqa: E402
from scripts.evaluate_v19_colqwen2_development import (  # noqa: E402
    maxsim_score_matrix,
)

DEFAULT_CONFIG = ROOT / "config/v19_runtime.json"
DEFAULT_MODEL = ROOT / "models/colqwen2-v1.0-hf"
DEFAULT_INDEX = ROOT / "outputs/user_library/colqwen2_v1_index"
TEXT_PYTHON = ROOT / ".venv/Scripts/python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--library-dir", type=Path, default=live_search.DEFAULT_LIBRARY_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _manifest(library_dir: Path) -> list[dict[str, Any]]:
    return live_search.combined_manifest(library_dir)


def _ocr_lines(
    library_dir: Path,
    item_id: str,
    *,
    minimum_confidence: float,
) -> list[str]:
    """Load human-corrected OCR first, then confidence-filtered base OCR."""

    override_path = library_dir / "ocr/overrides" / f"{item_id}.json"
    if override_path.is_file():
        payload = json.loads(override_path.read_text(encoding="utf-8-sig"))
        texts = payload.get("rec_texts", [])
        if not isinstance(texts, list):
            raise ValueError(f"Invalid OCR override: {override_path}")
        return [str(text).strip() for text in texts if str(text).strip()]
    return load_ocr_lines(
        library_dir / "ocr/json" / f"{item_id}.json",
        minimum_confidence=minimum_confidence,
    )


def _fallback(
    args: argparse.Namespace,
    *,
    query: str,
    reason: str,
    config: dict[str, Any],
) -> int:
    temporary = args.output.with_suffix(".v18_fallback.json")
    command = [
        str(TEXT_PYTHON),
        str(ROOT / "scripts/live_search.py"),
        query,
        "--library-dir",
        str(args.library_dir),
        "--output",
        str(temporary),
        "--method",
        str(config["fallback_method"]),
    ]
    if args.force:
        command.append("--force")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout)[-1800:]
        raise RuntimeError(detail)
    payload = json.loads(temporary.read_text(encoding="utf-8"))
    result = alias_fallback_payload(
        payload, reason=reason, fallback_method=str(config["fallback_method"])
    )
    result["search_policy_version"] = 19
    result["retrieval_config_revision"] = hashlib.sha256(
        args.config.read_bytes()
    ).hexdigest()[:12]
    write_json_atomic(args.output, result)
    temporary.unlink(missing_ok=True)
    return 0


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    query = " ".join(args.query.split())
    if not query:
        raise ValueError("query cannot be empty")
    args.library_dir = args.library_dir.resolve()
    args.output = args.output.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    eligibility = v19_2_override_eligibility(query)
    if not eligibility["eligible"]:
        return _fallback(
            args,
            query=query,
            reason=str(eligibility["reason"]),
            config=config,
        )
    revision = live_search.library_revision(args.library_dir)
    config_revision = hashlib.sha256(args.config.read_bytes()).hexdigest()[:12]
    if args.output.is_file() and not args.force:
        cached = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            cached.get("query") == query
            and cached.get("requested_method") == "v19_condition"
            and cached.get("library_revision") == revision
            and cached.get("retrieval_config_revision") == config_revision
        ):
            return 0
    manifest = _manifest(args.library_dir)
    manifest_by_id = {str(row["item_id"]): row for row in manifest}
    groups = extract_v19_2_literal_groups(query)
    explicit_contract = (
        len(groups) >= 2
        and all(group.source == "explicit_required" for group in groups)
    )
    if explicit_contract:
        all_lines_by_item = {
            item_id: _ocr_lines(
                args.library_dir,
                item_id,
                minimum_confidence=float(config["ocr_min_confidence"]),
            )
            for item_id in manifest_by_id
        }
        complete_evidence_ids = complete_explicit_evidence_item_ids(
            groups, all_lines_by_item
        )
        candidates = complete_evidence_ids[: int(config["top_k"])]
        decision = select_v19_2_literal_candidate(
            query,
            candidates,
            {item_id: all_lines_by_item[item_id] for item_id in candidates},
            fuzzy_threshold=float(config["ocr_fuzzy_threshold"]),
        )
        ranking = build_condition_ranking(
            manifest_by_id,
            candidates,
            {item_id: 1.0 for item_id in candidates},
            decision["candidate_evidence"],
            selected_item_id=decision["selected_item_id"],
        )
        elapsed = round(time.perf_counter() - started, 3)
        accepted = bool(decision["accepted"])
        payload = {
            "query": query,
            "search_policy_version": 19,
            "retrieval_config_revision": config_revision,
            "requested_method": "v19_condition",
            "query_mode": "v19_condition_evidence",
            "retrieval_route": "v19_condition_evidence",
            "library_dir": args.library_dir.relative_to(ROOT).as_posix(),
            "library_revision": revision,
            "cache_key": live_search.query_key(query, revision),
            "executed_branches": {
                "text": False,
                "bm25": False,
                "visual": False,
                "reranker": False,
                "color": False,
                "condition_evidence": True,
            },
            "timings": {
                "text_seconds": 0.0,
                "bm25_seconds": 0.0,
                "visual_seconds": 0.0,
                "reranker_seconds": 0.0,
                "condition_evidence_seconds": elapsed,
                "total_seconds": elapsed,
            },
            "rankings": {"v19_condition": ranking},
            "low_confidence_rankings": {"v19_condition": ranking},
            "acceptance": {
                "v19_condition": {
                    "accepted": accepted,
                    "query_mode": "v19_condition_evidence",
                    "reason": (
                        "全库OCR证据中存在同页满足全部显式必要条件的页面。"
                        if accepted
                        else "全库OCR证据中没有页面同时满足全部显式必要条件。"
                    ),
                    "signal_name": "full_corpus_complete_condition_match",
                    "signal": 1 if accepted else 0,
                    "threshold": 1,
                    "v19_condition_intervened": True,
                }
            },
            "v19_condition_runtime": {
                "status": "full_corpus_exact_condition_evidence",
                "intervened": True,
                "method": config["method"],
                "top_k": int(config["top_k"]),
                "complete_evidence_item_ids": complete_evidence_ids,
                "complete_evidence_promotion_applied": bool(
                    complete_evidence_ids
                ),
                "ocr_fuzzy_threshold": config["ocr_fuzzy_threshold"],
                "eligibility": eligibility,
                "decision": decision,
            },
        }
        write_json_atomic(args.output, payload)
        return 0
    if args.library_dir != live_search.DEFAULT_LIBRARY_DIR.resolve():
        return _fallback(
            args,
            query=query,
            reason="当前资料库没有独立的ColQwen2多向量索引",
            config=config,
        )
    required_paths = (args.model / "config.json", args.index_dir / "index_receipt.json")
    if not all(path.is_file() for path in required_paths):
        return _fallback(
            args,
            query=query,
            reason="本机未安装V19模型或完整索引",
            config=config,
        )
    receipt = json.loads(
        (args.index_dir / "index_receipt.json").read_text(encoding="utf-8")
    )
    item_ids = [str(value) for value in receipt.get("indexed_item_ids", [])]
    if (
        receipt.get("status") != "complete"
        or receipt.get("failed_item_count") != 0
        or len(item_ids) != len(manifest_by_id)
        or set(item_ids) != manifest_by_id.keys()
    ):
        return _fallback(
            args,
            query=query,
            reason="V19索引与当前资料库版本不一致",
            config=config,
        )
    shard_by_id = {
        path.stem: path for path in (args.index_dir / "shards").glob("*.npy")
    }
    if any(item_id not in shard_by_id for item_id in item_ids):
        raise ValueError("V19 index is missing page shards")

    import torch
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor

    if not torch.cuda.is_available():
        return _fallback(
            args,
            query=query,
            reason="当前环境没有可用CUDA设备",
            config=config,
        )
    model = ColQwen2ForRetrieval.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()  # type: ignore[no-untyped-call]
    processor = ColQwen2Processor.from_pretrained(args.model, use_fast=True)
    inputs = processor(text=[query]).to(model.device)
    with torch.inference_mode():
        query_embedding = model(**inputs).embeddings[0]
    scores = torch.empty(len(item_ids), dtype=torch.float32)
    batch_size = 16
    for start in range(0, len(item_ids), batch_size):
        batch_ids = item_ids[start : start + batch_size]
        passages = [
            torch.from_numpy(load_embedding(shard_by_id[item_id], expected_dim=128)).to(
                model.device, dtype=torch.bfloat16
            )
            for item_id in batch_ids
        ]
        with torch.inference_mode():
            block = maxsim_score_matrix([query_embedding], passages)[0].cpu()
        scores[start : start + len(batch_ids)] = block
        del passages
    all_lines_by_item = {
        item_id: _ocr_lines(
            args.library_dir,
            item_id,
            minimum_confidence=float(config["ocr_min_confidence"]),
        )
        for item_id in item_ids
    }
    complete_evidence_ids = complete_explicit_evidence_item_ids(
        extract_v19_2_literal_groups(query), all_lines_by_item
    )
    top_k = int(config["top_k"])
    order = torch.argsort(scores, descending=True).tolist()
    colqwen2_ranking = [item_ids[index] for index in order]
    complete_evidence_set = set(complete_evidence_ids)
    candidates = [
        *complete_evidence_ids,
        *(
            item_id
            for item_id in colqwen2_ranking
            if item_id not in complete_evidence_set
        ),
    ][:top_k]
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    scores_by_id = {
        item_id: float(scores[item_index[item_id]]) for item_id in candidates
    }
    lines_by_item = {item_id: all_lines_by_item[item_id] for item_id in candidates}
    decision = select_v19_2_literal_candidate(
        query,
        candidates,
        lines_by_item,
        fuzzy_threshold=float(config["ocr_fuzzy_threshold"]),
    )
    selected = decision["selected_item_id"]
    ranking = build_condition_ranking(
        manifest_by_id,
        candidates,
        scores_by_id,
        decision["candidate_evidence"],
        selected_item_id=selected,
    )
    acceptance = {
        "accepted": bool(decision["accepted"]),
        "query_mode": "v19_condition_evidence",
        "reason": (
            "Top-10候选中存在同页满足全部必要条件的页面。"
            if decision["accepted"]
            else "Top-10候选均未在同一页满足全部必要条件，系统保守拒答。"
        ),
        "signal_name": "complete_necessary_condition_match",
        "signal": 1 if decision["accepted"] else 0,
        "threshold": 1,
        "v19_condition_intervened": True,
    }
    payload = {
        "query": query,
        "search_policy_version": 19,
        "retrieval_config_revision": config_revision,
        "requested_method": "v19_condition",
        "query_mode": "v19_condition_evidence",
        "retrieval_route": "v19_condition_evidence",
        "library_dir": args.library_dir.relative_to(ROOT).as_posix(),
        "library_revision": revision,
        "cache_key": live_search.query_key(query, revision),
        "executed_branches": {
            "text": False,
            "bm25": False,
            "visual": True,
            "reranker": False,
            "color": False,
            "condition_evidence": True,
        },
        "timings": {
            "text_seconds": 0.0,
            "bm25_seconds": 0.0,
            "visual_seconds": round(time.perf_counter() - started, 3),
            "reranker_seconds": 0.0,
            "total_seconds": round(time.perf_counter() - started, 3),
        },
        "rankings": {"v19_condition": ranking},
        "low_confidence_rankings": {"v19_condition": ranking},
        "acceptance": {"v19_condition": acceptance},
        "v19_condition_runtime": {
            "status": "complete_condition_evidence",
            "intervened": True,
            "method": config["method"],
            "top_k": top_k,
            "complete_evidence_item_ids": complete_evidence_ids,
            "complete_evidence_promotion_applied": bool(complete_evidence_ids),
            "ocr_fuzzy_threshold": config["ocr_fuzzy_threshold"],
            "eligibility": eligibility,
            "decision": decision,
        },
    }
    write_json_atomic(args.output, payload)
    return 0


def main() -> int:
    args = parse_args()
    lock_path = args.index_dir / ".v19_live_search.lock"
    with exclusive_process_lock(lock_path):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
