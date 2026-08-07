"""Generate and evaluate a license-clear 40-page minimal retrieval corpus."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bm25_retrieval import build_bm25_payload, score_bm25  # noqa: E402

PAGE_COUNT = 40
EXPECTED_CPU_METRICS = {
    "answerable_recall_at_1": 1.0,
    "answerable_recall_at_10": 1.0,
    "no_answer_rejection_accuracy": 1.0,
    "end_to_end_top1_accuracy": 1.0,
}
REFERENCE_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]+-\d{3}\b", re.IGNORECASE)
FAMILIES = (
    ("HARBOR", "crane", "harbor", "beside the warehouse"),
    ("DESERT", "pyramid", "desert", "behind the survey marker"),
    ("LAB", "microscope", "laboratory", "beside the safety cabinet"),
    ("FOREST", "bridge", "forest", "over the narrow river"),
    ("STATION", "train", "station", "under the platform clock"),
    ("ORBIT", "satellite", "space", "above the blue planet"),
    ("CLINIC", "scanner", "clinic", "next to the reception desk"),
    ("MUSEUM", "statue", "museum", "between two display cases"),
)
COLORS = (
    ("amber", "#e0a629"),
    ("blue", "#2878c8"),
    ("red", "#bf3b35"),
    ("green", "#3f8f55"),
    ("black", "#30343b"),
)


@dataclass(frozen=True)
class PageSpec:
    item_id: str
    reference: str
    title: str
    color_name: str
    color_hex: str
    object_name: str
    scene: str
    relation: str
    note: str

    @property
    def text(self) -> str:
        return (
            f"{self.title}. Reference {self.reference}. "
            f"A {self.color_name} {self.object_name} appears in the {self.scene}, "
            f"{self.relation}. Maintenance note: {self.note}."
        )


def page_specs() -> list[PageSpec]:
    specs: list[PageSpec] = []
    for family_index, (prefix, object_name, scene, relation) in enumerate(FAMILIES):
        for variant, (color_name, color_hex) in enumerate(COLORS, start=1):
            reference = f"{prefix}-{variant:03d}"
            specs.append(
                PageSpec(
                    item_id=f"synthetic_{family_index * 5 + variant:03d}",
                    reference=reference,
                    title=f"{scene.title()} inspection record {variant}",
                    color_name=color_name,
                    color_hex=color_hex,
                    object_name=object_name,
                    scene=scene,
                    relation=relation,
                    note=f"verify panel {family_index + 1} before cycle {variant}",
                )
            )
    if len(specs) != PAGE_COUNT:
        raise AssertionError(f"Expected {PAGE_COUNT} pages, got {len(specs)}")
    return specs


def query_rows(specs: list[PageSpec]) -> list[dict[str, Any]]:
    selected = (0, 2, 4, 5, 8, 11, 15, 19, 23, 27, 34, 39)
    rows = [
        {
            "query_id": f"answerable_{index + 1:02d}",
            "query": (
                f"Find the {spec.color_name} {spec.object_name} in the "
                f"{spec.scene} with reference {spec.reference}."
            ),
            "is_no_answer": False,
            "relevant_item_ids": [spec.item_id],
            "group_id": f"{spec.reference.split('-')[0].lower()}_family",
        }
        for index, spec in enumerate(specs[source_index] for source_index in selected)
    ]
    for index, source_index in enumerate((0, 6, 17, 28), start=1):
        spec = specs[source_index]
        absent_reference = f"{spec.reference.split('-')[0]}-999"
        rows.append(
            {
                "query_id": f"no_answer_{index:02d}",
                "query": (
                    f"Find the {spec.color_name} {spec.object_name} in the "
                    f"{spec.scene} with reference {absent_reference}."
                ),
                "is_no_answer": True,
                "relevant_item_ids": [],
                "group_id": f"{spec.reference.split('-')[0].lower()}_family",
            }
        )
    return rows


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _render_page(spec: PageSpec, path: Path) -> None:
    image = Image.new("RGB", (720, 960), "#f6f2e9")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=18)
    draw.rounded_rectangle(
        (42, 42, 678, 918),
        radius=24,
        fill="white",
        outline="#333333",
    )
    draw.text((78, 82), spec.title, fill="#20242a", font=font)
    draw.text((78, 132), f"REFERENCE: {spec.reference}", fill="#20242a", font=small)
    draw.rectangle((78, 200, 642, 570), fill="#e9edf2", outline="#66707a", width=3)
    draw.ellipse((170, 285, 390, 505), fill=spec.color_hex, outline="#20242a", width=4)
    draw.polygon(
        ((420, 475), (535, 255), (625, 475)),
        fill="#d7ba83",
        outline="#20242a",
    )
    draw.line((360, 395, 465, 395), fill="#20242a", width=6)
    draw.polygon(((465, 395), (440, 380), (440, 410)), fill="#20242a")
    lines = (
        f"OBJECT: {spec.color_name.upper()} {spec.object_name.upper()}",
        f"SCENE: {spec.scene.upper()}",
        f"RELATION: {spec.relation.upper()}",
        f"NOTE: {spec.note.upper()}",
        "SELF-GENERATED SYNTHETIC PAGE - CC0-1.0",
    )
    for index, line in enumerate(lines):
        draw.text((78, 630 + index * 46), line, fill="#20242a", font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def ingest(output_dir: Path, *, build_visual: bool = False) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    specs = page_specs()
    manifest: list[dict[str, Any]] = []
    corpus: list[dict[str, Any]] = []
    for spec in specs:
        image_path = output_dir / "images" / f"{spec.item_id}.png"
        _render_page(spec, image_path)
        manifest.append(
            {
                **asdict(spec),
                "display_name_zh": spec.title,
                "category": "synthetic_repro",
                "source_path": str(image_path),
                "license": "CC0-1.0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "origin": (
                    "deterministically self-generated by "
                    "scripts/minimal_repro.py"
                ),
                "ai_generated": False,
            }
        )
        corpus.append(
            {
                "item_id": spec.item_id,
                "chunk_id": f"{spec.item_id}_text",
                "text": spec.text,
            }
        )
    queries = query_rows(specs)
    write_jsonl_atomic(output_dir / "manifest.jsonl", manifest)
    write_jsonl_atomic(output_dir / "corpus.jsonl", corpus)
    write_jsonl_atomic(output_dir / "queries.jsonl", queries)

    bm25 = build_bm25_payload([row["text"] for row in corpus])
    index_dir = output_dir / "bm25_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(index_dir / "index.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(bm25, handle, ensure_ascii=False, separators=(",", ":"))
    write_jsonl_atomic(index_dir / "metadata.jsonl", corpus)
    expected = {
        "schema_version": 1,
        "backend": "cpu_bm25",
        "page_count": PAGE_COUNT,
        "query_count": len(queries),
        "answerable_count": sum(not row["is_no_answer"] for row in queries),
        "no_answer_count": sum(row["is_no_answer"] for row in queries),
        "metrics": EXPECTED_CPU_METRICS,
    }
    write_json_atomic(output_dir / "expected_results.json", expected)
    visual_index_built = False
    if build_visual:
        visual_python = PROJECT_ROOT / ".venv-vl/Scripts/python.exe"
        if not visual_python.is_file():
            raise FileNotFoundError(f"Visual Python runtime not found: {visual_python}")
        subprocess.run(
            [
                str(visual_python),
                str(PROJECT_ROOT / "scripts/build_visual_index.py"),
                "--manifest",
                str(output_dir / "manifest.jsonl"),
                "--output",
                str(output_dir / "visual_index"),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        visual_index_built = True
    receipt = {
        **expected,
        "output_dir": str(output_dir),
        "visual_index_built": visual_index_built,
    }
    write_json_atomic(output_dir / "ingest_receipt.json", receipt)
    return receipt


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _cpu_scores(output_dir: Path, queries: list[dict[str, Any]]) -> list[list[float]]:
    with gzip.open(
        output_dir / "bm25_index/index.json.gz", "rt", encoding="utf-8"
    ) as handle:
        payload = json.load(handle)
    return [score_bm25(str(row["query"]), payload) for row in queries]


def _visual_scores(
    output_dir: Path, queries: list[dict[str, Any]]
) -> list[list[float]]:
    import numpy as np
    import torch

    official_repo = PROJECT_ROOT / "third_party/Qwen3-VL-Embedding"
    if str(official_repo) not in sys.path:
        sys.path.insert(0, str(official_repo))
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    if not torch.cuda.is_available():
        raise RuntimeError("Visual reproduction requires a CUDA GPU")
    index_dir = output_dir / "visual_index"
    vectors = np.load(index_dir / "embeddings.npy").astype(np.float32)
    model = Qwen3VLEmbedder(
        model_name_or_path=str(PROJECT_ROOT / "models/qwen3-vl-embedding-2b"),
        max_length=512,
        min_pixels=32 * 32 * 4,
        max_pixels=512 * 512,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    encoded = model.process(
        [
            {
                "text": row["query"],
                "instruction": "Retrieve the synthetic page matching every condition.",
            }
            for row in queries
        ]
    )
    query_vectors = encoded.detach().float().cpu().numpy().astype(np.float32)
    return (query_vectors @ vectors.T).tolist()


def evaluate(output_dir: Path, *, backend: str = "cpu") -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest = _read_jsonl(output_dir / "manifest.jsonl")
    corpus = _read_jsonl(output_dir / "corpus.jsonl")
    queries = _read_jsonl(output_dir / "queries.jsonl")
    item_ids = [str(row["item_id"]) for row in manifest]
    corpus_text = "\n".join(str(row["text"]) for row in corpus).casefold()
    scores_by_query = (
        _cpu_scores(output_dir, queries)
        if backend == "cpu"
        else _visual_scores(output_dir, queries)
    )
    details: list[dict[str, Any]] = []
    for query, scores in zip(queries, scores_by_query, strict=True):
        ranking = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], item_ids[index]),
        )
        ranked_ids = [item_ids[index] for index in ranking]
        references = REFERENCE_PATTERN.findall(str(query["query"]))
        missing_reference = any(
            reference.casefold() not in corpus_text for reference in references
        )
        accepted = bool(scores[ranking[0]] > 0.0) and not missing_reference
        relevant = set(query["relevant_item_ids"])
        relevant_rank = next(
            (
                index
                for index, item_id in enumerate(ranked_ids, start=1)
                if item_id in relevant
            ),
            None,
        )
        details.append(
            {
                "query_id": query["query_id"],
                "is_no_answer": bool(query["is_no_answer"]),
                "accepted": accepted,
                "missing_reference": missing_reference,
                "top_item_id": ranked_ids[0],
                "top_score": round(float(scores[ranking[0]]), 8),
                "relevant_rank": relevant_rank,
            }
        )
    answerable = [row for row in details if not row["is_no_answer"]]
    no_answer = [row for row in details if row["is_no_answer"]]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    metrics = {
        "answerable_recall_at_1": mean(
            [float(row["relevant_rank"] == 1) for row in answerable]
        ),
        "answerable_recall_at_10": mean(
            [
                float(row["relevant_rank"] is not None and row["relevant_rank"] <= 10)
                for row in answerable
            ]
        ),
        "no_answer_rejection_accuracy": mean(
            [float(not row["accepted"]) for row in no_answer]
        ),
        "end_to_end_top1_accuracy": mean(
            [
                float(not row["accepted"])
                if row["is_no_answer"]
                else float(row["accepted"] and row["relevant_rank"] == 1)
                for row in details
            ]
        ),
    }
    report = {
        "schema_version": 1,
        "backend": "cpu_bm25" if backend == "cpu" else "qwen3_vl_visual",
        "page_count": len(manifest),
        "query_count": len(queries),
        "answerable_count": len(answerable),
        "no_answer_count": len(no_answer),
        "metrics": metrics,
        "matches_expected_cpu_result": (
            metrics == EXPECTED_CPU_METRICS if backend == "cpu" else None
        ),
        "details": details,
    }
    write_json_atomic(output_dir / f"evaluation_{backend}.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("ingest", "evaluate"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/minimal_repro"),
    )
    parser.add_argument("--with-visual-index", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "visual"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    payload = (
        ingest(output_dir, build_visual=args.with_visual_index)
        if args.command == "ingest"
        else evaluate(output_dir, backend=args.backend)
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
