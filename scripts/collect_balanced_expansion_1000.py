"""Collect 1,000 real images to balance the first research library to 1,500.

The raw images stay under data/incoming/ and are ignored by Git. A tracked
source manifest records provenance, license, category, and audit metadata.
The checkpoint makes the collection resumable without redownloading accepted
images.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_public_dataset_200 import difference_hash
from scripts.collect_public_dataset_200 import (
    FUNSD_URL,
    normalized_jpeg_bytes,
)


CACHE_DIR = PROJECT_ROOT / "work/dataset_1500_cache"
OUTPUT_DIR = PROJECT_ROOT / "data/incoming/expansion_1000_balanced"
CHECKPOINT_PATH = CACHE_DIR / "balanced_1000_checkpoint.jsonl"
SOURCE_CSV = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_balanced_expansion_1000_sources.csv"
)
SUMMARY_PATH = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_balanced_expansion_1000_summary.json"
)
EXISTING_DIRS = (
    PROJECT_ROOT / "data/incoming/200页全部放这里",
    PROJECT_ROOT / "data/incoming/expansion_300_real",
)
EXISTING_SOURCE_CSVS = (
    PROJECT_ROOT / "data/evaluation/public_dataset_200_sources.csv",
    PROJECT_ROOT / "data/evaluation/public_dataset_expansion_300_sources.csv",
)
RVL_PARQUET = (
    CACHE_DIR / "rvl_cdip/train-00000-of-00002.parquet"
)
RICO_PARQUET = (
    CACHE_DIR / "rico_sca/train-00000-of-00036.parquet"
)
TEXTVQA_JSON = CACHE_DIR / "open_images/TextVQA_0.5.1_train.json"
TEXTVQA_IMAGE_CACHE = CACHE_DIR / "textvqa_images"
CALTECH_ZIP = CACHE_DIR / "caltech101/caltech-101.zip"
FUNSD_ZIP = PROJECT_ROOT / "work/dataset_200_cache/funsd.zip"

TARGETS = {
    "clear_document": 170,
    "complex_academic": 195,
    "table_form_ticket": 46,
    "ppt_poster_slide": 163,
    "software_web_code": 130,
    "scene_text": 162,
    "degraded_document": 4,
    "natural_no_text": 130,
}

RVL_LABELS = {
    0: "letter",
    1: "form",
    2: "email",
    3: "handwritten",
    4: "advertisement",
    5: "scientific_report",
    6: "scientific_publication",
    7: "specification",
    8: "file_folder",
    9: "news_article",
    10: "budget",
    11: "invoice",
    12: "presentation",
    13: "questionnaire",
    14: "resume",
    15: "memo",
}

RVL_PLAN = {
    "clear_document": {
        0: 34,
        2: 34,
        9: 34,
        14: 34,
        15: 34,
    },
    "complex_academic": {5: 65, 6: 65, 7: 65},
    "table_form_ticket": {1: 12, 10: 12, 11: 11, 13: 11},
    "ppt_poster_slide": {4: 81, 12: 82},
}

CALTECH_CATEGORIES = (
    "Faces",
    "Faces_easy",
    "minaret",
    "pagoda",
    "pyramid",
    "airplanes",
    "helicopter",
    "Motorbikes",
    "car_side",
    "ferry",
    "ketch",
    "schooner",
    "beaver",
    "butterfly",
    "cougar_body",
    "crab",
    "dolphin",
    "elephant",
    "flamingo",
    "ibis",
    "kangaroo",
    "llama",
    "panda",
    "rhino",
    "sea_horse",
    "wild_cat",
    "bonsai",
    "joshua_tree",
    "lotus",
    "sunflower",
    "water_lilly",
    "camera",
    "cellphone",
    "chair",
    "laptop",
    "umbrella",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def jpeg_dhash(payload: bytes) -> int:
    with Image.open(io.BytesIO(payload)) as image:
        grayscale = ImageOps.grayscale(
            ImageOps.exif_transpose(image)
        ).resize((9, 8))
    if hasattr(grayscale, "get_flattened_data"):
        pixels = list(grayscale.get_flattened_data())
    else:
        pixels = list(grayscale.get_flattened_data())
    value = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            value = (value << 1) | int(left > right)
    return value


def image_payload(value: Any) -> bytes | None:
    if isinstance(value, dict):
        payload = value.get("bytes")
        if isinstance(payload, bytes):
            return payload
    if isinstance(value, bytes):
        return value
    return None


class Curator:
    def __init__(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        self.hashes: set[str] = set()
        self.dhashes: list[int] = []
        self.counts: Counter[str] = Counter()
        self.source_files: set[tuple[str, str]] = set()

        for directory in EXISTING_DIRS:
            for path in sorted(directory.glob("*.jpg")):
                self.hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
                self.dhashes.append(difference_hash(path))

        if CHECKPOINT_PATH.is_file():
            for line in CHECKPOINT_PATH.read_text(
                encoding="utf-8"
            ).splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                path = OUTPUT_DIR / row["filename"]
                if not path.is_file():
                    raise RuntimeError(
                        f"Checkpoint image is missing: {path}"
                    )
                payload = path.read_bytes()
                self.rows.append(row)
                self.hashes.add(hashlib.sha256(payload).hexdigest())
                self.dhashes.append(jpeg_dhash(payload))
                self.counts[row["category"]] += 1
                self.source_files.add(
                    (row["source_name"], row["source_file"])
                )
        elif any(OUTPUT_DIR.iterdir()):
            raise RuntimeError(
                f"Non-empty output has no checkpoint: {OUTPUT_DIR}"
            )

    def remaining(self, category: str) -> int:
        return max(0, TARGETS[category] - self.counts[category])

    def add(
        self,
        payload: bytes,
        *,
        category: str,
        source_name: str,
        source_url: str,
        source_file: str,
        license_name: str,
        license_url: str,
        author: str,
        origin_type: str,
        language: str,
        has_text: bool,
        quality_tags: str,
        privacy_review_required: bool = False,
    ) -> bool:
        if self.remaining(category) <= 0:
            return False
        source_key = (source_name, source_file)
        if source_key in self.source_files:
            return False
        try:
            jpeg, width, height = normalized_jpeg_bytes(payload)
        except Exception:
            return False
        digest = hashlib.sha256(jpeg).hexdigest()
        dhash = jpeg_dhash(jpeg)
        if digest in self.hashes:
            return False
        if any((dhash ^ previous).bit_count() <= 5 for previous in self.dhashes):
            return False

        index = self.counts[category] + 1
        filename = f"balanced_{category}_{index:03d}.jpg"
        target = OUTPUT_DIR / filename
        target.write_bytes(jpeg)
        row = {
            "item_id": digest[:16],
            "filename": filename,
            "category": category,
            "language": language,
            "has_text": str(has_text).lower(),
            "quality_tags": quality_tags,
            "source_name": source_name,
            "source_url": source_url,
            "source_file": source_file,
            "license": license_name,
            "license_url": license_url,
            "author": author,
            "origin_type": origin_type,
            "ai_generated": "false",
            "declared_ai_generated_percent": "0",
            "hard_negative_group": (
                f"balanced_{category}_{(index - 1) // 2 + 1:03d}"
                if index <= 20
                else ""
            ),
            "width": width,
            "height": height,
            "review_status": "待人工审核",
            "dataset_name": "public_dataset_1500_balanced",
            "perceptual_group": "",
            "privacy_review_required": str(
                privacy_review_required
            ).lower(),
            "category_review_required": "true",
        }
        with CHECKPOINT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows.append(row)
        self.hashes.add(digest)
        self.dhashes.append(dhash)
        self.counts[category] += 1
        self.source_files.add(source_key)
        return True


def collect_rvl(curator: Curator) -> None:
    if all(curator.remaining(category) == 0 for category in RVL_PLAN):
        return
    table = pq.read_table(RVL_PARQUET, columns=["image", "label"])
    collected: dict[tuple[str, int], int] = defaultdict(int)
    for index in range(table.num_rows):
        label = table["label"][index].as_py()
        for category, label_targets in RVL_PLAN.items():
            required = label_targets.get(label, 0)
            if collected[(category, label)] >= required:
                continue
            source_file = f"train_shard_00_row_{index:04d}"
            if (
                "RVL-CDIP balanced subset",
                source_file,
            ) in curator.source_files:
                collected[(category, label)] += 1
                break
            payload = image_payload(table["image"][index].as_py())
            if payload is None:
                break
            if curator.add(
                payload,
                category=category,
                source_name="RVL-CDIP balanced subset",
                source_url=(
                    "https://huggingface.co/datasets/"
                    "hf-tuner/rvl-cdip-document-classification"
                ),
                source_file=source_file,
                license_name=(
                    "Other — IIT-CDIP/Industry Documents Library terms"
                ),
                license_url=(
                    "https://www.industrydocuments.ucsf.edu/"
                    "help/copyright/"
                ),
                author="RVL-CDIP / IIT-CDIP contributors",
                origin_type=f"real_scanned_{RVL_LABELS[label]}",
                language="en",
                has_text=True,
                quality_tags=f"real_scan,rvl_cdip,{RVL_LABELS[label]}",
                privacy_review_required=label == 14,
            ):
                collected[(category, label)] += 1
            break
    for category, label_targets in RVL_PLAN.items():
        expected = sum(label_targets.values())
        actual = sum(
            count
            for (current_category, _), count in collected.items()
            if current_category == category
        )
        if curator.remaining(category) and actual < expected:
            raise RuntimeError(
                f"RVL {category}: accepted {actual}/{expected}"
            )
        print(
            f"RVL {category}: "
            f"{TARGETS[category] - curator.remaining(category)}/"
            f"{TARGETS[category]}"
        )


def collect_rico(curator: Curator) -> None:
    category = "software_web_code"
    if curator.remaining(category) == 0:
        return
    table = pq.read_table(
        RICO_PARQUET,
        columns=[
            "screenId",
            "file_name",
            "play_store_name",
            "category",
            "image",
        ],
    )
    seen_screens: set[int] = set()
    for index in range(table.num_rows):
        screen_id = table["screenId"][index].as_py()
        if screen_id in seen_screens:
            continue
        seen_screens.add(screen_id)
        source_file = table["file_name"][index].as_py()
        payload = image_payload(table["image"][index].as_py())
        if payload is None:
            continue
        app_name = table["play_store_name"][index].as_py() or "unknown_app"
        app_category = table["category"][index].as_py() or "unknown"
        curator.add(
            payload,
            category=category,
            source_name="RICO-SCA",
            source_url="https://huggingface.co/datasets/bevaya/RICO-SCA",
            source_file=source_file,
            license_name="Apache-2.0",
            license_url="https://www.apache.org/licenses/LICENSE-2.0",
            author="RICO authors and RICO-SCA maintainers",
            origin_type="real_mobile_app_screenshot",
            language="mixed",
            has_text=True,
            quality_tags=(
                f"real_ui_screenshot,{app_category},"
                f"app={app_name}"
            ),
            privacy_review_required=True,
        )
        if curator.remaining(category) == 0:
            break
    if curator.remaining(category):
        raise RuntimeError(
            f"RICO missing {curator.remaining(category)} images"
        )
    print(f"RICO {category}: {TARGETS[category]}/{TARGETS[category]}")


def download_with_curl(
    urls: Iterable[str],
    destination: Path,
) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for url in urls:
        completed = subprocess.run(
            [
                "curl.exe",
                "-sS",
                "-L",
                "--fail",
                "--retry",
                "1",
                "--retry-all-errors",
                "--connect-timeout",
                "10",
                "--max-time",
                "12",
                "-H",
                "User-Agent: OCR-VLM-research-curator/1.0",
                "--output",
                str(temporary),
                url,
            ],
            check=False,
            capture_output=True,
            timeout=18,
        )
        if completed.returncode == 0 and temporary.stat().st_size > 0:
            temporary.replace(destination)
            return True
    if temporary.exists():
        temporary.unlink()
    return False


def prefetch_textvqa(
    entries: list[tuple[str, dict[str, Any]]],
) -> None:
    pending = [
        (image_id, row)
        for image_id, row in entries
        if not (TEXTVQA_IMAGE_CACHE / f"{image_id}.jpg").is_file()
    ]
    if not pending:
        return

    def download(entry: tuple[str, dict[str, Any]]) -> bool:
        image_id, row = entry
        try:
            return download_with_curl(
                (
                    row.get("flickr_300k_url", ""),
                    row.get("flickr_original_url", ""),
                ),
                TEXTVQA_IMAGE_CACHE / f"{image_id}.jpg",
            )
        except (OSError, subprocess.TimeoutExpired):
            return False

    completed = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download, entry) for entry in pending]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 40 == 0 or completed == len(futures):
                print(
                    f"TextVQA prefetch: {completed}/{len(futures)}"
                )


def collect_textvqa(curator: Curator) -> None:
    category = "scene_text"
    if curator.remaining(category) == 0:
        return
    data = json.loads(TEXTVQA_JSON.read_text(encoding="utf-8"))["data"]
    unique: dict[str, dict[str, Any]] = {}
    for row in data:
        unique.setdefault(row["image_id"], row)

    entries = sorted(unique.items())
    for start in range(0, len(entries), 240):
        if curator.remaining(category) == 0:
            break
        batch = entries[start : start + 240]
        prefetch_textvqa(batch)
        for image_id, row in batch:
            if curator.remaining(category) == 0:
                break
            source_file = f"{image_id}.jpg"
            cached = TEXTVQA_IMAGE_CACHE / source_file
            if not cached.is_file():
                continue
            before = curator.remaining(category)
            curator.add(
                cached.read_bytes(),
                category=category,
                source_name="TextVQA real scene images",
                source_url="https://textvqa.org/",
                source_file=source_file,
                license_name="CC BY 4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
                author="TextVQA/Open Images contributors",
                origin_type="real_scene_text_photo",
                language="mixed",
                has_text=True,
                quality_tags=(
                    "real_scene_text,textvqa,"
                    f"question={row.get('question', '')[:80]}"
                ),
                privacy_review_required=True,
            )
            completed = TARGETS[category] - curator.remaining(category)
            if (
                curator.remaining(category) != before
                and (completed % 20 == 0 or curator.remaining(category) == 0)
            ):
                print(
                    f"TextVQA {category}: "
                    f"{completed}/{TARGETS[category]}"
                )
    if curator.remaining(category):
        raise RuntimeError(
            f"TextVQA missing {curator.remaining(category)} images"
        )
    print(f"TextVQA {category}: {TARGETS[category]}/{TARGETS[category]}")


def caltech_members() -> dict[str, list[tuple[str, bytes]]]:
    with zipfile.ZipFile(CALTECH_ZIP) as outer:
        archive_bytes = outer.read(
            "caltech-101/101_ObjectCategories.tar.gz"
        )
    members: dict[str, list[tuple[str, bytes]]] = defaultdict(list)
    with tarfile.open(
        fileobj=io.BytesIO(archive_bytes), mode="r:gz"
    ) as archive:
        for info in archive.getmembers():
            if not info.isfile():
                continue
            parts = info.name.split("/")
            if len(parts) != 3 or parts[1] not in CALTECH_CATEGORIES:
                continue
            extracted = archive.extractfile(info)
            if extracted is not None:
                members[parts[1]].append((info.name, extracted.read()))
    return members


def collect_caltech(curator: Curator) -> None:
    category = "natural_no_text"
    if curator.remaining(category) == 0:
        return
    members = caltech_members()
    cursors: Counter[str] = Counter()
    while curator.remaining(category):
        progress = False
        for caltech_category in CALTECH_CATEGORIES:
            candidates = members.get(caltech_category, [])
            cursor = cursors[caltech_category]
            if cursor >= len(candidates):
                continue
            source_file, payload = candidates[cursor]
            cursors[caltech_category] += 1
            if curator.add(
                payload,
                category=category,
                source_name="Caltech-101",
                source_url="https://data.caltech.edu/records/20086",
                source_file=source_file,
                license_name="CC BY 4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
                author="Fei-Fei Li, Rob Fergus, Pietro Perona et al.",
                origin_type="real_object_or_natural_photo",
                language="none",
                has_text=False,
                quality_tags=(
                    f"real_photo,caltech101,{caltech_category}"
                ),
                privacy_review_required=caltech_category.startswith("Faces"),
            ):
                progress = True
            if curator.remaining(category) == 0:
                break
        if not progress:
            raise RuntimeError(
                f"Caltech-101 missing {curator.remaining(category)} images"
            )
    print(f"Caltech {category}: {TARGETS[category]}/{TARGETS[category]}")


def collect_funsd(curator: Curator) -> None:
    category = "degraded_document"
    if curator.remaining(category) == 0:
        return
    existing_files: set[str] = set()
    for csv_path in EXISTING_SOURCE_CSVS:
        for row in read_csv(csv_path):
            if row["source_name"] == "FUNSD":
                existing_files.add(row["source_file"])
    with zipfile.ZipFile(FUNSD_ZIP) as archive:
        for info in sorted(archive.infolist(), key=lambda value: value.filename):
            if (
                info.is_dir()
                or Path(info.filename).suffix.lower()
                not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
                or info.filename in existing_files
            ):
                continue
            curator.add(
                archive.read(info),
                category=category,
                source_name="FUNSD",
                source_url=FUNSD_URL,
                source_file=info.filename,
                license_name=(
                    "Non-commercial research and educational use"
                ),
                license_url="https://guillaumejaume.github.io/FUNSD/",
                author="FUNSD authors",
                origin_type="real_noisy_form_scan",
                language="en",
                has_text=True,
                quality_tags="real_form,low_resolution,scan_noise",
                privacy_review_required=False,
            )
            if curator.remaining(category) == 0:
                break
    if curator.remaining(category):
        raise RuntimeError(
            f"FUNSD missing {curator.remaining(category)} images"
        )
    print(f"FUNSD {category}: {TARGETS[category]}/{TARGETS[category]}")


def write_outputs(curator: Curator) -> dict[str, Any]:
    if dict(curator.counts) != TARGETS:
        raise RuntimeError(
            f"Category mismatch: {dict(curator.counts)} != {TARGETS}"
        )
    if len(curator.rows) != 1000:
        raise RuntimeError(f"Expected 1000 rows, found {len(curator.rows)}")
    if len({row["item_id"] for row in curator.rows}) != 1000:
        raise RuntimeError("Duplicate item_id found")
    files = sorted(OUTPUT_DIR.glob("*.jpg"))
    if len(files) != 1000:
        raise RuntimeError(f"Expected 1000 files, found {len(files)}")

    SOURCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curator.rows[0]))
        writer.writeheader()
        writer.writerows(curator.rows)

    source_counts = Counter(row["source_name"] for row in curator.rows)
    license_counts = Counter(row["license"] for row in curator.rows)
    summary = {
        "status": "collected_and_automatically_audited",
        "page_count": 1000,
        "final_library_target": 1500,
        "category_counts": dict(curator.counts),
        "source_counts": dict(source_counts),
        "license_counts": dict(license_counts),
        "exact_duplicates_with_existing_or_new": 0,
        "perceptual_near_duplicates_with_existing_or_new": 0,
        "declared_ai_generated_percent": 0,
        "category_review_required": True,
        "privacy_review_required_count": sum(
            row["privacy_review_required"] == "true"
            for row in curator.rows
        ),
        "total_size_mb": round(
            sum(path.stat().st_size for path in files) / 1024**2, 2
        ),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    required_files = (
        RVL_PARQUET,
        RICO_PARQUET,
        TEXTVQA_JSON,
        CALTECH_ZIP,
        FUNSD_ZIP,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source caches:\n" + "\n".join(missing))

    started = time.perf_counter()
    curator = Curator()
    print(
        f"Resume state: {len(curator.rows)}/1000, "
        f"{dict(curator.counts)}"
    )
    collect_rvl(curator)
    collect_rico(curator)
    collect_textvqa(curator)
    collect_caltech(curator)
    collect_funsd(curator)
    summary = write_outputs(curator)
    summary["collection_seconds"] = round(
        time.perf_counter() - started, 2
    )
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
