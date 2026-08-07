"""Collect a reproducible 200-page public OCR/VLM research set.

Raw images are intentionally kept under data/incoming/ and are ignored by Git.
The tracked source manifest records where every page came from and its license.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data/incoming/200页全部放这里"
CACHE_DIR = PROJECT_ROOT / "work/dataset_200_cache"
MANIFEST_PATH = (
    PROJECT_ROOT / "data/evaluation/public_dataset_200_sources.csv"
)
SUMMARY_PATH = (
    PROJECT_ROOT / "data/evaluation/public_dataset_200_summary.json"
)
USER_AGENT = "OCR-VLM-research-curator/1.0 (academic dataset build)"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

XFUND_URL = (
    "https://github.com/doc-analysis/XFUND/releases/download/v1.0/"
    "zh.train.zip"
)
FUNSD_URL = "https://www.crc.nd.edu/~pmoreira/funsd.zip"
PADDLE_REPO = "PaddlePaddle/PaddleOCR"
PADDLE_LICENSE_URL = (
    "https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE"
)
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

CATEGORY_SPECS = {
    "clear_document": 30,
    "complex_academic": 30,
    "table_form_ticket": 30,
    "ppt_poster_slide": 25,
    "software_web_code": 20,
    "scene_text": 25,
    "degraded_document": 20,
    "natural_no_text": 20,
}


def request_bytes(url: str, retries: int = 1) -> bytes:
    completed = subprocess.run(
        [
            "curl.exe",
            "-sS",
            "-L",
            "--fail",
            "--retry",
            str(retries),
            "--retry-all-errors",
            "--connect-timeout",
            "20",
            "--max-time",
            "35",
            "-H",
            f"User-Agent: {USER_AGENT}",
            url,
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"download failed: {completed.stderr.decode(errors='replace')}"
        )
    return completed.stdout


def request_json(url: str, params: dict[str, Any] | None = None) -> Any:
    if params:
        url += "?" + urllib.parse.urlencode(params)
    if url.startswith("https://api.github.com/"):
        completed = subprocess.run(
            [
                "curl.exe",
                "-sS",
                "-L",
                "--fail",
                "--retry",
                "6",
                "--retry-all-errors",
                "--connect-timeout",
                "20",
                "-H",
                f"User-Agent: {USER_AGENT}",
                url,
            ],
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"GitHub API request failed: {completed.stderr.decode(errors='replace')}"
            )
        return json.loads(completed.stdout.decode("utf-8"))
    return json.loads(request_bytes(url).decode("utf-8"))


def download_cached(url: str, destination: Path) -> Path:
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    completed = subprocess.run(
        [
            "curl.exe",
            "-L",
            "--fail",
            "--retry",
            "8",
            "--retry-all-errors",
            "--connect-timeout",
            "20",
            "--speed-limit",
            "1024",
            "--speed-time",
            "30",
            "--output",
            str(temporary),
            url,
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"curl failed with exit code {completed.returncode}: {url}"
        )
    temporary.replace(destination)
    return destination


def normalized_jpeg_bytes(payload: bytes) -> tuple[bytes, int, int]:
    with Image.open(io.BytesIO(payload)) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", image.size, "white")
            alpha = image.getchannel("A")
            background.paste(image.convert("RGB"), mask=alpha)
            image = background
        else:
            image = image.convert("RGB")
        width, height = image.size
        if min(width, height) < 240:
            raise ValueError(f"image too small: {width}x{height}")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=92, optimize=True)
        return output.getvalue(), width, height


def save_page(
    payload: bytes,
    category: str,
    index: int,
    seen_hashes: set[str],
) -> tuple[Path, str, int, int]:
    jpeg, width, height = normalized_jpeg_bytes(payload)
    digest = hashlib.sha256(jpeg).hexdigest()
    if digest in seen_hashes:
        raise ValueError("duplicate image content")
    seen_hashes.add(digest)
    output_path = OUTPUT_DIR / f"{category}_{index:03d}.jpg"
    output_path.write_bytes(jpeg)
    return output_path, digest, width, height


def image_members(archive_path: Path) -> list[str]:
    with zipfile.ZipFile(archive_path) as archive:
        return sorted(
            info.filename
            for info in archive.infolist()
            if not info.is_dir()
            and Path(info.filename).suffix.lower() in IMAGE_SUFFIXES
        )


def collect_zip_pages(
    archive_path: Path,
    category_counts: list[tuple[str, int]],
    source_name: str,
    source_url: str,
    license_name: str,
    license_url: str,
    language: str,
    quality_tags: str,
    seen_hashes: set[str],
    rows: list[dict[str, Any]],
) -> None:
    members = image_members(archive_path)
    required = sum(count for _, count in category_counts)
    if len(members) < required:
        raise RuntimeError(
            f"{source_name} has {len(members)} images; need {required}."
        )
    cursor = 0
    with zipfile.ZipFile(archive_path) as archive:
        for category, count in category_counts:
            created = 0
            while created < count:
                member = members[cursor]
                cursor += 1
                payload = archive.read(member)
                index = created + 1
                try:
                    output, digest, width, height = save_page(
                        payload, category, index, seen_hashes
                    )
                except ValueError:
                    continue
                group_number = (index - 1) // 2 + 1
                rows.append(
                    {
                        "item_id": digest[:16],
                        "filename": output.name,
                        "category": category,
                        "language": language,
                        "has_text": True,
                        "quality_tags": quality_tags,
                        "source_name": source_name,
                        "source_url": source_url,
                        "source_file": member,
                        "license": license_name,
                        "license_url": license_url,
                        "hard_negative_group": (
                            f"{category}_{group_number:02d}"
                            if index <= 20
                            else ""
                        ),
                        "width": width,
                        "height": height,
                        "review_status": "待人工审核",
                    }
                )
                created += 1


def github_tree(repo: str) -> tuple[str, list[dict[str, Any]]]:
    repository_dir = CACHE_DIR / repo.split("/")[-1]
    if not (repository_dir / ".git").is_dir():
        completed = subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--depth",
                "1",
                f"https://github.com/{repo}.git",
                str(repository_dir),
            ],
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Unable to clone Git tree for {repo}.")
    commit_sha = subprocess.run(
        ["git", "-C", str(repository_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    listing = subprocess.run(
        [
            "git",
            "-C",
            str(repository_dir),
            "ls-tree",
            "-r",
            "--name-only",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout.splitlines()
    return commit_sha, [
        {"type": "blob", "size": 100_000, "path": path}
        for path in listing
    ]


def paddle_candidates(
    tree: Iterable[dict[str, Any]],
    category: str,
) -> list[dict[str, Any]]:
    images = [
        row
        for row in tree
        if row.get("type") == "blob"
        and Path(row["path"]).suffix.lower() in IMAGE_SUFFIXES
        and 20_000 <= int(row.get("size", 0)) <= 10_000_000
    ]
    patterns = {
        "complex_academic": (
            r"ppstructure|CDLA_demo|publaynet_demo|table_|funsd_demo|"
            r"xfund_demo|doc_with_formula|book\.jpg"
        ),
        "ppt_poster_slide": (
            r"algorithm|framework|architecture|Banner|PP-OCRv3-pic|"
            r"docs/version2\.x/ppocr/"
        ),
        "software_web_code": (
            r"data_anno_synth|community/images|deploy/.*/images|"
            r"deploy/.*/Resources/SampleImages|test_tipc/docs|"
            r"docs/version2\.x/legacy"
        ),
    }
    pattern = re.compile(patterns[category], re.IGNORECASE)
    candidates = [row for row in images if pattern.search(row["path"])]
    return sorted(candidates, key=lambda row: row["path"])


def collect_paddle_pages(
    seen_hashes: set[str],
    rows: list[dict[str, Any]],
) -> None:
    commit_sha, tree = github_tree(PADDLE_REPO)
    used_paths: set[str] = set()
    for category in (
        "complex_academic",
        "ppt_poster_slide",
        "software_web_code",
    ):
        required = CATEGORY_SPECS[category]
        created = 0
        for candidate in paddle_candidates(tree, category):
            path = candidate["path"]
            if path in used_paths:
                continue
            raw_url = (
                f"https://raw.githubusercontent.com/{PADDLE_REPO}/"
                f"{commit_sha}/{urllib.parse.quote(path)}"
            )
            try:
                payload = request_bytes(raw_url)
                output, digest, width, height = save_page(
                    payload, category, created + 1, seen_hashes
                )
            except Exception as error:
                print(f"skip PaddleOCR {path}: {error}")
                continue
            used_paths.add(path)
            created += 1
            rows.append(
                {
                    "item_id": digest[:16],
                    "filename": output.name,
                    "category": category,
                    "language": "mixed",
                    "has_text": True,
                    "quality_tags": "open_source_example",
                    "source_name": "PaddleOCR official repository",
                    "source_url": (
                        f"https://github.com/{PADDLE_REPO}/blob/"
                        f"{commit_sha}/{path}"
                    ),
                    "source_file": path,
                    "license": "Apache-2.0",
                    "license_url": PADDLE_LICENSE_URL,
                    "hard_negative_group": (
                        f"{category}_{(created - 1) // 2 + 1:02d}"
                        if created <= 10
                        else ""
                    ),
                    "width": width,
                    "height": height,
                    "review_status": "待人工审核",
                }
            )
            if created == required:
                break
        if created != required:
            raise RuntimeError(
                f"Only collected {created}/{required} PaddleOCR "
                f"images for {category}."
            )


def strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value or "").strip()


def commons_files(category: str, limit: int = 200) -> list[str]:
    titles: list[str] = []
    continuation: str | None = None
    while len(titles) < limit:
        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtype": "file",
            "cmlimit": "50",
            "cmtitle": f"Category:{category}",
        }
        if continuation:
            params["cmcontinue"] = continuation
        payload = request_json(COMMONS_API, params)
        titles.extend(row["title"] for row in payload["query"]["categorymembers"])
        continuation = payload.get("continue", {}).get("cmcontinue")
        if not continuation:
            break
    return titles[:limit]


def commons_image_info(title: str) -> dict[str, Any] | None:
    payload = request_json(
        COMMONS_API,
        {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "titles": title,
            "iiprop": "url|extmetadata|mime|size",
            "iiurlwidth": "1600",
        },
    )
    page = next(iter(payload["query"]["pages"].values()))
    image_info = (page.get("imageinfo") or [None])[0]
    return image_info


def collect_commons_category(
    commons_categories: list[str],
    output_category: str,
    required: int,
    has_text: bool,
    language: str,
    seen_hashes: set[str],
    rows: list[dict[str, Any]],
) -> None:
    accepted_licenses = (
        "CC BY",
        "CC0",
        "Public domain",
        "PDM",
    )
    created = 0
    seen_titles: set[str] = set()
    for commons_category in commons_categories:
        for title in commons_files(commons_category):
            if title in seen_titles:
                continue
            seen_titles.add(title)
            try:
                info = commons_image_info(title)
                if not info:
                    continue
                metadata = info.get("extmetadata", {})
                license_name = metadata.get("LicenseShortName", {}).get(
                    "value", ""
                )
                if not license_name.startswith(accepted_licenses):
                    continue
                mime = info.get("mime", "")
                if not mime.startswith("image/"):
                    continue
                image_url = info.get("thumburl") or info.get("url")
                payload = request_bytes(image_url)
                output, digest, width, height = save_page(
                    payload, output_category, created + 1, seen_hashes
                )
            except Exception as error:
                print(f"skip Commons {title}: {error}")
                continue
            created += 1
            description_url = info.get("descriptionurl", "")
            rows.append(
                {
                    "item_id": digest[:16],
                    "filename": output.name,
                    "category": output_category,
                    "language": language,
                    "has_text": has_text,
                    "quality_tags": (
                        "scene_photo" if has_text else "natural_image"
                    ),
                    "source_name": "Wikimedia Commons",
                    "source_url": description_url,
                    "source_file": title,
                    "license": strip_html(license_name),
                    "license_url": metadata.get("LicenseUrl", {}).get(
                        "value", ""
                    ),
                    "hard_negative_group": (
                        f"{output_category}_{(created - 1) // 2 + 1:02d}"
                        if created <= 10
                        else ""
                    ),
                    "width": width,
                    "height": height,
                    "review_status": "待人工审核",
                }
            )
            if created == required:
                return
    raise RuntimeError(
        f"Only collected {created}/{required} Commons images "
        f"for {output_category}."
    )


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    images = sorted(OUTPUT_DIR.glob("*.jpg"))
    if len(images) != 200 or len(rows) != 200:
        raise RuntimeError(
            f"Expected 200 pages, found {len(images)} files and "
            f"{len(rows)} manifest rows."
        )
    counts: dict[str, int] = {}
    hashes: set[str] = set()
    total_bytes = 0
    for row in rows:
        path = OUTPUT_DIR / row["filename"]
        with Image.open(path) as image:
            image.verify()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            raise RuntimeError(f"Duplicate output: {path.name}")
        hashes.add(digest)
        total_bytes += path.stat().st_size
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    if counts != CATEGORY_SPECS:
        raise RuntimeError(f"Category mismatch: {counts}")
    hard_negative_pages = sum(
        bool(row["hard_negative_group"]) for row in rows
    )
    if hard_negative_pages < 50:
        raise RuntimeError("Fewer than 50 hard-negative pages.")
    return {
        "page_count": len(rows),
        "category_counts": counts,
        "unique_sha256_count": len(hashes),
        "hard_negative_pages": hard_negative_pages,
        "total_size_mb": round(total_bytes / 1024**2, 2),
        "review_status": "待人工审核",
    }


def write_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with MANIFEST_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = validate(rows)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(OUTPUT_DIR.iterdir())
    if existing:
        raise RuntimeError(
            f"{OUTPUT_DIR} is not empty. Move its contents before rebuilding."
        )

    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    try:
        xfund_zip = download_cached(
            XFUND_URL, CACHE_DIR / "xfund_zh_train.zip"
        )
        collect_zip_pages(
            xfund_zip,
            [("clear_document", 30), ("table_form_ticket", 30)],
            "XFUND Chinese train",
            XFUND_URL,
            "CC BY-NC-SA 4.0",
            "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "zh",
            "clear_scan,form_layout",
            seen_hashes,
            rows,
        )
        print("XFUND: 60/60")

        funsd_zip = download_cached(FUNSD_URL, CACHE_DIR / "funsd.zip")
        collect_zip_pages(
            funsd_zip,
            [("degraded_document", 20)],
            "FUNSD",
            FUNSD_URL,
            "Non-commercial research and educational use",
            "https://guillaumejaume.github.io/FUNSD/",
            "en",
            "low_resolution,scan_noise",
            seen_hashes,
            rows,
        )
        print("FUNSD: 20/20")

        collect_paddle_pages(seen_hashes, rows)
        print("PaddleOCR: 75/75")

        collect_commons_category(
            ["Road signs in China", "Shop signs in China", "Signs in China"],
            "scene_text",
            25,
            True,
            "zh_or_mixed",
            seen_hashes,
            rows,
        )
        print("Wikimedia scene text: 25/25")
        collect_commons_category(
            ["Landscapes of China", "Lakes of China"],
            "natural_no_text",
            20,
            False,
            "none",
            seen_hashes,
            rows,
        )
        print("Wikimedia natural: 20/20")

        summary = write_records(rows)
    except Exception:
        # Preserve downloaded archives, but keep a failed build out of the
        # user's upload folder so it cannot be mistaken for a valid dataset.
        failed_dir = (
            PROJECT_ROOT
            / "work"
            / f"dataset_200_failed_{int(time.time())}"
        )
        if any(OUTPUT_DIR.iterdir()):
            shutil.move(str(OUTPUT_DIR), str(failed_dir))
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        raise

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
