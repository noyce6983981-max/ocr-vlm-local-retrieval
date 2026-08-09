"""Run the public CPU reproduction and verify its frozen aggregate result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.minimal_repro import (  # noqa: E402
    evaluate,
    ingest,
    write_json_atomic,
)

DEFAULT_EXPECTED_RESULT = PROJECT_ROOT / "repro/expected_results.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/container_demo"
COMPARABLE_FIELDS = (
    "schema_version",
    "backend",
    "page_count",
    "query_count",
    "answerable_count",
    "no_answer_count",
    "metrics",
)


class DemoVerificationError(RuntimeError):
    """Raised when the generated demonstration differs from the frozen result."""


def _load_expected(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DemoVerificationError("Expected result must be a JSON object.")
    missing = [field for field in COMPARABLE_FIELDS if field not in payload]
    if missing:
        raise DemoVerificationError(
            f"Expected result is missing fields: {', '.join(missing)}"
        )
    return payload


def _comparable(payload: dict[str, Any]) -> dict[str, Any]:
    return {field: payload.get(field) for field in COMPARABLE_FIELDS}


def run_demo(output_dir: Path, expected_path: Path) -> dict[str, Any]:
    """Generate, evaluate, and strictly compare the CPU demonstration."""
    resolved_output = output_dir.resolve()
    resolved_expected = expected_path.resolve()
    expected = _load_expected(resolved_expected)
    receipt = ingest(resolved_output)
    report = evaluate(resolved_output, backend="cpu")
    observed = _comparable(report)
    frozen = _comparable(expected)
    if observed != frozen:
        failure = {
            "status": "failed",
            "expected": frozen,
            "observed": observed,
        }
        write_json_atomic(resolved_output / "verification.json", failure)
        raise DemoVerificationError(
            "CPU demonstration does not match repro/expected_results.json"
        )

    result = {
        "status": "passed",
        "backend": report["backend"],
        "page_count": report["page_count"],
        "query_count": report["query_count"],
        "metrics": report["metrics"],
        "expected_result": str(resolved_expected),
        "evaluation_result": str(
            resolved_output / "evaluation_cpu.json"
        ),
        "visual_index_built": receipt["visual_index_built"],
        "scope": (
            "Synthetic known-text CPU retrieval demonstration; "
            "not PaddleOCR or Qwen3-VL inference."
        ),
    }
    write_json_atomic(resolved_output / "verification.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated pages, indexes, and evaluation output.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=DEFAULT_EXPECTED_RESULT,
        help="Frozen aggregate result used for strict verification.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_demo(args.output_dir, args.expected)
    except (DemoVerificationError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Container demonstration failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
