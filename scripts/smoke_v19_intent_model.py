from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.routing import (  # noqa: E402
    IntentEvidence,
    IntentSchemaError,
    TransformersIntentBackend,
    map_evidence_to_route,
)

DEFAULT_MODEL = ROOT / "models/v19-intent-qwen3-0.6b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--query",
        default="找胸牌上写着‘志愿者’的那位穿蓝衣服的人。",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--prompt-lookup-num-tokens", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = TransformersIntentBackend(
        args.model,
        load_in_4bit=args.load_in_4bit,
        local_files_only=True,
        prompt_lookup_num_tokens=args.prompt_lookup_num_tokens,
    )
    started = time.perf_counter()
    raw_output = backend.generate_intent_json(args.query)
    elapsed_ms = (time.perf_counter() - started) * 1000
    try:
        evidence = IntentEvidence.from_json(raw_output)
    except IntentSchemaError as error:
        print(
            json.dumps(
                {
                    "backend": backend.name,
                    "query": args.query,
                    "raw_output": raw_output,
                    "cold_latency_ms": elapsed_ms,
                    "schema_error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "backend": backend.name,
                "query": args.query,
                "raw_output": raw_output,
                "route": map_evidence_to_route(evidence),
                "evidence": evidence.to_mapping(),
                "cold_latency_ms": elapsed_ms,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
