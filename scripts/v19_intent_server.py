"""Run the optional V19 hybrid intent router as a persistent loopback service."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.routing.cached_backend import (  # noqa: E402
    CachedIntentBackend,
)
from ocr_vlm_retrieval.routing.hybrid_router import HybridRouter  # noqa: E402
from ocr_vlm_retrieval.routing.llm_router import LLMRouter  # noqa: E402
from ocr_vlm_retrieval.routing.rule_router import RuleRouter  # noqa: E402
from ocr_vlm_retrieval.routing.service import route_payload  # noqa: E402
from ocr_vlm_retrieval.routing.transformers_backend import (  # noqa: E402
    TransformersIntentBackend,
)

DEFAULT_MODEL = ROOT / "models/v19-intent-qwen3-1.7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1",))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--prompt-lookup-num-tokens", type=int, default=5)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--cache-entries", type=int, default=1024)
    return parser.parse_args()


def build_router(args: argparse.Namespace) -> HybridRouter:
    model = args.model if args.model.is_absolute() else ROOT / args.model
    backend = TransformersIntentBackend(
        str(model.resolve()),
        max_new_tokens=args.max_new_tokens,
        load_in_4bit=args.load_in_4bit,
        local_files_only=True,
        prompt_lookup_num_tokens=args.prompt_lookup_num_tokens,
    )
    cached_backend = CachedIntentBackend(
        backend,
        max_entries=args.cache_entries,
    )
    return HybridRouter(
        RuleRouter.v19_calibrated(),
        LLMRouter(cached_backend),
    )


def handler_for(router: HybridRouter) -> type[BaseHTTPRequestHandler]:
    class IntentHandler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send(404, {"error": "not_found"})
                return
            self._send(200, {"status": "ok", "feature": "v19_intent_routing"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/route":
                self._send(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 16_384:
                    raise ValueError("invalid request body length")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict) or not isinstance(
                    payload.get("query"), str
                ):
                    raise ValueError("query must be a string")
                response = route_payload(router, payload["query"])
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._send(400, {"error": type(exc).__name__})
                return
            self._send(200, response)

        def log_message(self, format: str, *args: object) -> None:
            return

    return IntentHandler


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    router = build_router(args)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(router))
    print(f"V19 intent routing service: http://{args.host}:{args.port}")
    print("The model loads lazily on the first ambiguous query.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
