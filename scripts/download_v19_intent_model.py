from __future__ import annotations

import argparse
from pathlib import Path

from modelscope import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {
    "0.6b": ("Qwen/Qwen3-0.6B", ROOT / "models/v19-intent-qwen3-0.6b"),
    "1.7b": ("Qwen/Qwen3-1.7B", ROOT / "models/v19-intent-qwen3-1.7b"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download one isolated V19 intent-routing model candidate."
    )
    parser.add_argument("candidate", choices=tuple(CANDIDATES))
    parser.add_argument("--revision", default="master")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_id, target = CANDIDATES[args.candidate]
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        model_id=model_id,
        revision=args.revision,
        local_dir=str(target),
        max_workers=4,
    )
    print(f"Downloaded {model_id}@{args.revision} to {resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
