"""Study-only live-search entry point that can bypass entity short-circuiting."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import live_search

_resolve_search_intent = live_search.resolve_search_intent


def resolve_v18_search_intent(
    method: str, query: str
) -> tuple[str, bool, str, str | None]:
    route, exploratory, visual_query, strict_entity = _resolve_search_intent(
        method, query
    )
    if route == "entity_exact" and strict_entity is not None:
        return "mixed", False, query, None
    return route, exploratory, visual_query, strict_entity


def main() -> None:
    live_search.resolve_search_intent = resolve_v18_search_intent
    live_search.main()


if __name__ == "__main__":
    main()
