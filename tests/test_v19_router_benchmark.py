from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.benchmark_v18_1_router import percentile


def test_percentile_uses_conservative_nearest_rank() -> None:
    assert percentile([1.0, 2.0, 3.0, 100.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0, 100.0], 0.95) == 100.0
    assert percentile([], 0.95) == 0.0
