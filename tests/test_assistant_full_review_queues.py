"""Tests for blind assistant full-review queue construction."""

from __future__ import annotations

from scripts.build_assistant_full_review_queues import (
    BLIND_SPLIT_CODES,
    blind_order,
)


def test_blind_orders_are_deterministic_and_round_specific() -> None:
    item_id = "user_example"
    assert blind_order(item_id, 4) == blind_order(item_id, 4)
    assert blind_order(item_id, 4) != blind_order(item_id, 5)


def test_blind_split_codes_are_unique() -> None:
    assert len(set(BLIND_SPLIT_CODES.values())) == 3
