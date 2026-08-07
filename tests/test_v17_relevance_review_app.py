from __future__ import annotations

import pytest

from scripts.v17_relevance_review_app import (
    install_translation_guard,
    moved_query_position,
    packet_index,
    packet_pool_sha256,
    review_is_complete,
    updated_relevance_draft,
)


def test_checkbox_change_is_copied_to_persistent_draft() -> None:
    original = {"candidate_a": False, "candidate_b": True}
    updated = updated_relevance_draft(original, "candidate_a", True)
    assert original == {"candidate_a": False, "candidate_b": True}
    assert updated == {"candidate_a": True, "candidate_b": True}


def test_packet_index_preserves_exact_query_binding() -> None:
    packets = [
        {"query_id": "q1", "query": "under his cap"},
        {"query_id": "q2", "query": "on his cap"},
    ]
    indexed = packet_index(packets)
    assert indexed["q1"]["query"] == "under his cap"
    assert indexed["q2"]["query"] == "on his cap"


def test_packet_index_rejects_duplicate_query_id() -> None:
    packets = [
        {"query_id": "q1", "query": "first"},
        {"query_id": "q1", "query": "second"},
    ]
    with pytest.raises(ValueError, match="unique query IDs"):
        packet_index(packets)


def test_translation_guard_marks_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_html(source: str, **kwargs: object) -> None:
        captured["source"] = source
        captured.update(kwargs)

    monkeypatch.setattr("scripts.v17_relevance_review_app.st.html", fake_html)
    install_translation_guard()
    source = str(captured["source"])
    assert 'lang = "zh-CN"' in source
    assert 'setAttribute("translate", "no")' in source
    assert 'classList.add("notranslate")' in source
    assert 'meta[name="google"]' in source
    assert 'meta.content = "notranslate"' in source
    assert captured["unsafe_allow_javascript"] is True


def test_query_position_navigation_is_bounded() -> None:
    assert moved_query_position(1, 1, 40) == 2
    assert moved_query_position(2, -1, 40) == 1
    assert moved_query_position(1, -1, 40) == 1
    assert moved_query_position(40, 1, 40) == 40
    with pytest.raises(ValueError, match="total must be positive"):
        moved_query_position(1, 1, 0)


def test_final_review_does_not_count_uncertain_as_complete() -> None:
    uncertain = {"pool_relevance": "uncertain"}
    relevant = {"pool_relevance": "relevant_candidate_in_pool"}
    assert review_is_complete(uncertain, require_final_decision=False)
    assert not review_is_complete(uncertain, require_final_decision=True)
    assert review_is_complete(relevant, require_final_decision=True)


def test_legacy_packet_gets_stable_pool_hash() -> None:
    packet = {
        "query_id": "q1",
        "study_fingerprint": "study",
        "candidates": [{"item_id": "b"}, {"item_id": "a"}],
    }
    assert packet_pool_sha256(packet) == packet_pool_sha256(packet)
    explicit = {**packet, "pool_sha256": "frozen"}
    assert packet_pool_sha256(explicit) == "frozen"
