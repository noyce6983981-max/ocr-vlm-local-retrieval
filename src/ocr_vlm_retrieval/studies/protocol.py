"""Typed loading and validation for immutable study protocols."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProtocolError(ValueError):
    """Raised when a study protocol violates its declared design."""


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{field} must be an object")
    return value


def _text(value: object, field: str) -> str:
    result = str(value).strip()
    if not result:
        raise ProtocolError(f"{field} must be non-empty")
    return result


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProtocolError(f"{field} must be a positive integer")
    return value


def _count_mapping(value: object, field: str) -> dict[str, int]:
    source = _mapping(value, field)
    result: dict[str, int] = {}
    for raw_key, raw_count in source.items():
        key = _text(raw_key, f"{field} key")
        result[key] = _positive_int(raw_count, f"{field}.{key}")
    if not result:
        raise ProtocolError(f"{field} must not be empty")
    return result


@dataclass(frozen=True)
class QueryDesign:
    """Frozen sample-size and grouping contract for a study."""

    total_queries: int
    source_group_count: int
    queries_per_source_group: int
    splits: dict[str, int]
    roles: dict[str, int]
    strata: dict[str, int]
    languages: dict[str, int]

    @classmethod
    def from_mapping(cls, value: object) -> QueryDesign:
        source = _mapping(value, "query_design")
        design = cls(
            total_queries=_positive_int(
                source.get("total_queries"), "query_design.total_queries"
            ),
            source_group_count=_positive_int(
                source.get("source_group_count"),
                "query_design.source_group_count",
            ),
            queries_per_source_group=_positive_int(
                source.get("queries_per_source_group"),
                "query_design.queries_per_source_group",
            ),
            splits=_count_mapping(source.get("splits"), "query_design.splits"),
            roles=_count_mapping(source.get("roles"), "query_design.roles"),
            strata=_count_mapping(source.get("strata"), "query_design.strata"),
            languages=_count_mapping(source.get("languages"), "query_design.languages"),
        )
        design.validate()
        return design

    def validate(self) -> None:
        if (
            self.source_group_count * self.queries_per_source_group
            != self.total_queries
        ):
            raise ProtocolError(
                "source groups multiplied by queries per group must equal total_queries"
            )
        for name, counts in (
            ("splits", self.splits),
            ("roles", self.roles),
            ("strata", self.strata),
            ("languages", self.languages),
        ):
            if sum(counts.values()) != self.total_queries:
                raise ProtocolError(f"query_design.{name} must sum to total_queries")
        if set(self.splits) != {"calibration", "holdout"}:
            raise ProtocolError("query_design.splits must be calibration and holdout")

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "source_group_count": self.source_group_count,
            "queries_per_source_group": self.queries_per_source_group,
            "splits": dict(self.splits),
            "roles": dict(self.roles),
            "strata": dict(self.strata),
            "languages": dict(self.languages),
        }


@dataclass(frozen=True)
class StudyProtocol:
    """Validated protocol plus research and governance declarations."""

    schema_version: int
    study_id: str
    study_version: str
    lifecycle: str
    predecessor_release: str
    research_question: str
    query_design: QueryDesign
    hypotheses: tuple[dict[str, Any], ...]
    methods: tuple[dict[str, Any], ...]
    review_design: dict[str, Any]
    governance: dict[str, Any]
    amendments: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object) -> StudyProtocol:
        source = _mapping(value, "protocol")
        predecessor = _mapping(source.get("predecessor"), "predecessor")
        hypotheses = _object_sequence(source.get("hypotheses"), "hypotheses")
        methods = _object_sequence(source.get("methods"), "methods")
        protocol = cls(
            schema_version=_positive_int(
                source.get("schema_version"), "schema_version"
            ),
            study_id=_text(source.get("study_id"), "study_id"),
            study_version=_text(source.get("study_version"), "study_version"),
            lifecycle=_text(source.get("lifecycle"), "lifecycle"),
            predecessor_release=_text(
                predecessor.get("release"), "predecessor.release"
            ),
            research_question=_text(
                source.get("research_question"), "research_question"
            ),
            query_design=QueryDesign.from_mapping(source.get("query_design")),
            hypotheses=hypotheses,
            methods=methods,
            review_design=dict(_mapping(source.get("review_design"), "review_design")),
            governance=dict(_mapping(source.get("governance"), "governance")),
            amendments=_text_sequence(source.get("amendments"), "amendments"),
        )
        protocol.validate()
        return protocol

    def validate(self) -> None:
        hypothesis_ids = [
            _text(row.get("id"), "hypotheses.id") for row in self.hypotheses
        ]
        method_ids = [_text(row.get("id"), "methods.id") for row in self.methods]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ProtocolError("hypothesis IDs must be unique")
        if len(method_ids) != len(set(method_ids)):
            raise ProtocolError("method IDs must be unique")
        if not hypothesis_ids or not method_ids:
            raise ProtocolError("protocol needs at least one hypothesis and method")
        reviewer_count = self.review_design.get("independent_relevance_reviewers")
        if (
            _positive_int(
                reviewer_count, "review_design.independent_relevance_reviewers"
            )
            < 2
        ):
            raise ProtocolError(
                "at least two independent relevance reviewers are required"
            )
        if self.governance.get("holdout_runs") != 1:
            raise ProtocolError("governance.holdout_runs must equal one")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "study_version": self.study_version,
            "lifecycle": self.lifecycle,
            "predecessor": {"release": self.predecessor_release},
            "research_question": self.research_question,
            "query_design": self.query_design.to_dict(),
            "hypotheses": [dict(row) for row in self.hypotheses],
            "methods": [dict(row) for row in self.methods],
            "review_design": dict(self.review_design),
            "governance": dict(self.governance),
        }
        if self.amendments:
            result["amendments"] = list(self.amendments)
        return result


def _object_sequence(value: object, field: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise ProtocolError(f"{field} must be a non-empty array")
    return tuple(dict(_mapping(row, f"{field} item")) for row in value)


def _text_sequence(value: object, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ProtocolError(f"{field} must be an array")
    result = tuple(_text(item, f"{field} item") for item in value)
    if len(result) != len(set(result)):
        raise ProtocolError(f"{field} values must be unique")
    return result


def load_study_protocol(path: Path) -> StudyProtocol:
    """Load a UTF-8 JSON protocol and enforce its internal count contract."""

    return StudyProtocol.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def protocol_fingerprint(protocol: StudyProtocol) -> str:
    """Return the canonical SHA-256 identity of a validated protocol."""

    payload = json.dumps(
        protocol.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
