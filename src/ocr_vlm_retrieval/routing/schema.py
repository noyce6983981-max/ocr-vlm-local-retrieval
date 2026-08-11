"""Strict structured evidence schema for V19 intent routing."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal, TypeAlias, cast

Route: TypeAlias = Literal[
    "text_evidence",
    "visual_discovery",
    "visual_metadata",
    "entity_exact",
    "topic_discovery",
    "mixed",
]

ROUTES: Final[frozenset[str]] = frozenset(
    {
        "text_evidence",
        "visual_discovery",
        "visual_metadata",
        "entity_exact",
        "topic_discovery",
        "mixed",
    }
)

REQUIRED_INTENT_FIELDS: Final[tuple[str, ...]] = (
    "needs_literal_text",
    "needs_visual_semantics",
    "needs_layout_structure",
    "needs_exact_entity",
    "needs_topic_discovery",
    "is_compositional",
)
OPTIONAL_INTENT_FIELDS: Final[frozenset[str]] = frozenset({"confidence"})


class IntentSchemaError(ValueError):
    """Raised when a model response is not a valid V19 intent object."""


def validate_route(value: str) -> Route:
    """Validate and narrow one public route string."""

    if value not in ROUTES:
        raise ValueError(f"Unsupported retrieval route: {value!r}")
    return cast(Route, value)


@dataclass(frozen=True, slots=True)
class IntentEvidence:
    """Evidence needs predicted by a local language model.

    ``confidence`` is diagnostic metadata only. The hybrid router must not use
    a model's self-reported confidence as an invocation or acceptance gate.
    """

    needs_literal_text: bool
    needs_visual_semantics: bool
    needs_layout_structure: bool
    needs_exact_entity: bool
    needs_topic_discovery: bool
    is_compositional: bool
    confidence: float | None = None

    def __post_init__(self) -> None:
        for field_name in REQUIRED_INTENT_FIELDS:
            if type(getattr(self, field_name)) is not bool:
                raise IntentSchemaError(f"{field_name} must be a boolean")
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(
                self.confidence, (int, float)
            ):
                raise IntentSchemaError("confidence must be numeric")
            if not 0.0 <= float(self.confidence) <= 1.0:
                raise IntentSchemaError("confidence must be between 0 and 1")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> IntentEvidence:
        """Parse a mapping while rejecting missing and unknown fields."""

        keys = set(value)
        required = set(REQUIRED_INTENT_FIELDS)
        missing = sorted(required - keys)
        unknown = sorted(keys - required - OPTIONAL_INTENT_FIELDS)
        if missing:
            raise IntentSchemaError(
                "Missing intent fields: " + ", ".join(missing)
            )
        if unknown:
            raise IntentSchemaError(
                "Unknown intent fields: " + ", ".join(unknown)
            )
        return cls(
            needs_literal_text=value["needs_literal_text"],
            needs_visual_semantics=value["needs_visual_semantics"],
            needs_layout_structure=value["needs_layout_structure"],
            needs_exact_entity=value["needs_exact_entity"],
            needs_topic_discovery=value["needs_topic_discovery"],
            is_compositional=value["is_compositional"],
            confidence=value.get("confidence"),
        )

    @classmethod
    def from_json(cls, value: str) -> IntentEvidence:
        """Parse one JSON object without accepting prose or JSON arrays."""

        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise IntentSchemaError("Intent output is not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise IntentSchemaError("Intent output must be one JSON object")
        return cls.from_mapping(payload)

    def to_mapping(self) -> dict[str, bool | float | None]:
        """Return a serializable audit representation."""

        return asdict(self)
