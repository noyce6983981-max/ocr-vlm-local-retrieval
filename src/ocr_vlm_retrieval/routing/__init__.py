"""V19 structured intent routing interfaces."""

from ocr_vlm_retrieval.routing.arbitration import (
    guard_llm_route,
    has_explicit_literal_requirement,
)
from ocr_vlm_retrieval.routing.cached_backend import (
    CachedIntentBackend,
    CacheStats,
)
from ocr_vlm_retrieval.routing.hybrid_router import (
    HybridDecision,
    HybridRouter,
)
from ocr_vlm_retrieval.routing.intervention import (
    INTENT_ROUTING_MODES,
    IntentRoutingMode,
    TransitionDecision,
    guard_v18_transition,
    validate_intent_routing_mode,
)
from ocr_vlm_retrieval.routing.llm_router import (
    IntentBackend,
    LLMDecision,
    LLMRouter,
)
from ocr_vlm_retrieval.routing.mapping import map_evidence_to_route
from ocr_vlm_retrieval.routing.prompt import SYSTEM_PROMPT, build_intent_messages
from ocr_vlm_retrieval.routing.rule_router import RuleDecision, RuleRouter
from ocr_vlm_retrieval.routing.schema import (
    IntentEvidence,
    IntentSchemaError,
    Route,
)
from ocr_vlm_retrieval.routing.transformers_backend import (
    TransformersIntentBackend,
)

__all__ = [
    "guard_llm_route",
    "has_explicit_literal_requirement",
    "CacheStats",
    "CachedIntentBackend",
    "HybridDecision",
    "HybridRouter",
    "INTENT_ROUTING_MODES",
    "IntentBackend",
    "IntentEvidence",
    "IntentRoutingMode",
    "IntentSchemaError",
    "LLMDecision",
    "LLMRouter",
    "Route",
    "RuleDecision",
    "RuleRouter",
    "SYSTEM_PROMPT",
    "TransformersIntentBackend",
    "TransitionDecision",
    "build_intent_messages",
    "guard_v18_transition",
    "map_evidence_to_route",
    "validate_intent_routing_mode",
]
