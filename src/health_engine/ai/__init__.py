"""V2.8 Model-Routed AI Synthesis Layer governed by AGENTS.md and config/ai_routing.yaml."""

from health_engine.ai.packet_builder import (
    assert_packet_smoking_free,
    build_ai_briefing_packet,
)
from health_engine.ai.router import (
    AICallResult,
    BudgetCircuitBreaker,
    GeminiConfigGuard,
    ModelRouter,
    RouteDecision,
)

__all__ = [
    "AICallResult",
    "BudgetCircuitBreaker",
    "GeminiConfigGuard",
    "ModelRouter",
    "RouteDecision",
    "assert_packet_smoking_free",
    "build_ai_briefing_packet",
]
