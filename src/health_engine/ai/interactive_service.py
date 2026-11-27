"""Module 4A: Stateless Interactive Wisdom Q&A & Simulator Advisor Service (`interactive_service.py`).

Designed for direct invocation by a backend / Cloudflare Worker endpoint without ever
exposing `GEMINI_API_KEY` to client browsers. Enforces `assert_packet_smoking_free` on every call.
"""

from __future__ import annotations

import json
from typing import Any

from health_engine.ai.fallbacks import fallback_interactive_qa, fallback_simulator_advice
from health_engine.ai.packet_builder import assert_packet_smoking_free, serialize_briefing_packet
from health_engine.ai.router import ModelRouter
from health_engine.ai.schemas import InteractiveAnswerSchema, SimulatorAdviceSchema

SYSTEM_PROMPT_4A_QA = """You are the Interactive Personal Health Intelligence Advisor (gemini-4-hydrogen).
Answer the user's question strictly using pre-computed statistics in AI_Briefing_Packet (and Google Search literature when enabled).
Separate 'what_your_data_shows' from 'what_research_says', cite verified URLs, classify confidence_tier honestly, set epistemic_label accurately ('Measured' | 'Descriptive' | 'Correlation' | 'Prediction' | 'Causal_Hypothesis' | 'Screening_Context'), and include a clear ask_your_doctor_note."""

SYSTEM_PROMPT_4A_SIM = """You are the Metabolic & Body-Composition Simulator Advisor (gemini-4-hydrogen).
Given the user's target weight, behavioral slider adjustments (sleep/bedtime, daily protein, steps, strength sessions), and pre-computed AI_Briefing_Packet:
1. Quote only pre-computed velocities and predictive feature weights without claiming causal certainty.
2. Provide lean-mass preservation guardrails (protein g/kg FFM, resistance training, sustainable weekly rate).
3. Set epistemic_label = 'Prediction' or 'Descriptive' and include an ask_your_doctor_note."""


def answer_health_question(
    question: str,
    packet: dict[str, Any],
    *,
    use_search: bool = True,
    router: ModelRouter | None = None,
) -> dict[str, Any]:
    """Stateless Q&A handler returning structured JSON dict with epistemic & citation metadata."""
    q_clean = str(question or "").strip()
    assert_packet_smoking_free(q_clean, path="user_question")
    packet_json = serialize_briefing_packet(packet)
    active_router = router or ModelRouter()

    res = active_router.generate_structured(
        module_id="4A_interactive_qa_and_simulator",
        system_prompt=SYSTEM_PROMPT_4A_QA,
        user_prompt=f"User Question: {q_clean}\n\nValidated AI_Briefing_Packet:\n{packet_json}",
        schema_cls=InteractiveAnswerSchema,
        fallback_fn=lambda: fallback_interactive_qa(q_clean, packet),
        override_use_search=use_search,
    )
    ans: InteractiveAnswerSchema = res.parsed or fallback_interactive_qa(q_clean, packet)

    citations_out = (
        [{"title": c.title, "url": c.url} for c in res.verified_citations]
        if res.verified_citations
        else [c.model_dump() for c in ans.citations]
    )
    return {
        "what_your_data_shows": ans.what_your_data_shows,
        "what_research_says": ans.what_research_says,
        "citations": citations_out,
        "confidence_tier": ans.confidence_tier,
        "epistemic_label": ans.epistemic_label,
        "ask_your_doctor_note": ans.ask_your_doctor_note,
        "engine_badge": res.engine_badge,
        "model_used": res.model_used,
    }


def advise_simulator_target(
    target_weight_kg: float,
    sliders_dict: dict[str, Any],
    packet: dict[str, Any],
    *,
    use_search: bool = False,
    router: ModelRouter | None = None,
) -> dict[str, Any]:
    """Stateless Simulator Advisor returning structured JSON dict for target weight & lever settings."""
    assert_packet_smoking_free(sliders_dict, path="sliders_dict")
    packet_json = serialize_briefing_packet(packet)
    active_router = router or ModelRouter()
    sliders_json = json.dumps(sliders_dict, ensure_ascii=False)

    res = active_router.generate_structured(
        module_id="4A_interactive_qa_and_simulator",
        system_prompt=SYSTEM_PROMPT_4A_SIM,
        user_prompt=(
            f"Target Weight: {float(target_weight_kg):.1f} kg\n"
            f"Proposed Behavioral Sliders: {sliders_json}\n\n"
            f"Validated AI_Briefing_Packet:\n{packet_json}"
        ),
        schema_cls=SimulatorAdviceSchema,
        fallback_fn=lambda: fallback_simulator_advice(float(target_weight_kg), sliders_dict, packet),
        override_use_search=use_search,
    )
    sim: SimulatorAdviceSchema = res.parsed or fallback_simulator_advice(
        float(target_weight_kg), sliders_dict, packet
    )
    citations_out = (
        [{"title": c.title, "url": c.url} for c in res.verified_citations]
        if res.verified_citations
        else [c.model_dump() for c in sim.citations]
    )
    return {
        "what_your_data_shows": sim.what_your_data_shows,
        "what_research_says": sim.what_research_says,
        "realistic_timeline_and_lean_mass_guardrail": sim.realistic_timeline_and_lean_mass_guardrail,
        "prioritized_lever_adjustments": sim.prioritized_lever_adjustments,
        "citations": citations_out,
        "confidence_tier": sim.confidence_tier,
        "epistemic_label": sim.epistemic_label,
        "ask_your_doctor_note": sim.ask_your_doctor_note,
        "engine_badge": res.engine_badge,
        "model_used": res.model_used,
    }
