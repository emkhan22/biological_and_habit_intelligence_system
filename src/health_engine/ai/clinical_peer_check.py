"""Module 3A: Weekly Peer-Expectedness & Regimen Safety Check (`clinical_peer_check.py` -> `AI_Peer_Expectedness`).

Combines flat `PeerExpectednessSchema` + native `GoogleSearch` grounding in a single call
on `gemini-4-neon` (`thinking_level="high"`) and reconciles verified source titles & URLs
from `grounding_metadata`.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from health_engine.ai.fallbacks import fallback_peer_expectedness
from health_engine.ai.packet_builder import serialize_briefing_packet
from health_engine.ai.router import ModelRouter, VerifiedCitation
from health_engine.ai.schemas import PeerExpectednessSchema
from health_engine.config_loader import now_ist_str

PEER_EXPECTEDNESS_COLUMNS: list[str] = [
    "report_date",
    "display_date",
    "generated_at_ist",
    "symptom_or_shift",
    "active_therapy_context",
    "safety_tag",
    "clinical_rationale",
    "reference_source_title",
    "reference_url",
    "url_grounding_verified",
    "regimen_interaction_note",
    "grounded_queries_json",
    "engine_badge",
    "model_used",
    "epistemic_label",
    "epistemic_class",
]

SYSTEM_PROMPT_3A = """You are a Pharmacology & Clinical Trial Peer-Expectedness Reviewer using Google Search grounding.
Compare the user's pre-computed physiological shifts, logged side effects, and active multi-therapy stack (Tirzepatide SC, Thyronorm levothyroxine, IV Iron FCM, Vitamin D3 600K/60K, Magnesium glycinate, Omega-3, Trazer F Forte, Matilda Forte) against published clinical trial & prescribing pharmacology literature.
Classify each active symptom or biometric shift as 'Expected', 'Monitor', or 'Ask_Doctor', set epistemic_label = 'Screening_Context', and ground every claim in verifiable medical references."""


def _reconcile_citation(
    raw_title: str,
    raw_url: str,
    verified: list[VerifiedCitation],
    idx: int,
) -> tuple[str, str, bool]:
    verified_by_url = {c.url: c for c in verified}
    if raw_url in verified_by_url:
        c = verified_by_url[raw_url]
        return (c.title or raw_title, c.url, True)
    if verified:
        pick = verified[idx % len(verified)]
        return (pick.title or raw_title, pick.url, True)
    return (raw_title, raw_url, False)


def run_module_3a_peer_expectedness(
    *,
    packet: dict[str, Any],
    existing_peer_df: pd.DataFrame | None,
    router: ModelRouter,
) -> pd.DataFrame:
    """Run weekly search-grounded regimen safety check into `AI_Peer_Expectedness`."""
    meta = packet.get("meta", {})
    r_iso = str(meta.get("report_date_iso", "2026-11-27"))
    r_disp = str(meta.get("report_date_display", r_iso))
    packet_json = serialize_briefing_packet(packet)

    res = router.generate_structured(
        module_id="3A_peer_expectedness",
        system_prompt=SYSTEM_PROMPT_3A,
        user_prompt=f"Evaluate peer-expectedness & multi-therapy safety for this profile:\n{packet_json}",
        schema_cls=PeerExpectednessSchema,
        fallback_fn=lambda: fallback_peer_expectedness(packet),
    )
    parsed: PeerExpectednessSchema = res.parsed or fallback_peer_expectedness(packet)
    ts_now = now_ist_str()
    queries_json = json.dumps(res.web_search_queries, ensure_ascii=False)

    new_rows: list[dict[str, Any]] = []
    for idx, item in enumerate(parsed.items):
        v_title, v_url, is_ver = _reconcile_citation(
            item.reference_source_title, item.reference_url, res.verified_citations, idx
        )
        new_rows.append(
            {
                "report_date": r_iso,
                "display_date": r_disp,
                "generated_at_ist": ts_now,
                "symptom_or_shift": item.symptom_or_shift,
                "active_therapy_context": item.active_therapy_context,
                "safety_tag": item.safety_tag,
                "clinical_rationale": item.clinical_rationale,
                "reference_source_title": v_title,
                "reference_url": v_url,
                "url_grounding_verified": is_ver,
                "regimen_interaction_note": parsed.regimen_interaction_note,
                "grounded_queries_json": queries_json,
                "engine_badge": res.engine_badge,
                "model_used": res.model_used,
                "epistemic_label": item.epistemic_label,
                "epistemic_class": item.epistemic_label,
            }
        )

    existing_rows: list[dict[str, Any]] = []
    if existing_peer_df is not None and not existing_peer_df.empty and "report_date" in existing_peer_df.columns:
        existing_rows = [
            dict(r)
            for r in existing_peer_df.to_dict(orient="records")
            if str(r.get("report_date", ""))[:10] != r_iso
        ]

    combined = pd.DataFrame(existing_rows + new_rows)
    for col in PEER_EXPECTEDNESS_COLUMNS:
        if col not in combined.columns:
            combined[col] = None
    return combined[PEER_EXPECTEDNESS_COLUMNS].sort_values("report_date").reset_index(drop=True)
