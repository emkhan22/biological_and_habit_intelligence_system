"""Module 2C: Monthly Whole-Body Clinical Review (`brief_monthly.py` -> `AI_Monthly_Briefs`)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from health_engine.ai.fallbacks import fallback_monthly_review
from health_engine.ai.packet_builder import serialize_briefing_packet
from health_engine.ai.router import ModelRouter
from health_engine.ai.schemas import MonthlyReviewSchema
from health_engine.config_loader import now_ist_str

MONTHLY_BRIEF_COLUMNS: list[str] = [
    "month_yyyy_mm",
    "report_date",
    "display_date",
    "generated_at_ist",
    "headline",
    "body_comp_fat_vs_lean_summary",
    "masld_and_metabolic_trajectory",
    "menstrual_cycle_and_hematology_summary",
    "treatment_scorecards_synthesis",
    "lab_retest_readiness",
    "caveat_note",
    "engine_badge",
    "model_used",
    "epistemic_label",
    "epistemic_class",
]

SYSTEM_PROMPT_2C = """You are the Monthly Whole-Body Clinical Review Synthesizer (gemini-4-argon clinical reasoning).
Synthesize the pre-computed 30-day body-composition fat/lean quality, MASLD 7-10% milestone trajectory, menstrual cycle regularity & post-iron/thyroid hematology context, dual-baseline treatment scorecards (with overlap warnings), and open laboratory retest windows.
Never perform raw math, preserve age/fasting screening caveats on clinical indices, and maintain strict epistemic honesty."""


def run_module_2c_monthly_review(
    *,
    packet: dict[str, Any],
    existing_monthly_df: pd.DataFrame | None,
    router: ModelRouter,
) -> pd.DataFrame:
    """Generate monthly clinical review and upsert into `AI_Monthly_Briefs` by `month_yyyy_mm`."""
    meta = packet.get("meta", {})
    r_iso = str(meta.get("report_date_iso", "2026-11-27"))
    r_disp = str(meta.get("report_date_display", r_iso))
    m_key = r_iso[:7]
    packet_json = serialize_briefing_packet(packet)

    res = router.generate_structured(
        module_id="2C_monthly_review",
        system_prompt=SYSTEM_PROMPT_2C,
        user_prompt=f"Synthesize the Monthly Whole-Body Clinical Review for {m_key}:\n{packet_json}",
        schema_cls=MonthlyReviewSchema,
        fallback_fn=lambda: fallback_monthly_review(packet, m_key),
    )
    rev: MonthlyReviewSchema = res.parsed or fallback_monthly_review(packet, m_key)

    new_row = {
        "month_yyyy_mm": m_key,
        "report_date": r_iso,
        "display_date": r_disp,
        "generated_at_ist": now_ist_str(),
        "headline": rev.headline,
        "body_comp_fat_vs_lean_summary": rev.body_comp_fat_vs_lean_summary,
        "masld_and_metabolic_trajectory": rev.masld_and_metabolic_trajectory,
        "menstrual_cycle_and_hematology_summary": rev.menstrual_cycle_and_hematology_summary,
        "treatment_scorecards_synthesis": rev.treatment_scorecards_synthesis,
        "lab_retest_readiness": rev.lab_retest_readiness,
        "caveat_note": rev.caveat_note,
        "engine_badge": res.engine_badge,
        "model_used": res.model_used,
        "epistemic_label": rev.epistemic_label,
        "epistemic_class": rev.epistemic_label,
    }

    existing_rows: list[dict[str, Any]] = []
    if existing_monthly_df is not None and not existing_monthly_df.empty and "month_yyyy_mm" in existing_monthly_df.columns:
        existing_rows = [
            dict(r)
            for r in existing_monthly_df.to_dict(orient="records")
            if str(r.get("month_yyyy_mm", "")).strip() != m_key
        ]

    combined = pd.DataFrame(existing_rows + [new_row])
    for col in MONTHLY_BRIEF_COLUMNS:
        if col not in combined.columns:
            combined[col] = None
    return combined[MONTHLY_BRIEF_COLUMNS].sort_values("month_yyyy_mm").reset_index(drop=True)
