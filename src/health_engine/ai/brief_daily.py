"""Module 2A: Daily Holistic Brief & Coaching Nudges (`brief_daily.py` -> `AI_Insights_Cache`)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from health_engine.ai.fallbacks import fallback_daily_brief
from health_engine.ai.packet_builder import serialize_briefing_packet
from health_engine.ai.router import ModelRouter
from health_engine.ai.schemas import DailyBriefSchema
from health_engine.config_loader import now_ist_str

DAILY_BRIEF_COLUMNS: list[str] = [
    "report_date",
    "display_date",
    "generated_at_ist",
    "overview_one_liner",
    "headline",
    "inside_your_body_summary",
    "top_3_patterns_json",
    "action_nudges_json",
    "engine_badge",
    "model_used",
    "epistemic_label",
    "epistemic_class",
]

SYSTEM_PROMPT_2A = """You are the Daily Holistic Health Intelligence Synthesizer for an Indian user managing Tirzepatide GLP-1/GIP therapy, MASLD fatty liver reversal, post-IV-Iron & Vitamin D recovery, and Thyronorm equilibration.
Rules:
1. Never perform arithmetic—quote ONLY the pre-computed metrics, Z-scores, readiness pillars, and ML/statistical summaries inside AI_Briefing_Packet.
2. Never zero-fill or scold unlogged wearable/nutrition days; respect Not_Worn and Not_Logged tags.
3. Enforce strict epistemic honesty: tag every pattern and nudge with 'Measured', 'Descriptive', 'Correlation', 'Prediction', 'Causal_Hypothesis', or 'Screening_Context', and NEVER phrase correlations or Ridge/RandomForest/SHAP weights as causal proof.
4. Use DD-MM-YYYY for display dates and IST context."""


def run_module_2a_daily_brief(
    *,
    packet: dict[str, Any],
    existing_cache_df: pd.DataFrame | None,
    router: ModelRouter,
) -> pd.DataFrame:
    """Generate today's holistic brief and upsert into `AI_Insights_Cache` by `report_date`."""
    packet_json = serialize_briefing_packet(packet)
    meta = packet.get("meta", {})
    r_iso = str(meta.get("report_date_iso", ""))
    r_disp = str(meta.get("report_date_display", r_iso))

    res = router.generate_structured(
        module_id="2A_daily_brief",
        system_prompt=SYSTEM_PROMPT_2A,
        user_prompt=f"Synthesize today's Daily Holistic Brief from this validated AI_Briefing_Packet:\n{packet_json}",
        schema_cls=DailyBriefSchema,
        fallback_fn=lambda: fallback_daily_brief(packet),
    )
    brief: DailyBriefSchema = res.parsed or fallback_daily_brief(packet)

    new_row = {
        "report_date": r_iso,
        "display_date": r_disp,
        "generated_at_ist": now_ist_str(),
        "overview_one_liner": brief.overview_one_liner,
        "headline": brief.headline,
        "inside_your_body_summary": brief.inside_your_body_summary,
        "top_3_patterns_json": json.dumps(
            [p.model_dump() for p in brief.top_3_patterns], ensure_ascii=False
        ),
        "action_nudges_json": json.dumps(
            [n.model_dump() for n in brief.action_nudges], ensure_ascii=False
        ),
        "engine_badge": res.engine_badge,
        "model_used": res.model_used,
        "epistemic_label": brief.epistemic_label,
        "epistemic_class": brief.epistemic_label,
    }

    existing_rows: list[dict[str, Any]] = []
    if existing_cache_df is not None and not existing_cache_df.empty and "report_date" in existing_cache_df.columns:
        existing_rows = [
            dict(r)
            for r in existing_cache_df.to_dict(orient="records")
            if str(r.get("report_date", ""))[:10] != r_iso
        ]

    combined = pd.DataFrame(existing_rows + [new_row])
    for col in DAILY_BRIEF_COLUMNS:
        if col not in combined.columns:
            combined[col] = None
    return combined[DAILY_BRIEF_COLUMNS].sort_values("report_date").reset_index(drop=True)
