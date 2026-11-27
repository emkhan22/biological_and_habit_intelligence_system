"""Module 3B: On-Demand Specialist Doctor Visit Prep (`doctor_prep.py` -> `AI_Doctor_Briefs`).

Supports `--specialty` in:
`Endocrinology` | `General_Medicine` | `Gynaecology` | `Orthopaedics_Physio` | `Whole_Body`.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from health_engine.ai.fallbacks import fallback_doctor_prep
from health_engine.ai.packet_builder import serialize_briefing_packet
from health_engine.ai.router import ModelRouter
from health_engine.ai.schemas import DoctorVisitPrepSchema
from health_engine.config_loader import now_ist_str

VALID_SPECIALTIES: tuple[str, ...] = (
    "Endocrinology",
    "General_Medicine",
    "Gynaecology",
    "Orthopaedics_Physio",
    "Whole_Body",
)

DOCTOR_BRIEF_COLUMNS: list[str] = [
    "brief_id",
    "report_date",
    "display_date",
    "specialty",
    "generated_at_ist",
    "printable_clinical_summary",
    "lab_delta_checklist_json",
    "treatment_scorecard_highlights",
    "top_5_questions_json",
    "verified_citations_json",
    "caveat_note",
    "engine_badge",
    "model_used",
    "epistemic_label",
    "epistemic_class",
]

SYSTEM_PROMPT_3B = """You are a Specialist Physician Visit Preparation Assistant (gemini-4-argon with Google Search grounding).
Prepare a concise, physician-friendly clinical summary tailored to the requested specialty, a longitudinal lab-delta & retest checklist, dual-baseline treatment scorecard highlights (honoring overlap_warning flags and age/fasting index caveats), and the top 5 prioritized questions for the patient to ask their specialist.
Set epistemic_label = 'Screening_Context' and never issue autonomous prescribing orders."""


def run_module_3b_doctor_prep(
    *,
    specialty: str,
    packet: dict[str, Any],
    existing_doc_df: pd.DataFrame | None,
    router: ModelRouter,
) -> pd.DataFrame:
    """Generate specialist consultation brief and upsert into `AI_Doctor_Briefs`."""
    spec_norm = str(specialty or "Whole_Body").strip()
    if spec_norm not in VALID_SPECIALTIES:
        raise ValueError(
            f"Unsupported specialty '{spec_norm}'. Expected one of {VALID_SPECIALTIES}."
        )

    meta = packet.get("meta", {})
    r_iso = str(meta.get("report_date_iso", "2026-11-27"))
    r_disp = str(meta.get("report_date_display", r_iso))
    brief_id = f"DOC-{r_iso.replace('-', '')}-{spec_norm}"
    packet_json = serialize_briefing_packet(packet)

    res = router.generate_structured(
        module_id="3B_doctor_visit_prep",
        system_prompt=SYSTEM_PROMPT_3B,
        user_prompt=(
            f"Prepare a {spec_norm} consultation brief from this validated AI_Briefing_Packet:\n{packet_json}"
        ),
        schema_cls=DoctorVisitPrepSchema,
        fallback_fn=lambda: fallback_doctor_prep(packet, spec_norm),
    )
    doc: DoctorVisitPrepSchema = res.parsed or fallback_doctor_prep(packet, spec_norm)
    cites_list = [{"title": c.title, "url": c.url} for c in res.verified_citations]

    new_row = {
        "brief_id": brief_id,
        "report_date": r_iso,
        "display_date": r_disp,
        "specialty": spec_norm,
        "generated_at_ist": now_ist_str(),
        "printable_clinical_summary": doc.printable_clinical_summary,
        "lab_delta_checklist_json": json.dumps(doc.lab_delta_checklist, ensure_ascii=False),
        "treatment_scorecard_highlights": doc.treatment_scorecard_highlights,
        "top_5_questions_json": json.dumps(
            [q.model_dump() for q in doc.top_5_questions], ensure_ascii=False
        ),
        "verified_citations_json": json.dumps(cites_list, ensure_ascii=False),
        "caveat_note": doc.caveat_note,
        "engine_badge": res.engine_badge,
        "model_used": res.model_used,
        "epistemic_label": doc.epistemic_label,
        "epistemic_class": doc.epistemic_label,
    }

    existing_rows: list[dict[str, Any]] = []
    if existing_doc_df is not None and not existing_doc_df.empty and "brief_id" in existing_doc_df.columns:
        existing_rows = [
            dict(r)
            for r in existing_doc_df.to_dict(orient="records")
            if str(r.get("brief_id", "")).strip() != brief_id
        ]

    combined = pd.DataFrame(existing_rows + [new_row])
    for col in DOCTOR_BRIEF_COLUMNS:
        if col not in combined.columns:
            combined[col] = None
    return combined[DOCTOR_BRIEF_COLUMNS].sort_values(["report_date", "specialty"]).reset_index(drop=True)
