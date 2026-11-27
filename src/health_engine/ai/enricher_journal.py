"""Module 1B: Micro Journal Hinglish/English Semantic Enricher (`enricher_journal.py` -> `Gold_Journal_Themes_AI`).

Runs incrementally ONLY on new/unenriched rows of non-smoking `Silver_Journal_Mood`
and `Silver_Symptoms_Log` (cached by `event_id` so historical rows are never re-billed).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from health_engine.ai.fallbacks import fallback_journal_enrichment
from health_engine.ai.packet_builder import assert_packet_smoking_free
from health_engine.ai.router import ModelRouter
from health_engine.ai.schemas import JournalEnrichmentBatchSchema
from health_engine.config_loader import iso_to_display_date, now_ist_str, to_iso_date_str
from health_engine.ml.journal_nlp import _sanitize_event_id

JOURNAL_AI_COLUMNS: list[str] = [
    "event_id",
    "source_tab",
    "date",
    "display_date",
    "time_ist",
    "detected_themes_ai",
    "trigger_context_ai",
    "pain_radiation_note_ai",
    "sentiment_polarity_ai",
    "engine_badge",
    "model_used",
    "enriched_at_ist",
    "epistemic_label",
    "epistemic_class",
]

SYSTEM_PROMPT_1B = """You are a clinical & behavioral health NLP tagger for an Indian user's non-smoking Micro Journal (English + Hinglish notes).
Extract structured multi-label themes, concise trigger context, pain-radiation notes (e.g., right shoulder/neck/arm ache or calf cramp, else 'None'), and nuanced sentiment polarity ('Positive' | 'Negative' | 'Mixed' | 'Neutral') for each event_id.
Canonical themes to prioritize when applicable: Work_Overwhelm, Late_Night_Work, Weigh_In_Anxiety, Appetite_Suppression_Meal_Skip, GI_Distress, Musculoskeletal_Pain, Nocturnal_Cramp, Missed_Walk_Guilt, Positive_Momentum, Fatigue_Sleep_Debt.
Epistemic contract: set epistemic_label = 'Descriptive' on every item and never make causal medical claims."""


def _collect_journal_candidates(
    silver_journal_mood: pd.DataFrame | None,
    silver_symptoms: pd.DataFrame | None,
    det_themes_df: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    det_map: dict[str, tuple[str, str]] = {}
    if det_themes_df is not None and not det_themes_df.empty and "event_id" in det_themes_df.columns:
        for r in det_themes_df.to_dict(orient="records"):
            eid = str(r.get("event_id", "")).strip()
            if eid:
                det_map[eid] = (
                    str(r.get("detected_themes", "") or ""),
                    str(r.get("sentiment_polarity_simple", "Neutral") or "Neutral"),
                )

    candidates: list[dict[str, Any]] = []

    if silver_journal_mood is not None and not silver_journal_mood.empty:
        for idx, r in enumerate(silver_journal_mood.to_dict(orient="records")):
            ds = to_iso_date_str(r.get("date") or r.get("Date")) or ""
            if not ds:
                continue
            tm = str(r.get("time_ist") or r.get("Time") or "").strip()
            raw_eid = r.get("Event_ID") or r.get("event_id")
            eid = _sanitize_event_id(raw_eid, "JM", ds, tm, idx)
            det_th, det_sent = det_map.get(eid, ("", "Neutral"))
            candidates.append(
                {
                    "event_id": eid,
                    "source_tab": "Journal_Mood",
                    "date": ds,
                    "display_date": iso_to_display_date(ds) or ds,
                    "time_ist": tm,
                    "mood_tags": str(r.get("Mood_Tags") or ""),
                    "context_tag": str(r.get("Context_Tag") or ""),
                    "symptom": "",
                    "raw_text": str(r.get("Summary") or r.get("Notes") or ""),
                    "deterministic_themes": det_th,
                    "deterministic_sentiment": det_sent,
                }
            )

    if silver_symptoms is not None and not silver_symptoms.empty:
        for idx, r in enumerate(silver_symptoms.to_dict(orient="records")):
            ds = to_iso_date_str(r.get("date") or r.get("Date")) or ""
            if not ds:
                continue
            tm = str(r.get("time_ist") or r.get("Time") or "").strip()
            raw_eid = r.get("Event_ID") or r.get("event_id")
            eid = _sanitize_event_id(raw_eid, "SYM", ds, tm, idx)
            det_th, det_sent = det_map.get(eid, ("", "Negative"))
            candidates.append(
                {
                    "event_id": eid,
                    "source_tab": "Symptoms_Log",
                    "date": ds,
                    "display_date": iso_to_display_date(ds) or ds,
                    "time_ist": tm,
                    "mood_tags": "",
                    "context_tag": str(r.get("Context_Tag") or ""),
                    "symptom": str(r.get("Normalized_Symptom") or r.get("Symptom") or ""),
                    "raw_text": str(r.get("Notes") or ""),
                    "deterministic_themes": det_th,
                    "deterministic_sentiment": det_sent,
                }
            )

    return candidates


def run_module_1b_journal_enricher(
    *,
    silver_journal_mood: pd.DataFrame | None,
    silver_symptoms: pd.DataFrame | None,
    det_themes_df: pd.DataFrame | None,
    existing_ai_df: pd.DataFrame | None,
    router: ModelRouter,
    batch_size: int = 25,
) -> pd.DataFrame:
    """Incrementally enrich new non-smoking journal/symptom rows into `Gold_Journal_Themes_AI`."""
    existing_rows: list[dict[str, Any]] = []
    cached_ids: set[str] = set()
    if existing_ai_df is not None and not existing_ai_df.empty and "event_id" in existing_ai_df.columns:
        for r in existing_ai_df.to_dict(orient="records"):
            eid = str(r.get("event_id", "")).strip()
            if eid:
                cached_ids.add(eid)
                existing_rows.append(dict(r))

    all_candidates = _collect_journal_candidates(silver_journal_mood, silver_symptoms, det_themes_df)
    new_candidates = [c for c in all_candidates if c["event_id"] not in cached_ids]

    if not new_candidates:
        if not existing_rows:
            return pd.DataFrame(columns=JOURNAL_AI_COLUMNS)
        out_df = pd.DataFrame(existing_rows)
        for col in JOURNAL_AI_COLUMNS:
            if col not in out_df.columns:
                out_df[col] = None
        return out_df[JOURNAL_AI_COLUMNS].sort_values(["date", "time_ist"]).reset_index(drop=True)

    new_enriched_rows: list[dict[str, Any]] = []
    ts_now = now_ist_str()

    for start_i in range(0, len(new_candidates), batch_size):
        batch = new_candidates[start_i : start_i + batch_size]
        assert_packet_smoking_free(batch, path="module_1b_batch")
        user_payload = json.dumps(batch, ensure_ascii=False, indent=2)

        res = router.generate_structured(
            module_id="1B_journal_nlp",
            system_prompt=SYSTEM_PROMPT_1B,
            user_prompt=f"Enrich every item in this Micro Journal batch:\n{user_payload}",
            schema_cls=JournalEnrichmentBatchSchema,
            fallback_fn=lambda b=batch: fallback_journal_enrichment(b),
        )

        parsed_batch: JournalEnrichmentBatchSchema = res.parsed or fallback_journal_enrichment(batch)
        by_id = {it.event_id: it for it in parsed_batch.items}

        for cand in batch:
            eid = cand["event_id"]
            item = by_id.get(eid)
            if item is None:
                item = fallback_journal_enrichment([cand]).items[0]
            new_enriched_rows.append(
                {
                    "event_id": eid,
                    "source_tab": cand["source_tab"],
                    "date": cand["date"],
                    "display_date": cand["display_date"],
                    "time_ist": cand["time_ist"],
                    "detected_themes_ai": item.detected_themes_csv,
                    "trigger_context_ai": item.trigger_context,
                    "pain_radiation_note_ai": item.pain_radiation_note,
                    "sentiment_polarity_ai": item.sentiment_polarity,
                    "engine_badge": res.engine_badge,
                    "model_used": res.model_used,
                    "enriched_at_ist": ts_now,
                    "epistemic_label": item.epistemic_label,
                    "epistemic_class": item.epistemic_label,
                }
            )

    combined = pd.DataFrame(existing_rows + new_enriched_rows)
    for col in JOURNAL_AI_COLUMNS:
        if col not in combined.columns:
            combined[col] = None
    return combined[JOURNAL_AI_COLUMNS].drop_duplicates(subset=["event_id"], keep="last").sort_values(
        ["date", "time_ist"]
    ).reset_index(drop=True)
