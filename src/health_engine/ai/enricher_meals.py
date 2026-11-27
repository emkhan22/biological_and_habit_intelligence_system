"""Module 1C: Home-Cooked Meal & Bioavailability Auditor (`enricher_meals.py` -> `Gold_Meal_Bioavailability_AI`).

Runs incrementally ONLY on newly logged dates joining `Silver_Nutrition_Log` + `Silver_Meds_Log`
+ `Gold_Event_Windows` (cached by ISO `date` so historical days are never re-billed).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from health_engine.ai.fallbacks import fallback_meal_bioavailability
from health_engine.ai.packet_builder import _clean_scalar, assert_packet_smoking_free
from health_engine.ai.router import ModelRouter
from health_engine.ai.schemas import MealBioavailabilityBatchSchema
from health_engine.config_loader import iso_to_display_date, now_ist_str, to_iso_date_str

MEAL_AI_COLUMNS: list[str] = [
    "date",
    "display_date",
    "meals_logged_cnt",
    "meds_logged_cnt",
    "fat_cofactor_assessment",
    "thyronorm_fasting_assessment",
    "late_spicy_gi_sleep_note",
    "protein_fiber_tweaks",
    "bioavailability_status",
    "engine_badge",
    "model_used",
    "audited_at_ist",
    "epistemic_label",
    "epistemic_class",
]

SYSTEM_PROMPT_1C = """You are an Indian clinical nutrition & medication-bioavailability auditor.
Given pre-computed daily macros, meal logs (Indian home-cooked meals such as phulkas, curries, fried/grilled fish, bone broth, curds, protein shakes), and medication timestamps (Thyronorm levothyroxine, D-Rise 60K cholecalciferol, Omega-3 fish oil, Magnesium glycinate, Trazer F Forte, Matilda Forte):
1. Audit dietary fat co-factor adequacy (>=10g fat) with D-Rise 60K / Omega-3 doses.
2. Audit morning Thyronorm water-only fasting gap (>=30-60 min before morning tea/breakfast).
3. Flag late/heavy/spicy evening meals relative to next-day GI symptoms or overnight sleep HR spikes.
4. Suggest 1-2 realistic Indian home-cooked protein/fiber menu tweaks using the exact pre-computed numbers.
Never perform raw arithmetic; set epistemic_label = 'Descriptive' or 'Causal_Hypothesis' without claiming causal proof."""


def _build_unprocessed_day_payloads(
    *,
    silver_nutr_log: pd.DataFrame | None,
    silver_meds_log: pd.DataFrame | None,
    gold_daily: pd.DataFrame | None,
    gold_event_windows: pd.DataFrame | None,
    cached_dates: set[str],
) -> list[dict[str, Any]]:
    nutr_dates: set[str] = set()
    meals_by_date: dict[str, list[dict[str, Any]]] = {}
    if silver_nutr_log is not None and not silver_nutr_log.empty:
        for r in silver_nutr_log.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date")) or ""
            if ds:
                nutr_dates.add(ds)
                meals_by_date.setdefault(ds, []).append(
                    {
                        "time_ist": _clean_scalar(r.get("time_ist")),
                        "meal_type": _clean_scalar(r.get("meal_type")),
                        "meal_name": _clean_scalar(r.get("meal_name")),
                        "calories_kcal": _clean_scalar(r.get("calories_kcal")),
                        "protein_g": _clean_scalar(r.get("protein_g")),
                        "fat_g": _clean_scalar(r.get("fat_g")),
                        "fiber_g": _clean_scalar(r.get("fiber_g")),
                    }
                )

    med_dates: set[str] = set()
    meds_by_date: dict[str, list[dict[str, Any]]] = {}
    if silver_meds_log is not None and not silver_meds_log.empty:
        for r in silver_meds_log.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date") or r.get("Date")) or ""
            if ds:
                med_dates.add(ds)
                meds_by_date.setdefault(ds, []).append(
                    {
                        "time_ist": _clean_scalar(r.get("time_ist") or r.get("Time")),
                        "med": _clean_scalar(r.get("Med_or_Supplement")),
                        "dose": _clean_scalar(r.get("Dose")),
                        "status": _clean_scalar(r.get("Status")),
                        "timing_context": _clean_scalar(r.get("Timing_Context")),
                    }
                )

    daily_map: dict[str, dict[str, Any]] = {}
    if gold_daily is not None and not gold_daily.empty and "date" in gold_daily.columns:
        for r in gold_daily.to_dict(orient="records"):
            ds = str(r.get("date", ""))[:10]
            if ds:
                daily_map[ds] = r

    ev_map: dict[str, dict[str, Any]] = {}
    if gold_event_windows is not None and not gold_event_windows.empty and "date" in gold_event_windows.columns:
        for r in gold_event_windows.to_dict(orient="records"):
            ds = str(r.get("date", ""))[:10]
            if ds:
                ev_map[ds] = r

    target_dates = sorted((nutr_dates | med_dates) - cached_dates)
    payloads: list[dict[str, Any]] = []
    for ds in target_dates:
        drow = daily_map.get(ds, {})
        erow = ev_map.get(ds, {})
        meals = meals_by_date.get(ds, [])
        meds = meds_by_date.get(ds, [])
        payloads.append(
            {
                "date": ds,
                "display_date": iso_to_display_date(ds) or ds,
                "meals_count": len(meals),
                "meds_count": len(meds),
                "calories_kcal": _clean_scalar(drow.get("calories_kcal")),
                "protein_g": _clean_scalar(drow.get("protein_g")),
                "fat_g": _clean_scalar(drow.get("fat_g")),
                "fiber_g": _clean_scalar(drow.get("fiber_g")),
                "hydration_liters": _clean_scalar(drow.get("hydration_liters")),
                "late_kcal_after_21": _clean_scalar(drow.get("late_kcal_after_21")),
                "thyronorm_fasting_gap_min": _clean_scalar(erow.get("thyronorm_fasting_gap_min")),
                "fat_cofactor_valid_10g": _clean_scalar(erow.get("fat_soluble_cofactor_valid_10g")),
                "following_night_spikes_per_hr": _clean_scalar(
                    erow.get("out_following_night_spikes_per_asleep_hr")
                ),
                "shot_day_int": _clean_scalar(drow.get("shot_day_int")),
                "meals": meals,
                "meds": meds,
            }
        )
    return payloads


def run_module_1c_meal_auditor(
    *,
    silver_nutr_log: pd.DataFrame | None,
    silver_meds_log: pd.DataFrame | None,
    gold_daily: pd.DataFrame | None,
    gold_event_windows: pd.DataFrame | None,
    existing_ai_df: pd.DataFrame | None,
    router: ModelRouter,
    batch_size: int = 14,
) -> pd.DataFrame:
    """Incrementally audit newly logged nutrition/med dates into `Gold_Meal_Bioavailability_AI`."""
    existing_rows: list[dict[str, Any]] = []
    cached_dates: set[str] = set()
    if existing_ai_df is not None and not existing_ai_df.empty and "date" in existing_ai_df.columns:
        for r in existing_ai_df.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date")) or ""
            if ds:
                cached_dates.add(ds)
                existing_rows.append(dict(r))

    day_payloads = _build_unprocessed_day_payloads(
        silver_nutr_log=silver_nutr_log,
        silver_meds_log=silver_meds_log,
        gold_daily=gold_daily,
        gold_event_windows=gold_event_windows,
        cached_dates=cached_dates,
    )

    if not day_payloads:
        if not existing_rows:
            return pd.DataFrame(columns=MEAL_AI_COLUMNS)
        out = pd.DataFrame(existing_rows)
        for col in MEAL_AI_COLUMNS:
            if col not in out.columns:
                out[col] = None
        return out[MEAL_AI_COLUMNS].sort_values("date").reset_index(drop=True)

    new_rows: list[dict[str, Any]] = []
    ts_now = now_ist_str()

    for start_i in range(0, len(day_payloads), batch_size):
        batch = day_payloads[start_i : start_i + batch_size]
        assert_packet_smoking_free(batch, path="module_1c_batch")
        prompt_json = json.dumps(batch, ensure_ascii=False, indent=2)

        res = router.generate_structured(
            module_id="1C_meal_bioavailability",
            system_prompt=SYSTEM_PROMPT_1C,
            user_prompt=f"Audit home-cooked meal & supplement bioavailability for each date:\n{prompt_json}",
            schema_cls=MealBioavailabilityBatchSchema,
            fallback_fn=lambda b=batch: fallback_meal_bioavailability(b),
        )

        if res.status == "Skipped_NonEssential_Above_450_INR" or res.parsed is None:
            # Budget >= ₹450 skips non-essential 1C without wiping existing cached rows
            break

        parsed_batch: MealBioavailabilityBatchSchema = res.parsed
        by_date = {it.date[:10]: it for it in parsed_batch.items}
        for rec in batch:
            ds = rec["date"]
            item = by_date.get(ds) or fallback_meal_bioavailability([rec]).items[0]
            new_rows.append(
                {
                    "date": ds,
                    "display_date": rec["display_date"],
                    "meals_logged_cnt": rec["meals_count"],
                    "meds_logged_cnt": rec["meds_count"],
                    "fat_cofactor_assessment": item.fat_cofactor_assessment,
                    "thyronorm_fasting_assessment": item.thyronorm_fasting_assessment,
                    "late_spicy_gi_sleep_note": item.late_spicy_gi_sleep_note,
                    "protein_fiber_tweaks": item.protein_fiber_tweaks,
                    "bioavailability_status": item.bioavailability_status,
                    "engine_badge": res.engine_badge,
                    "model_used": res.model_used,
                    "audited_at_ist": ts_now,
                    "epistemic_label": item.epistemic_label,
                    "epistemic_class": item.epistemic_label,
                }
            )

    if not existing_rows and not new_rows:
        return pd.DataFrame(columns=MEAL_AI_COLUMNS)
    combined = pd.DataFrame(existing_rows + new_rows)
    for col in MEAL_AI_COLUMNS:
        if col not in combined.columns:
            combined[col] = None
    return combined[MEAL_AI_COLUMNS].drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(
        drop=True
    )
