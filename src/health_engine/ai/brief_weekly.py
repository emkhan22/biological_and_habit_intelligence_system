"""Module 2B: Friday Weekly Deep Review (`brief_weekly.py` -> `AI_Weekly_Briefs`).

Supports:
- Current completed or active Thu-Wed week window, AND
- `--backfill-weeks` CLI flag generating historical Thu-Wed weeks since `2026-10-21`
  (skipping already-cached `week_start_date` rows unless `force_refresh=True`).
"""

from __future__ import annotations

from datetime import date, timedelta
import json
from typing import Any

import pandas as pd

from health_engine.ai.fallbacks import fallback_weekly_review
from health_engine.ai.packet_builder import build_ai_briefing_packet, serialize_briefing_packet
from health_engine.ai.router import ModelRouter
from health_engine.ai.schemas import WeeklyReviewSchema
from health_engine.config_loader import iso_to_display_date, now_ist_str

WEEKLY_BRIEF_COLUMNS: list[str] = [
    "week_id",
    "week_start_date",
    "display_start_date",
    "week_end_date",
    "display_end_date",
    "generated_at_ist",
    "headline",
    "dose_working",
    "week_influencers",
    "habit_review_improved_vs_slipped",
    "next_week_priorities_json",
    "clinical_and_bioavailability_notes",
    "caveat_note",
    "engine_badge",
    "model_used",
    "epistemic_label",
    "epistemic_class",
]

SYSTEM_PROMPT_2B = """You are the Friday Weekly Deep Review Synthesizer (Thu-Wed treatment week) for an Indian user on Tirzepatide + metabolic recovery therapy.
Synthesize the exact pre-computed numbers in AI_Briefing_Packet into four rigorous sections:
1. dose_working: weekly Tirzepatide response, hunger/food-noise wave, confounder-aware weight-interval / EMA trend progress, and BMR/TDEE calorie balance audit consistency (`Within_Expected_Band` vs `Potential_Under_Logging` / `Unclear_Water_Cycle_Or_Log_Gap`).
2. week_influencers: sleep/bedtime phenotypes, activity consistency, cycle phase, meal timing, GI/constipation flags, and sweet-intake -> hunger patterns (always noting reverse-causality risk if prior hunger/sleep debt preceded sweets).
3. habit_review_improved_vs_slipped: concrete wins vs friction points, Peak vs Trough habit carryover (and Post-GLP-1 durability if `Off_GLP1_Transition` is active), plus 2-3 prioritized next-week habits.
4. clinical_and_bioavailability_notes: Thyronorm fasting gap, fat-soluble supplement pairing, dual-baseline treatment scorecards, and overlap warnings.
Never do raw math; never upgrade observational correlations or ML weights to causal proof."""


def compute_thu_wed_windows(
    max_date_iso: str,
    *,
    backfill_since: str = "2026-10-21",
    include_backfill: bool = False,
) -> list[tuple[str, str]]:
    """Return `(thu_start_iso, wed_end_iso)` tuples anchored to Thursday starts."""
    end_d = date.fromisoformat(max_date_iso[:10])
    # Thursday is weekday() == 3
    days_since_thu = (end_d.weekday() - 3) % 7
    latest_thu = end_d - timedelta(days=days_since_thu)
    latest_wed = min(end_d, latest_thu + timedelta(days=6))

    if not include_backfill:
        return [(latest_thu.isoformat(), latest_wed.isoformat())]

    anchor_d = date.fromisoformat(backfill_since[:10])
    # Align first window to Thursday on or immediately before anchor_d (or Oct 22 Thu)
    offset_to_thu = (3 - anchor_d.weekday()) % 7
    cur_thu = anchor_d + timedelta(days=offset_to_thu)
    if cur_thu > anchor_d:
        # Include the partial/opening treatment days starting from anchor_d's week if needed
        cur_thu = anchor_d - timedelta(days=(anchor_d.weekday() - 3) % 7)

    windows: list[tuple[str, str]] = []
    while cur_thu <= end_d:
        w_end = min(end_d, cur_thu + timedelta(days=6))
        windows.append((cur_thu.isoformat(), w_end.isoformat()))
        cur_thu = cur_thu + timedelta(days=7)
    return windows


def run_module_2b_weekly_review(
    *,
    tables: dict[str, pd.DataFrame],
    silver_tables: dict[str, pd.DataFrame] | None,
    existing_weekly_df: pd.DataFrame | None,
    router: ModelRouter,
    backfill_weeks: bool = False,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Generate current Thu-Wed review (and optional historical backfill since 2026-10-21)."""
    gold_daily = tables.get("Gold_Daily_Unified", pd.DataFrame())
    max_iso = (
        str(gold_daily["date"].max())[:10]
        if not gold_daily.empty and "date" in gold_daily.columns
        else "2026-11-27"
    )

    existing_map: dict[str, dict[str, Any]] = {}
    if existing_weekly_df is not None and not existing_weekly_df.empty and "week_start_date" in existing_weekly_df.columns:
        for r in existing_weekly_df.to_dict(orient="records"):
            ws = str(r.get("week_start_date", ""))[:10]
            if ws:
                existing_map[ws] = dict(r)

    windows = compute_thu_wed_windows(max_iso, include_backfill=backfill_weeks)
    ts_now = now_ist_str()

    for idx, (w_start, w_end) in enumerate(windows, start=1):
        if (not force_refresh) and (w_start in existing_map) and backfill_weeks and w_end != windows[-1][1]:
            continue

        week_packet = build_ai_briefing_packet(tables, silver_tables, as_of_date=w_end)
        disp_s = iso_to_display_date(w_start) or w_start
        disp_e = iso_to_display_date(w_end) or w_end
        w_label = f"{disp_s} to {disp_e}"
        packet_json = serialize_briefing_packet(week_packet)

        res = router.generate_structured(
            module_id="2B_weekly_review",
            system_prompt=SYSTEM_PROMPT_2B,
            user_prompt=f"Generate the Friday Weekly Deep Review for Thu-Wed window {w_label}:\n{packet_json}",
            schema_cls=WeeklyReviewSchema,
            fallback_fn=lambda p=week_packet, lbl=w_label: fallback_weekly_review(p, lbl),
        )
        rev: WeeklyReviewSchema = res.parsed or fallback_weekly_review(week_packet, w_label)

        existing_map[w_start] = {
            "week_id": f"WK-{w_start.replace('-', '')}",
            "week_start_date": w_start,
            "display_start_date": disp_s,
            "week_end_date": w_end,
            "display_end_date": disp_e,
            "generated_at_ist": ts_now,
            "headline": rev.headline,
            "dose_working": rev.dose_working,
            "week_influencers": rev.week_influencers,
            "habit_review_improved_vs_slipped": rev.habit_review_improved_vs_slipped,
            "next_week_priorities_json": json.dumps(rev.next_week_priorities, ensure_ascii=False),
            "clinical_and_bioavailability_notes": rev.clinical_and_bioavailability_notes,
            "caveat_note": rev.caveat_note,
            "engine_badge": res.engine_badge,
            "model_used": res.model_used,
            "epistemic_label": rev.epistemic_label,
            "epistemic_class": rev.epistemic_label,
        }

    if not existing_map:
        return pd.DataFrame(columns=WEEKLY_BRIEF_COLUMNS)
    out = pd.DataFrame(list(existing_map.values()))
    for col in WEEKLY_BRIEF_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[WEEKLY_BRIEF_COLUMNS].sort_values("week_start_date").reset_index(drop=True)
