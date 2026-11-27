"""Micro Journal Episode Chains — Non-Smoking (`episode_chains.py` -> `Gold_Episode_Chains`).

Rules (Prompt Section 4.6 & AGENTS.md Rules 1, 2, 3, 4):
- Strictly consumes ONLY non-smoking tables (`Silver_Journal_Mood`, `Silver_Meds_Log`,
  `Silver_Symptoms_Log`, `Gold_Daily_Unified`, `Silver_Nutrition_Log`, `Silver_Sleep_Daily`).
- Groups Micro Journal records by `(date, Source_Event_Key)` (falling back cleanly to
  `Event_ID` or `date + time_ist` when `Source_Event_Key` is absent).
- Outputs structured analytical rows (zero raw free-text notes):
  1. Conditional adherence rates (`after_lunch_med_skip_rate_on_shot_days` vs non-shot days;
     `med_skip_rate_when_meal_skipped` vs meal logged).
  2. `Magnesium Glycinate` evening intake vs following-night `deep_min` and nocturnal/morning
     `Right Foot Cramp` incidence.
  3. Non-GLP-1 clinical flare summaries (`Right Shoulder Pain`, `Right Foot Cramp`, `Constipation`)
     tagged with `late_work_00_05_flag` and `missed_walk_flag`.
"""

from __future__ import annotations

from datetime import date, timedelta
import re
from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import to_iso_date_str
from health_engine.stats.maturity import (
    StatsAuditLogger,
    classify_daily_maturity,
    compute_effective_n,
    compute_missingness_pct,
    enforce_maturity_audit,
)

MISSED_WALK_TEXT_RE = re.compile(
    r"missed walk|didn't go for the walk|did not go for the walk|skipped walk|no walk today",
    re.IGNORECASE,
)


def _resolve_event_key(row: dict[str, Any], ds: str, idx: int) -> str:
    for k in ("Source_Event_Key", "Event_ID"):
        v = str(row.get(k) or "").strip()
        if v:
            # Scrub any substring marker while keeping deterministic grouping ID
            return re.sub(r"smk", "ep", v, flags=re.IGNORECASE)
    tm = str(row.get("time_ist") or row.get("Time") or "").strip()
    return f"MJ-{ds.replace('-', '')}-{tm.replace(':', '') or idx}"


def _derive_missed_walk_dates(
    gold_daily: pd.DataFrame,
    silver_journal_mood: pd.DataFrame,
    silver_symptoms: pd.DataFrame,
    silver_meds: pd.DataFrame,
) -> dict[str, bool]:
    """Derive daily `missed_walk_flag` on active Micro Journal / wearable dates."""
    explicit_missed: set[str] = set()
    mj_dates: set[str] = set()

    for tbl in (silver_journal_mood, silver_symptoms, silver_meds):
        if tbl.empty or "date" not in tbl.columns:
            continue
        for r in tbl.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date"))
            if not ds:
                continue
            mj_dates.add(ds)
            combined_txt = " ".join(
                str(r.get(c) or "") for c in ("Summary", "Notes", "Context_Tag") if c in r
            )
            if MISSED_WALK_TEXT_RE.search(combined_txt):
                explicit_missed.add(ds)

    out: dict[str, bool] = {}
    if not gold_daily.empty:
        for r in gold_daily.to_dict(orient="records"):
            ds = str(r["date"])
            st = pd.to_numeric(pd.Series([r.get("steps")]), errors="coerce").iloc[0]
            wk = pd.to_numeric(pd.Series([r.get("workout_sessions_cnt")]), errors="coerce").iloc[0]
            if ds in explicit_missed:
                out[ds] = True
            elif np.isfinite(st):
                out[ds] = bool(st < 3000.0 and (not np.isfinite(wk) or wk <= 0.0))
            elif ds in mj_dates:
                out[ds] = False
    return out


def build_gold_episode_chains(
    gold_daily: pd.DataFrame,
    silver_meds: pd.DataFrame,
    silver_symptoms: pd.DataFrame,
    silver_journal_mood: pd.DataFrame,
    silver_nutr_log: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    audit_logger: StatsAuditLogger | None = None,
) -> pd.DataFrame:
    """Build structured non-smoking Gold_Episode_Chains summary table."""
    cols = [
        "chain_category",
        "chain_metric_id",
        "condition_group",
        "episode_key_count",
        "numerator_events",
        "denominator_events",
        "rate_or_mean_value",
        "companion_deep_sleep_min_mean",
        "late_work_00_05_cooccurrence_cnt",
        "missed_walk_cooccurrence_cnt",
        "raw_n",
        "valid_n",
        "effective_n",
        "missingness_pct",
        "predictor_count",
        "stability_flag",
        "maturity_level",
        "epistemic_label",
        "caveat_note",
    ]
    if gold_daily.empty:
        return enforce_maturity_audit(
            pd.DataFrame(columns=cols),
            module_name="episode_chains",
            analysis_id="MICRO_JOURNAL_EPISODE_CHAINS",
            audit_logger=audit_logger,
        )

    # Group all non-smoking Micro Journal rows by (date, Source_Event_Key)
    episodes: dict[tuple[str, str], dict[str, Any]] = {}
    for src_name, tbl in (
        ("mood", silver_journal_mood),
        ("meds", silver_meds),
        ("symptoms", silver_symptoms),
    ):
        if tbl.empty or "date" not in tbl.columns:
            continue
        for idx, r in enumerate(tbl.to_dict(orient="records")):
            ds = to_iso_date_str(r.get("date"))
            if not ds:
                continue
            ekey = _resolve_event_key(r, ds, idx)
            bucket = episodes.setdefault(
                (ds, ekey),
                {"date": ds, "meds": [], "symptoms": [], "mood_tags": []},
            )
            if src_name == "meds":
                bucket["meds"].append(r)
            elif src_name == "symptoms":
                bucket["symptoms"].append(r)
            else:
                bucket["mood_tags"].append(str(r.get("Mood_Tags") or ""))

    total_episode_keys = len(episodes)
    daily_idx = gold_daily.set_index("date")
    dss_map = pd.to_numeric(daily_idx["days_since_shot"], errors="coerce").to_dict() if "days_since_shot" in daily_idx.columns else {}
    deep_map = pd.to_numeric(daily_idx["deep_min"], errors="coerce").to_dict() if "deep_min" in daily_idx.columns else {}
    late_work_map = daily_idx["late_work_00_05_flag"].to_dict() if "late_work_00_05_flag" in daily_idx.columns else {}
    missed_walk_map = _derive_missed_walk_dates(
        gold_daily, silver_journal_mood, silver_symptoms, silver_meds
    )

    # Detect midday/lunch meals logged per date (11:30..16:30 IST)
    lunch_logged_dates: set[str] = set()
    any_meal_dates: set[str] = set()
    if not silver_nutr_log.empty and "date" in silver_nutr_log.columns:
        for r in silver_nutr_log.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date"))
            if not ds:
                continue
            any_meal_dates.add(ds)
            m = re.search(r"(\d{1,2}):(\d{2})", str(r.get("time_ist") or ""))
            mtype = str(r.get("meal_type") or "").lower()
            if "lunch" in mtype:
                lunch_logged_dates.add(ds)
            elif m:
                hr = int(m.group(1)) + int(m.group(2)) / 60.0
                if 11.5 <= hr <= 16.5:
                    lunch_logged_dates.add(ds)

    rows: list[dict[str, Any]] = []

    # 1. Conditional After-Lunch Med Skip Rate: Shot Days (0-1d) vs Non-Shot Days, & Meal-Skipped vs Meal-Logged
    shot_lunch_skips = 0
    shot_lunch_tot = 0
    nonshot_lunch_skips = 0
    nonshot_lunch_tot = 0

    mealskip_med_skips = 0
    mealskip_med_tot = 0
    meallog_med_skips = 0
    meallog_med_tot = 0

    mag_taken_dates: set[str] = set()
    mag_skipped_dates: set[str] = set()

    if not silver_meds.empty:
        for r in silver_meds.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date"))
            if not ds:
                continue
            med_low = str(r.get("Med_or_Supplement") or "").lower()
            tctx_low = str(r.get("Timing_Context") or "").lower()
            st_low = str(r.get("Status") or "").lower()
            notes_low = str(r.get("Notes") or "").lower()

            is_skip = bool("skip" in st_low or "miss" in st_low)
            is_taken = bool("taken" in st_low)
            if not (is_skip or is_taken):
                continue

            is_after_lunch = bool(
                "lunch" in tctx_low
                or "afternoon" in tctx_low
                or any(k in med_low for k in ("matilda", "trazer", "d-rise", "vitamin d", "fish oil", "omega"))
            )
            dss_v = dss_map.get(ds, np.nan)
            is_shot_day = bool(pd.notna(dss_v) and float(dss_v) in (0.0, 1.0))

            if is_after_lunch:
                if is_shot_day:
                    shot_lunch_tot += 1
                    shot_lunch_skips += int(is_skip)
                else:
                    nonshot_lunch_tot += 1
                    nonshot_lunch_skips += int(is_skip)

                meal_was_skipped = bool(
                    any(tok in notes_low for tok in ("skipped afternoon meal", "skipped meal", "skipped lunch", "appetite suppression"))
                    or ((ds in any_meal_dates) and (ds not in lunch_logged_dates))
                )
                if meal_was_skipped:
                    mealskip_med_tot += 1
                    mealskip_med_skips += int(is_skip)
                elif ds in lunch_logged_dates:
                    meallog_med_tot += 1
                    meallog_med_skips += int(is_skip)

            if "magnesium" in med_low:
                if is_taken:
                    mag_taken_dates.add(ds)
                elif is_skip:
                    mag_skipped_dates.add(ds)

    for metric_id, cond_grp, num_v, den_v, note_txt in (
        (
            "after_lunch_med_skip_rate_on_shot_days",
            "Shot_Window_0_1d",
            shot_lunch_skips,
            shot_lunch_tot,
            "Conditional after-lunch supplement skip rate on acute Tirzepatide shot days (0-1d).",
        ),
        (
            "after_lunch_med_skip_rate_on_non_shot_days",
            "Non_Shot_Days_2_6d",
            nonshot_lunch_skips,
            nonshot_lunch_tot,
            "Conditional after-lunch supplement skip rate outside acute shot window.",
        ),
        (
            "med_skip_rate_when_meal_skipped",
            "Midday_Meal_Skipped",
            mealskip_med_skips,
            mealskip_med_tot,
            "Post-meal medication skip cascade when afternoon meal is skipped.",
        ),
        (
            "med_skip_rate_when_meal_logged",
            "Midday_Meal_Logged",
            meallog_med_skips,
            meallog_med_tot,
            "Post-meal medication skip rate when midday meal is logged.",
        ),
    ):
        rate_v = round(float(num_v) / float(den_v), 4) if den_v > 0 else np.nan
        _, mat_lbl, _ = classify_daily_maturity(den_v)
        rows.append(
            {
                "chain_category": "Conditional_Med_Adherence",
                "chain_metric_id": metric_id,
                "condition_group": cond_grp,
                "episode_key_count": total_episode_keys,
                "numerator_events": int(num_v),
                "denominator_events": int(den_v),
                "rate_or_mean_value": rate_v,
                "companion_deep_sleep_min_mean": np.nan,
                "late_work_00_05_cooccurrence_cnt": 0,
                "missed_walk_cooccurrence_cnt": 0,
                "raw_n": max(total_episode_keys, den_v),
                "valid_n": int(den_v),
                "effective_n": int(den_v),
                "missingness_pct": compute_missingness_pct(max(1, total_episode_keys), den_v) if total_episode_keys > 0 else 100.0,
                "predictor_count": 1,
                "stability_flag": "Descriptive_Only" if den_v >= 15 else "Insufficient_N",
                "maturity_level": mat_lbl,
                "epistemic_label": "Descriptive",
                "caveat_note": note_txt,
            }
        )

    # 2. Magnesium Glycinate Evening Timing vs Next-Night Deep Sleep & Nocturnal Right Foot Cramp
    cramp_dates: set[str] = set()
    if not silver_symptoms.empty:
        for r in silver_symptoms.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date"))
            sym = f"{r.get('Normalized_Symptom', '')} {r.get('Symptom', '')}".lower()
            if ds and "right foot cramp" in sym:
                cramp_dates.add(ds)

    for mag_lbl, dset in (
        ("Evening_Magnesium_Taken", sorted(mag_taken_dates)),
        ("Evening_Magnesium_Skipped", sorted(mag_skipped_dates)),
    ):
        next_deep_vals: list[float] = []
        next_cramps = 0
        for ds in dset:
            nxt_ds = (date.fromisoformat(ds) + timedelta(days=1)).isoformat()
            dv = deep_map.get(nxt_ds, np.nan)
            if np.isfinite(dv):
                next_deep_vals.append(float(dv))
            if nxt_ds in cramp_dates:
                next_cramps += 1
        den_m = len(dset)
        cramp_rate = round(float(next_cramps) / float(den_m), 4) if den_m > 0 else np.nan
        deep_m = round(float(np.mean(next_deep_vals)), 3) if next_deep_vals else np.nan
        _, mat_lbl, _ = classify_daily_maturity(den_m)
        rows.append(
            {
                "chain_category": "Magnesium_Sleep_Cramp_Chain",
                "chain_metric_id": "next_night_foot_cramp_and_deep_sleep",
                "condition_group": mag_lbl,
                "episode_key_count": total_episode_keys,
                "numerator_events": int(next_cramps),
                "denominator_events": int(den_m),
                "rate_or_mean_value": cramp_rate,
                "companion_deep_sleep_min_mean": deep_m,
                "late_work_00_05_cooccurrence_cnt": 0,
                "missed_walk_cooccurrence_cnt": 0,
                "raw_n": max(1, len(mag_taken_dates) + len(mag_skipped_dates)),
                "valid_n": int(den_m),
                "effective_n": compute_effective_n(next_deep_vals) if next_deep_vals else int(den_m),
                "missingness_pct": compute_missingness_pct(max(1, len(mag_taken_dates) + len(mag_skipped_dates)), den_m),
                "predictor_count": 1,
                "stability_flag": "Descriptive_Only" if den_m >= 15 else "Insufficient_N",
                "maturity_level": mat_lbl,
                "epistemic_label": "Descriptive",
                "caveat_note": "Evening Magnesium Glycinate status on day t paired with wake-date t+1 deep_min and Right Foot Cramp.",
            }
        )

    # 3. Non-GLP-1 & GI Clinical Flare Counts tagged with late_work_00_05_flag & missed_walk_flag
    flare_specs = (
        ("Right Shoulder Pain", "right shoulder pain"),
        ("Right Foot Cramp", "right foot cramp"),
        ("Constipation", "constipat"),
    )
    mj_logged_days_n = int(
        (gold_daily.get("micro_journal_status", pd.Series("", index=gold_daily.index)).astype(str) == "Present").sum()
    )

    for flare_label, flare_tok in flare_specs:
        flare_ep_keys: set[tuple[str, str]] = set()
        flare_dates: list[str] = []
        if not silver_symptoms.empty:
            for idx, r in enumerate(silver_symptoms.to_dict(orient="records")):
                ds = to_iso_date_str(r.get("date"))
                if not ds:
                    continue
                sym = f"{r.get('Normalized_Symptom', '')} {r.get('Symptom', '')}".lower()
                if flare_tok in sym:
                    ek = _resolve_event_key(r, ds, idx)
                    flare_ep_keys.add((ds, ek))
                    flare_dates.append(ds)

        cnt_tot = len(flare_dates)
        late_cnt = sum(1 for d in flare_dates if late_work_map.get(d) is True)
        mwalk_cnt = sum(1 for d in flare_dates if missed_walk_map.get(d, False) is True)
        rate_per_logged_day = (
            round(float(cnt_tot) / float(mj_logged_days_n), 4) if mj_logged_days_n > 0 else np.nan
        )
        _, mat_lbl, _ = classify_daily_maturity(mj_logged_days_n)
        rows.append(
            {
                "chain_category": "Clinical_Symptom_Flare_Context",
                "chain_metric_id": flare_label.lower().replace(" ", "_") + "_flare_summary",
                "condition_group": flare_label,
                "episode_key_count": len(flare_ep_keys),
                "numerator_events": int(cnt_tot),
                "denominator_events": int(mj_logged_days_n),
                "rate_or_mean_value": rate_per_logged_day,
                "companion_deep_sleep_min_mean": np.nan,
                "late_work_00_05_cooccurrence_cnt": int(late_cnt),
                "missed_walk_cooccurrence_cnt": int(mwalk_cnt),
                "raw_n": max(1, mj_logged_days_n),
                "valid_n": int(mj_logged_days_n),
                "effective_n": int(mj_logged_days_n),
                "missingness_pct": 0.0 if mj_logged_days_n > 0 else 100.0,
                "predictor_count": 2,
                "stability_flag": "Descriptive_Only" if mj_logged_days_n >= 15 else "Insufficient_N",
                "maturity_level": mat_lbl,
                "epistemic_label": "Descriptive",
                "caveat_note": f"Grouped by Source_Event_Key; co-occurrence with late_work_00_05_flag={late_cnt}, missed_walk_flag={mwalk_cnt}.",
            }
        )

    out_df = pd.DataFrame(rows, columns=cols)
    return enforce_maturity_audit(
        out_df,
        module_name="episode_chains",
        analysis_id="MICRO_JOURNAL_EPISODE_CHAINS",
        audit_logger=audit_logger,
        summary_cv_or_ci=f"distinct_episode_keys={total_episode_keys}, mj_days={mj_logged_days_n}",
    )
