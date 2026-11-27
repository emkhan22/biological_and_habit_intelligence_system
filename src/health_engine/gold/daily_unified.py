"""Gold Table 1: Gold_Daily_Unified (Master Daily Feature Store).

Rules (Prompt Section 4.1 & AGENTS.md Rules 1, 3, 4, 5):
- Continuous IST calendar spine from `2025-06-05` to max date across Silver tables.
- Work Mode (`Office_Mon_Wed`, `WFH_Tue_Thu_Fri`, `Weekend`).
- Coverage-filtered Vitals (`vitals_valid_coverage == True`).
- Primary Fitbit Staged Sleep (`Silver_Sleep_Daily`).
- Nutrition totals + `Nutrition_Log` meal-timing features (`first_meal_hr`, `last_meal_hr`,
  `eating_window_hr`, `late_kcal_after_21`, `late_fat_after_21`, `last_meal_to_bed_min`).
- Body Composition: measured columns NEVER forward-filled (`NaN` when unweighed); forward-fill
  strictly confined to explicit `latest_*` helper columns.
- Glapp mood & side-effect scores, stitched Cycle phase, and non-smoking Micro Journal counts
  (`meds_taken_cnt`, `meds_skipped_cnt`, `symptoms_cnt`, `late_work_00_05_flag`).
- Tirzepatide PK columns (`Model_Estimate`).
"""

from __future__ import annotations

from datetime import date, timedelta
import re
from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import add_display_date_col


def _num_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col].replace("", np.nan), errors="coerce")


def _parse_hhmm_to_decimal_hr(val: Any) -> float:
    if val is None or pd.isna(val):
        return np.nan
    s = str(val).strip()
    m = re.search(r"(\d{1,2}):(\d{2})", s)
    if not m:
        return np.nan
    return float(int(m.group(1))) + float(int(m.group(2))) / 60.0


def _work_mode_for_iso_date(ds: str) -> str:
    wd = date.fromisoformat(ds).weekday()  # 0=Mon .. 6=Sun
    if wd in (0, 2):
        return "Office_Mon_Wed"
    if wd in (1, 3, 4):
        return "WFH_Tue_Thu_Fri"
    return "Weekend"


def _compute_meal_timing_daily(
    nutr_log_silver: pd.DataFrame,
    sleep_daily_silver: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "date",
        "first_meal_hr",
        "last_meal_hr",
        "eating_window_hr",
        "evening_kcal_after_18",
        "evening_fat_after_18",
        "late_kcal_after_21",
        "late_fat_after_21",
        "last_meal_to_bed_min",
    ]
    if nutr_log_silver.empty:
        return pd.DataFrame(columns=cols)

    nl = nutr_log_silver.copy()
    nl["meal_hr"] = nl["time_ist"].map(_parse_hhmm_to_decimal_hr)
    nl["kcal_val"] = _num_col(nl, "calories_kcal").fillna(0.0)
    nl["fat_val"] = _num_col(nl, "fat_g").fillna(0.0)
    nl = nl.loc[nl["date"].notna() & nl["meal_hr"].notna()].copy()
    if nl.empty:
        return pd.DataFrame(columns=cols)

    # Map wake_date -> nocturnal sleep start timestamp for night (d -> d+1)
    sleep_start_by_wake_date: dict[str, pd.Timestamp] = {}
    if not sleep_daily_silver.empty:
        for r in sleep_daily_silver.to_dict(orient="records"):
            wd = str(r.get("date") or "")
            st = pd.to_datetime(r.get("start_time_ist"), errors="coerce")
            if wd and pd.notna(st):
                sleep_start_by_wake_date[wd] = st

    out_rows: list[dict[str, Any]] = []
    for ds, grp in nl.groupby("date", sort=True):
        first_hr = float(grp["meal_hr"].min())
        last_hr = float(grp["meal_hr"].max())
        win_hr = round(last_hr - first_hr, 4)

        eve_mask = grp["meal_hr"] >= 18.0
        late_mask = grp["meal_hr"] >= 21.0
        eve_kcal = float(grp.loc[eve_mask, "kcal_val"].sum())
        eve_fat = float(grp.loc[eve_mask, "fat_val"].sum())
        late_kcal = float(grp.loc[late_mask, "kcal_val"].sum())
        late_fat = float(grp.loc[late_mask, "fat_val"].sum())

        # Pair day `ds` last meal with the following night's sleep session (wake date = ds + 1 day)
        last_hh = int(last_hr)
        last_mm = int(round((last_hr - last_hh) * 60.0))
        if last_mm >= 60:
            last_hh = min(23, last_hh + 1)
            last_mm = 59
        last_meal_ts = pd.Timestamp(f"{ds} {last_hh:02d}:{last_mm:02d}:00")

        next_wake_ds = (date.fromisoformat(str(ds)) + timedelta(days=1)).isoformat()
        bed_ts = sleep_start_by_wake_date.get(next_wake_ds)
        meal_to_bed = np.nan
        if bed_ts is not None:
            diff_min = (bed_ts - last_meal_ts).total_seconds() / 60.0
            if -60.0 <= diff_min <= 1080.0:
                meal_to_bed = round(float(diff_min), 2)

        out_rows.append(
            {
                "date": str(ds),
                "first_meal_hr": round(first_hr, 4),
                "last_meal_hr": round(last_hr, 4),
                "eating_window_hr": win_hr,
                "evening_kcal_after_18": round(eve_kcal, 2),
                "evening_fat_after_18": round(eve_fat, 2),
                "late_kcal_after_21": round(late_kcal, 2),
                "late_fat_after_21": round(late_fat, 2),
                "last_meal_to_bed_min": meal_to_bed,
            }
        )

    return pd.DataFrame(out_rows, columns=cols)


def build_gold_daily_unified(
    spine_dates: list[str],
    silver_tables: dict[str, pd.DataFrame],
    daily_pk_df: pd.DataFrame,
    missingness_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble Gold_Daily_Unified across the continuous IST calendar spine."""
    spine = pd.DataFrame({"date": spine_dates})
    spine["work_mode"] = spine["date"].map(_work_mode_for_iso_date)

    # 1. Steps & Activity & Workouts
    steps_df = silver_tables.get("Silver_Steps", pd.DataFrame())
    if not steps_df.empty:
        st_sub = pd.DataFrame(
            {
                "date": steps_df["date"],
                "steps": _num_col(steps_df, "steps"),
                "fitbit_steps": _num_col(steps_df, "fitbit_steps"),
                "phone_fallback_steps": _num_col(steps_df, "phone_fallback_steps"),
            }
        )
        spine = spine.merge(st_sub, on="date", how="left")
    else:
        spine["steps"] = np.nan
        spine["fitbit_steps"] = np.nan
        spine["phone_fallback_steps"] = np.nan

    act_df = silver_tables.get("Silver_Activity_Daily", pd.DataFrame())
    if not act_df.empty:
        act_sub = pd.DataFrame(
            {
                "date": act_df["date"],
                "activity_workout_mins": _num_col(act_df, "workout_mins"),
                "moderate_mins": _num_col(act_df, "moderate_mins"),
                "vigorous_mins": _num_col(act_df, "vigorous_mins"),
                "peak_mins": _num_col(act_df, "peak_mins"),
                "zone_mins": _num_col(act_df, "zone_mins"),
                "activity_total_kcal": _num_col(act_df, "total_kcal"),
                "active_hours": _num_col(act_df, "active_hours"),
            }
        )
        spine = spine.merge(act_sub, on="date", how="left")
    else:
        for c in (
            "activity_workout_mins",
            "moderate_mins",
            "vigorous_mins",
            "peak_mins",
            "zone_mins",
            "activity_total_kcal",
            "active_hours",
        ):
            spine[c] = np.nan

    wk_df = silver_tables.get("Silver_Workouts", pd.DataFrame())
    if not wk_df.empty:
        wk_tmp = wk_df.copy()
        wk_tmp["_dur"] = _num_col(wk_tmp, "duration_mins").fillna(0.0)
        ex_low = wk_tmp["exercise_type"].astype(str).str.lower() if "exercise_type" in wk_tmp.columns else pd.Series("", index=wk_tmp.index)
        wk_tmp["_is_str"] = ex_low.str.contains("strength|weight|resistance", regex=True).astype(int)
        wk_grp = (
            wk_tmp.groupby("date", as_index=False)
            .agg(
                workout_sessions_cnt=("date", "count"),
                workout_mins=("_dur", "sum"),
                strength_sessions_cnt=("_is_str", "sum"),
            )
        )
        spine = spine.merge(wk_grp, on="date", how="left")
    else:
        spine["workout_sessions_cnt"] = np.nan
        spine["workout_mins"] = np.nan
        spine["strength_sessions_cnt"] = np.nan

    # True_Zero workouts when wearable steps/activity were recorded that day, NaN when Not_Worn
    wearable_active = spine["steps"].notna() | spine["active_hours"].notna()
    for wc in ("workout_sessions_cnt", "workout_mins", "strength_sessions_cnt"):
        spine[wc] = np.where(spine[wc].notna(), spine[wc], np.where(wearable_active, 0.0, np.nan))

    # 2. Coverage-Filtered Vitals
    vit_df = silver_tables.get("Silver_Vitals", pd.DataFrame())
    if not vit_df.empty:
        v_valid = vit_df["vitals_valid_coverage"].fillna(False)
        vit_sub = pd.DataFrame(
            {
                "date": vit_df["date"],
                "vitals_valid_coverage": v_valid,
                "hr_hours_covered": vit_df["hr_hours_covered"],
                "hr_sample_count": vit_df["hr_sample_count_val"],
                "sampling_era": vit_df["sampling_era"],
                "avg_hr": np.where(v_valid, vit_df["avg_hr_val"], np.nan),
                "daytime_avg_hr": np.where(v_valid, vit_df["daytime_avg_hr"], np.nan),
                "resting_hr": np.where(v_valid, vit_df["resting_hr_val"], np.nan),
                "p95_hr": np.where(v_valid, vit_df["p95_hr_val"], np.nan),
                "hrv_rmssd_ms": np.where(v_valid, vit_df["hrv_rmssd_ms_val"], np.nan),
                "breathing_rate_rpm": np.where(v_valid, vit_df["breathing_rate_rpm_val"], np.nan),
            }
        )
        spine = spine.merge(vit_sub, on="date", how="left")
    else:
        for c in (
            "vitals_valid_coverage",
            "hr_hours_covered",
            "hr_sample_count",
            "sampling_era",
            "avg_hr",
            "daytime_avg_hr",
            "resting_hr",
            "p95_hr",
            "hrv_rmssd_ms",
            "breathing_rate_rpm",
        ):
            spine[c] = np.nan

    # 3. Fitbit Staged Sleep (Primary Nocturnal Session)
    sl_df = silver_tables.get("Silver_Sleep_Daily", pd.DataFrame())
    sleep_keep = [
        "date",
        "session_id",
        "asleep_min",
        "light_min",
        "deep_min",
        "rem_min",
        "awake_min",
        "bedtime_hr_rel_mid",
        "wake_hr",
        "sleep_eff_pct",
        "sleep_fragmentation_per_hr",
        "sleep_hr_avg",
        "sleep_hr_p95",
    ]
    if not sl_df.empty:
        sl_sub = sl_df[[c for c in sleep_keep if c in sl_df.columns]].rename(
            columns={"session_id": "sleep_session_id"}
        )
        spine = spine.merge(sl_sub, on="date", how="left")
    else:
        spine["sleep_session_id"] = np.nan
        for c in sleep_keep[2:]:
            spine[c] = np.nan

    # 4. Nutrition Totals + Meal Timing
    nu_df = silver_tables.get("Silver_Nutrition", pd.DataFrame())
    if not nu_df.empty:
        nu_sub = pd.DataFrame(
            {
                "date": nu_df["date"],
                "calories_kcal": _num_col(nu_df, "calories_kcal"),
                "protein_g": _num_col(nu_df, "protein_g"),
                "carbs_g": _num_col(nu_df, "carbs_g"),
                "fat_g": _num_col(nu_df, "fat_g"),
                "fiber_g": _num_col(nu_df, "fiber_g"),
                "sugar_g": _num_col(nu_df, "sugar_g"),
                "hydration_liters": _num_col(nu_df, "hydration_liters"),
                "nutrition_entries_logged": _num_col(nu_df, "entries_logged"),
            }
        )
        spine = spine.merge(nu_sub, on="date", how="left")
    else:
        for c in (
            "calories_kcal",
            "protein_g",
            "carbs_g",
            "fat_g",
            "fiber_g",
            "sugar_g",
            "hydration_liters",
            "nutrition_entries_logged",
        ):
            spine[c] = np.nan

    meal_timing_df = _compute_meal_timing_daily(
        silver_tables.get("Silver_Nutrition_Log", pd.DataFrame()),
        sl_df,
    )
    spine = spine.merge(meal_timing_df, on="date", how="left")

    # 5. Body Composition & Canonical Weight (Measured = NaN when unweighed; helper latest_* ffilled)
    bc_df = silver_tables.get("Silver_Body_Composition", pd.DataFrame())
    hc_w_df = silver_tables.get("Silver_HC_Weight", pd.DataFrame())
    gl_w_df = silver_tables.get("Silver_Glapp_Weight", pd.DataFrame())

    weight_by_date: dict[str, tuple[float, str]] = {}
    if not gl_w_df.empty:
        for ds, w in zip(gl_w_df["date"], _num_col(gl_w_df, "weight_kg")):
            if pd.notna(w):
                weight_by_date[str(ds)] = (float(w), "Glapp")
    if not hc_w_df.empty:
        for ds, w in zip(hc_w_df["date"], _num_col(hc_w_df, "weight_kg")):
            if pd.notna(w):
                weight_by_date[str(ds)] = (float(w), "HealthConnect")
    if not bc_df.empty:
        for ds, w in zip(bc_df["date"], _num_col(bc_df, "weight_kg")):
            if pd.notna(w):
                weight_by_date[str(ds)] = (float(w), "Fitdays")

    spine["weight_kg"] = [
        weight_by_date[d][0] if d in weight_by_date else np.nan for d in spine["date"]
    ]
    spine["weight_source"] = [
        weight_by_date[d][1] if d in weight_by_date else "" for d in spine["date"]
    ]

    bc_cols = [
        "bmi",
        "body_fat_pct",
        "fat_mass_kg",
        "muscle_mass_kg",
        "skeletal_muscle_pct",
        "visceral_fat",
        "bone_mass_kg",
        "protein_mass_kg",
    ]
    if not bc_df.empty:
        bc_sub = pd.DataFrame({"date": bc_df["date"]})
        for c in bc_cols:
            bc_sub[c] = _num_col(bc_df, c)
        spine = spine.merge(bc_sub, on="date", how="left")
    else:
        for c in bc_cols:
            spine[c] = np.nan

    # Explicit forward-filled helper columns ONLY (measured columns above stay strictly NaN!)
    spine["latest_weight_kg"] = spine["weight_kg"].ffill()
    spine["latest_weight_date"] = pd.Series(
        np.where(spine["weight_kg"].notna(), spine["date"], np.nan),
        index=spine.index,
        dtype="object",
    ).ffill()
    for c in ("body_fat_pct", "fat_mass_kg", "muscle_mass_kg", "skeletal_muscle_pct", "visceral_fat"):
        spine[f"latest_{c}"] = spine[c].ffill()

    # 6. Glapp Mood & Side Effects
    gl_mood = silver_tables.get("Silver_Glapp_Mood", pd.DataFrame())
    if not gl_mood.empty:
        gm_sub = pd.DataFrame(
            {
                "date": gl_mood["date"],
                "glapp_mood": _num_col(gl_mood, "mood"),
                "hunger": _num_col(gl_mood, "hunger"),
                "food_noise": _num_col(gl_mood, "food_noise"),
                "feeling": gl_mood["feeling"].astype(str) if "feeling" in gl_mood.columns else "",
            }
        )
        spine = spine.merge(gm_sub, on="date", how="left")
    else:
        spine["glapp_mood"] = np.nan
        spine["hunger"] = np.nan
        spine["food_noise"] = np.nan
        spine["feeling"] = ""

    gl_se = silver_tables.get("Silver_Glapp_Side_Effects", pd.DataFrame())
    gi_regex = r"nausea|belch|burp|bloat|reflux|acid|indigest|constipat|diarrh|vomit|stomach|gas|gi"
    if not gl_se.empty:
        se_tmp = gl_se.copy()
        se_tmp["_lvl"] = _num_col(se_tmp, "level").fillna(1.0)
        se_low = se_tmp["side_effect"].astype(str).str.lower()
        se_tmp["_gi_lvl"] = np.where(se_low.str.contains(gi_regex, regex=True), se_tmp["_lvl"], 0.0)
        se_grp = (
            se_tmp.groupby("date", as_index=False)
            .agg(
                glapp_side_effects_cnt=("side_effect", "count"),
                glapp_side_effect_score=("_lvl", "sum"),
                glapp_gi_score=("_gi_lvl", "sum"),
            )
        )
        spine = spine.merge(se_grp, on="date", how="left")
    else:
        spine["glapp_side_effects_cnt"] = np.nan
        spine["glapp_side_effect_score"] = np.nan
        spine["glapp_gi_score"] = np.nan

    # 7. Stitched Menstrual Cycle Dimension
    cyc_df = silver_tables.get("Silver_Cycle_Dim", pd.DataFrame())
    if not cyc_df.empty:
        cyc_sub = cyc_df[
            [
                "date",
                "cycle_day",
                "is_period_day",
                "is_spotting_day",
                "flow_level",
                "cycle_phase",
                "source_owner",
            ]
        ].rename(columns={"source_owner": "cycle_source_owner"})
        spine = spine.merge(cyc_sub, on="date", how="left")
    else:
        for c in ("cycle_day", "is_period_day", "is_spotting_day", "flow_level", "cycle_phase", "cycle_source_owner"):
            spine[c] = np.nan

    # 8. Non-Smoking Micro Journal Daily Aggregates
    meds_df = silver_tables.get("Silver_Meds_Log", pd.DataFrame())
    sym_df = silver_tables.get("Silver_Symptoms_Log", pd.DataFrame())
    jm_df = silver_tables.get("Silver_Journal_Mood", pd.DataFrame())

    mj_logged_dates: set[str] = set()
    for dframe in (meds_df, sym_df, jm_df):
        if not dframe.empty and "date" in dframe.columns:
            mj_logged_dates |= set(dframe["date"].dropna().astype(str))

    if not meds_df.empty:
        m_tmp = meds_df.copy()
        st_low = m_tmp["Status"].astype(str).str.lower()
        m_tmp["_taken"] = st_low.str.contains("taken").astype(int)
        m_tmp["_skipped"] = st_low.str.contains("skip|miss", regex=True).astype(int)
        m_grp = (
            m_tmp.groupby("date", as_index=False)
            .agg(meds_taken_cnt=("_taken", "sum"), meds_skipped_cnt=("_skipped", "sum"))
        )
        spine = spine.merge(m_grp, on="date", how="left")
    else:
        spine["meds_taken_cnt"] = np.nan
        spine["meds_skipped_cnt"] = np.nan

    if not sym_df.empty:
        s_grp = sym_df.groupby("date", as_index=False).agg(symptoms_cnt=("Normalized_Symptom", "count"))
        spine = spine.merge(s_grp, on="date", how="left")
    else:
        spine["symptoms_cnt"] = np.nan

    # Late work 00:00 - 05:00 IST flag from non-smoking Micro Journal entries
    late_work_dates: set[str] = set()
    for dframe in (sym_df, jm_df, meds_df):
        if dframe.empty or "time_ist" not in dframe.columns:
            continue
        hrs = dframe["time_ist"].map(_parse_hhmm_to_decimal_hr)
        ctx = (
            dframe["Context_Tag"].astype(str).str.lower()
            if "Context_Tag" in dframe.columns
            else pd.Series("", index=dframe.index)
        )
        notes_col = "Notes" if "Notes" in dframe.columns else ("Summary" if "Summary" in dframe.columns else None)
        notes_txt = dframe[notes_col].astype(str).str.lower() if notes_col else pd.Series("", index=dframe.index)
        is_late_work = hrs.between(0.0, 4.9999) & (
            ctx.str.contains("work", na=False) | notes_txt.str.contains("work", na=False)
        )
        late_work_dates |= set(dframe.loc[is_late_work, "date"].dropna().astype(str))

    # Keep unlogged Micro Journal days as NaN (never silent zero-filling); True_Zero when journal active
    mj_present_mask = spine["date"].isin(mj_logged_dates)
    spine["symptoms_cnt"] = np.where(
        spine["symptoms_cnt"].notna(),
        spine["symptoms_cnt"],
        np.where(mj_present_mask, 0.0, np.nan),
    )
    spine["late_work_00_05_flag"] = [
        True if d in late_work_dates else (False if d in mj_logged_dates else np.nan)
        for d in spine["date"]
    ]

    # 9. Tirzepatide PK Columns
    if not daily_pk_df.empty:
        spine = spine.merge(daily_pk_df, on="date", how="left")

    # 10. Missingness Provenance Tags
    if not missingness_df.empty:
        miss_cols = [
            c
            for c in missingness_df.columns
            if c == "date" or c.endswith("_status") or c.endswith("_missing_reason")
        ]
        spine = spine.merge(missingness_df[miss_cols], on="date", how="left")

    spine["epistemic_class"] = "Descriptive"
    return add_display_date_col(spine.sort_values("date").reset_index(drop=True), "date")
