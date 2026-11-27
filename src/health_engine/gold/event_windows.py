"""Gold Table 4: Gold_Event_Windows & Gold_Med_Bioavailability (Anti-Reverse-Causality).

Rules (Prompt Section 4.4 & AGENTS.md Rules 3, 4):
Build strictly time-ordered exposure -> outcome pairs (NOT blanket `t-1` lags):
1. Day `t` meal timing (`last_meal_to_bed_min`, `late_kcal_after_21`, `late_fat_after_21`)
   -> following night (`wake_date = t+1`) `sleep_hr_avg` and `spikes_per_asleep_hr`.
2. Evening fat/kcal (`>= 18:00` and `>= 21:00` on day `t-1`)
   -> overnight + next-morning GI symptoms on day `t`.
3. Prior-day `hydration_liters` (day `t-1`, strictly where logged)
   -> next-day constipation/headache on day `t`.
4. Rolling 24h (`t-1`) & 48h (`t-2..t-1`) `fiber_g` -> subsequent GI/constipation score on day `t`.
5. `Micro_Journal!Meds_Log` dose timestamps vs `Nutrition_Log` meal timestamps:
   - Thyronorm fasting gap (`min` to first subsequent meal)
   - Fat co-factor (`>= 10g` meal fat within +/-90 min of Vit D / Fish Oil)
   - Conditional medication skip cascade on shot days (`days_since_shot in {0, 1}`).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import add_display_date_col


GI_SYMPTOM_REGEX = re.compile(
    r"nausea|belch|burp|bloat|reflux|acid|indigest|constipat|diarrh|vomit|stomach|gas|gi",
    re.IGNORECASE,
)


def _time_to_min(val: Any) -> float:
    if val is None or pd.isna(val):
        return np.nan
    m = re.search(r"(\d{1,2}):(\d{2})", str(val).strip())
    if not m:
        return np.nan
    return float(int(m.group(1)) * 60 + int(m.group(2)))


def _num_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col].replace("", np.nan), errors="coerce")


def build_med_bioavailability_table(
    silver_meds: pd.DataFrame,
    silver_nutr_log: pd.DataFrame,
    daily_pk_df: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Audit each Meds_Log event against Nutrition_Log meal timestamps and shot-day context."""
    cols = [
        "date",
        "display_date",
        "time_ist",
        "med_or_supplement",
        "dose",
        "status",
        "days_since_shot",
        "is_shot_window_0_1d",
        "thyronorm_fasting_gap_min",
        "thyronorm_fasting_status",
        "fat_cofactor_meal_fat_g",
        "fat_cofactor_valid_10g",
        "bioavailability_note",
        "epistemic_class",
    ]
    if silver_meds.empty:
        return pd.DataFrame(columns=cols)

    bio_cfg = thresholds_cfg["bioavailability_gates"]
    opt_fast = float(bio_cfg["thyronorm_optimal_fasting_min"])
    acc_fast = float(bio_cfg["thyronorm_acceptable_fasting_min"])
    fat_win = float(bio_cfg["fat_soluble_window_min"])
    min_fat = float(bio_cfg["fat_soluble_min_fat_g"])

    meals_by_date: dict[str, list[dict[str, float]]] = {}
    if not silver_nutr_log.empty:
        nl = silver_nutr_log.copy()
        nl["_tmin"] = nl["time_ist"].map(_time_to_min)
        nl["_fat"] = _num_col(nl, "fat_g").fillna(0.0)
        nl["_kcal"] = _num_col(nl, "calories_kcal").fillna(0.0)
        for ds, grp in nl.loc[nl["date"].notna() & nl["_tmin"].notna()].groupby("date"):
            meals_by_date[str(ds)] = (
                grp.sort_values("_tmin")[["_tmin", "_fat", "_kcal"]].to_dict(orient="records")
            )

    shot_days_map = (
        daily_pk_df.set_index("date")["days_since_shot"].to_dict()
        if not daily_pk_df.empty
        else {}
    )

    out_rows: list[dict[str, Any]] = []
    for r in silver_meds.to_dict(orient="records"):
        ds = str(r.get("date") or "")
        t_str = str(r.get("time_ist") or "")
        t_min = _time_to_min(t_str)
        med = str(r.get("Med_or_Supplement") or "").strip()
        med_low = med.lower()
        dose = str(r.get("Dose") or "").strip()
        status = str(r.get("Status") or "").strip()
        st_low = status.lower()

        dss = shot_days_map.get(ds, np.nan)
        is_shot_win = bool(pd.notna(dss) and float(dss) in (0.0, 1.0))

        thy_gap = np.nan
        thy_stat = ""
        fat_g_near = np.nan
        fat_valid: Any = np.nan
        note = ""

        if "skip" in st_low or "miss" in st_low:
            note = (
                "Skipped within 0-1d post-shot window"
                if is_shot_win
                else "Skipped outside acute shot window"
            )
        elif "thyronorm" in med_low or "levothyroxine" in med_low:
            day_meals = meals_by_date.get(ds, [])
            subsequent = [
                m for m in day_meals if not np.isnan(t_min) and m["_tmin"] >= t_min and m["_kcal"] > 0
            ]
            if subsequent:
                thy_gap = round(float(subsequent[0]["_tmin"] - t_min), 1)
                if thy_gap >= opt_fast:
                    thy_stat = "Optimal_Fasting_Gap"
                elif thy_gap >= acc_fast:
                    thy_stat = "Acceptable_Fasting_Gap"
                else:
                    thy_stat = "Short_Fasting_Gap"
                note = f"First caloric meal +{thy_gap} min after Thyronorm"
            else:
                thy_stat = "Unlogged_Subsequent_Meal"
                note = "No subsequent meal timestamp logged on date"
        elif any(k in med_low for k in ("vitamin d", "d-rise", "cholecalciferol", "fish oil", "omega")):
            day_meals = meals_by_date.get(ds, [])
            if day_meals and not np.isnan(t_min):
                near = [m["_fat"] for m in day_meals if abs(m["_tmin"] - t_min) <= fat_win]
                fat_g_near = round(float(max(near)), 2) if near else 0.0
                fat_valid = bool(fat_g_near >= min_fat)
                note = (
                    f"Max meal fat within +/-{int(fat_win)}m: {fat_g_near}g"
                    if near
                    else f"No meal logged within +/-{int(fat_win)}m"
                )
            else:
                note = "Meal timestamps unlogged for fat co-factor check"
        else:
            note = "Recorded non-fasting/non-lipid-soluble dose"

        out_rows.append(
            {
                "date": ds,
                "time_ist": t_str,
                "med_or_supplement": med,
                "dose": dose,
                "status": status,
                "days_since_shot": dss,
                "is_shot_window_0_1d": is_shot_win,
                "thyronorm_fasting_gap_min": thy_gap,
                "thyronorm_fasting_status": thy_stat,
                "fat_cofactor_meal_fat_g": fat_g_near,
                "fat_cofactor_valid_10g": fat_valid,
                "bioavailability_note": note,
                "epistemic_class": "Causal_Hypothesis",
            }
        )

    out_df = add_display_date_col(pd.DataFrame(out_rows), "date")
    return out_df[cols]


def build_gold_event_windows(
    gold_daily: pd.DataFrame,
    gold_sleep_spikes: pd.DataFrame,
    silver_glapp_se: pd.DataFrame,
    silver_symptoms: pd.DataFrame,
    med_bio_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build daily anti-reverse-causality exposure -> outcome Gold_Event_Windows table."""
    if gold_daily.empty:
        return pd.DataFrame()

    df = gold_daily.sort_values("date").reset_index(drop=True).copy()

    # Map wake_date -> sleep HR & spike rate
    spikes_map: dict[str, dict[str, Any]] = {}
    if not gold_sleep_spikes.empty:
        for r in gold_sleep_spikes.to_dict(orient="records"):
            spikes_map[str(r["date"])] = r

    # Extract overnight + morning (00:00..12:00) and daily GI / constipation / headache outcomes on day t
    overnight_morning_gi_by_date: dict[str, int] = {}
    constip_dates: set[str] = set()
    headache_dates: set[str] = set()
    symptom_active_dates: set[str] = set()

    if not silver_symptoms.empty:
        for r in silver_symptoms.to_dict(orient="records"):
            ds = str(r.get("date") or "")
            if not ds:
                continue
            symptom_active_dates.add(ds)
            sym_txt = f"{r.get('Normalized_Symptom', '')} {r.get('Symptom', '')}".lower()
            tmin = _time_to_min(r.get("time_ist"))
            if GI_SYMPTOM_REGEX.search(sym_txt):
                if np.isnan(tmin) or tmin <= 720.0:  # 00:00 - 12:00 IST overnight/morning window
                    overnight_morning_gi_by_date[ds] = overnight_morning_gi_by_date.get(ds, 0) + 1
            if "constipat" in sym_txt:
                constip_dates.add(ds)
            if "headache" in sym_txt or "migraine" in sym_txt:
                headache_dates.add(ds)

    glapp_constip_score: dict[str, float] = {}
    if not silver_glapp_se.empty:
        for r in silver_glapp_se.to_dict(orient="records"):
            ds = str(r.get("date") or "")
            if not ds:
                continue
            symptom_active_dates.add(ds)
            se_txt = str(r.get("side_effect") or "").lower()
            lvl_raw = r.get("level")
            lvl = float(lvl_raw) if lvl_raw not in (None, "") and pd.notna(lvl_raw) else 1.0
            if "constipat" in se_txt:
                constip_dates.add(ds)
                glapp_constip_score[ds] = glapp_constip_score.get(ds, 0.0) + lvl
            if "headache" in se_txt or "migraine" in se_txt:
                headache_dates.add(ds)

    # Meds daily bioavailability summaries
    thy_gap_by_date: dict[str, float] = {}
    fat_cof_by_date: dict[str, bool] = {}
    if not med_bio_df.empty:
        thy_sub = med_bio_df.loc[med_bio_df["thyronorm_fasting_gap_min"].notna()]
        for ds, grp in thy_sub.groupby("date"):
            thy_gap_by_date[str(ds)] = float(grp["thyronorm_fasting_gap_min"].min())
        fat_sub = med_bio_df.loc[med_bio_df["fat_cofactor_valid_10g"].notna()]
        for ds, grp in fat_sub.groupby("date"):
            fat_cof_by_date[str(ds)] = bool(grp["fat_cofactor_valid_10g"].all())

    # Pre-extract vectors for clean causal window indexing
    dates = df["date"].astype(str).tolist()
    n = len(dates)
    eve_kcal_18 = pd.to_numeric(df["evening_kcal_after_18"], errors="coerce").to_numpy(dtype=float)
    eve_fat_18 = pd.to_numeric(df["evening_fat_after_18"], errors="coerce").to_numpy(dtype=float)
    late_kcal_21 = pd.to_numeric(df["late_kcal_after_21"], errors="coerce").to_numpy(dtype=float)
    late_fat_21 = pd.to_numeric(df["late_fat_after_21"], errors="coerce").to_numpy(dtype=float)
    meal_to_bed = pd.to_numeric(df["last_meal_to_bed_min"], errors="coerce").to_numpy(dtype=float)
    hydr = pd.to_numeric(df["hydration_liters"], errors="coerce").to_numpy(dtype=float)
    fiber = pd.to_numeric(df["fiber_g"], errors="coerce").to_numpy(dtype=float)
    sleep_hr = pd.to_numeric(df["sleep_hr_avg"], errors="coerce").to_numpy(dtype=float)
    glapp_gi = pd.to_numeric(df["glapp_gi_score"], errors="coerce").to_numpy(dtype=float)
    glapp_mood_present = df["glapp_mood_status"].astype(str).eq("Present").to_numpy()
    meds_skip = pd.to_numeric(df["meds_skipped_cnt"], errors="coerce").to_numpy(dtype=float)
    dss_arr = pd.to_numeric(df["days_since_shot"], errors="coerce").to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    for i, ds in enumerate(dates):
        # 1. Exposure on day `t` -> Following night (`wake_date = t+1`) autonomic sleep outcomes
        next_sleep_hr = sleep_hr[i + 1] if i + 1 < n else np.nan
        next_wake_ds = dates[i + 1] if i + 1 < n else ""
        next_sp_row = spikes_map.get(next_wake_ds, {})
        next_spikes_rate = next_sp_row.get("spikes_per_asleep_hr", np.nan)

        # 2. Exposure on day `t-1` -> Overnight + morning/day `t` GI & symptom outcomes
        t_minus_1_eve_kcal_18 = eve_kcal_18[i - 1] if i >= 1 else np.nan
        t_minus_1_eve_fat_18 = eve_fat_18[i - 1] if i >= 1 else np.nan
        t_minus_1_late_kcal_21 = late_kcal_21[i - 1] if i >= 1 else np.nan
        t_minus_1_late_fat_21 = late_fat_21[i - 1] if i >= 1 else np.nan

        # 3. Prior-day hydration (t-1, strictly NaN when unlogged) -> day `t` constipation/headache
        prior_day_hydration = hydr[i - 1] if i >= 1 else np.nan

        # 4. Preceding 24h (t-1) and 48h (mean of t-2, t-1 when both logged) fiber_g -> day `t` GI/constipation
        fiber_24h = fiber[i - 1] if i >= 1 else np.nan
        if i >= 2 and not np.isnan(fiber[i - 1]) and not np.isnan(fiber[i - 2]):
            fiber_48h = round(float(fiber[i - 1] + fiber[i - 2]) / 2.0, 3)
        else:
            fiber_48h = np.nan

        # Day `t` observed symptom outcomes (NaN when neither Glapp nor Micro Journal active on day t)
        obs_active = bool(glapp_mood_present[i] or (ds in symptom_active_dates))
        gi_overnight_morn = overnight_morning_gi_by_date.get(ds, 0)
        gi_day_t_score = (
            (0.0 if np.isnan(glapp_gi[i]) else float(glapp_gi[i])) + float(gi_overnight_morn)
            if obs_active
            else np.nan
        )
        constip_flag: Any = (ds in constip_dates) if obs_active else np.nan
        constip_score = (
            glapp_constip_score.get(ds, 1.0 if ds in constip_dates else 0.0)
            if obs_active
            else np.nan
        )
        headache_flag: Any = (ds in headache_dates) if obs_active else np.nan

        # 5. Medication bioavailability & shot-day skip cascade on day `t`
        dss_val = dss_arr[i]
        in_shot_win = bool(not np.isnan(dss_val) and dss_val in (0.0, 1.0))
        skip_cascade_flag: Any = (
            bool(in_shot_win and meds_skip[i] > 0)
            if not np.isnan(meds_skip[i])
            else np.nan
        )

        rows.append(
            {
                "date": ds,
                # Pair 1: Day t meal timing -> Night (t -> t+1) sleep HR & spikes
                "exp_last_meal_to_bed_min": meal_to_bed[i],
                "exp_late_kcal_after_21_t": late_kcal_21[i],
                "exp_late_fat_after_21_t": late_fat_21[i],
                "out_following_night_sleep_hr_avg": next_sleep_hr,
                "out_following_night_spikes_per_asleep_hr": next_spikes_rate,
                # Pair 2: Day t-1 evening fat/kcal -> Day t overnight/morning GI
                "exp_t_minus_1_evening_kcal_after_18": t_minus_1_eve_kcal_18,
                "exp_t_minus_1_evening_fat_after_18": t_minus_1_eve_fat_18,
                "exp_t_minus_1_late_kcal_after_21": t_minus_1_late_kcal_21,
                "exp_t_minus_1_late_fat_after_21": t_minus_1_late_fat_21,
                "out_day_t_gi_symptom_score": gi_day_t_score,
                # Pair 3: Day t-1 hydration -> Day t constipation / headache
                "exp_t_minus_1_hydration_liters": prior_day_hydration,
                "out_day_t_constipation_flag": constip_flag,
                "out_day_t_headache_flag": headache_flag,
                # Pair 4: Trailing 24h & 48h fiber -> Day t GI / constipation score
                "exp_rolling_24h_fiber_g": fiber_24h,
                "exp_rolling_48h_fiber_avg_g": fiber_48h,
                "out_day_t_constipation_score": constip_score,
                # Pair 5: Meds x Meal bioavailability & shot-day skip cascade
                "thyronorm_fasting_gap_min": thy_gap_by_date.get(ds, np.nan),
                "fat_soluble_cofactor_valid_10g": fat_cof_by_date.get(ds, np.nan),
                "days_since_shot": dss_val,
                "meds_skipped_cnt": meds_skip[i],
                "shot_day_skip_cascade_flag": skip_cascade_flag,
                "epistemic_class": "Causal_Hypothesis",
            }
        )

    return add_display_date_col(pd.DataFrame(rows), "date")
