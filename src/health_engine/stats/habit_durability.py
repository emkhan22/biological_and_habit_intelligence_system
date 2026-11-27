"""Taper / Off-GLP-1 Habit & Eating Durability Engine (`habit_durability.py` -> `Gold_Habit_Durability`).

Automatically tags dates and Thu-Wed weeks into:
- `Pre_GLP1` (`date < 2026-10-21`)
- `On_GLP1` (tagged by active dose, e.g. `Tirz_2.5mg`, `Tirz_5.0mg`)
- `Off_GLP1_Transition`, triggered automatically when ANY of the following occurs after `2026-10-21`:
  (a) `days_since_shot > 10` (missed/delayed past expected 7-day cadence + 3-day grace window),
  (b) active Tirzepatide dose steps down below the established steady-state dose AND is sustained
      across the next scheduled cycle (filtering out one-off adjustments), OR
  (c) an explicit stop/taper flag is logged (`taper_or_stop_flag == True` or `status == 'Stopped'`).

Emits three row types in `Gold_Habit_Durability`:
1. `Weekly_Durability`: Weekly habit-building & maintenance scorecard (protein consistency,
   clean-eating score, eating window & late eating, steps/activity/strength consistency, sleep
   duration & bedtime regularity, hunger/food noise, weight drift).
2. `On_GLP1_Peak_vs_Trough`: Comparison of Peak Suppression (Days 0-2) vs Late-Cycle Trough
   (Days 4-6) while on GLP-1 to quantify habit carryover prior to tapering.
3. `Post_GLP1_Week`: `Post_GLP1_Week_1, Week_2, ...` comparison against `On_GLP1` and `Pre_GLP1`
   baselines whenever `Off_GLP1_Transition` is active.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import iso_to_display_date
from health_engine.stats.maturity import (
    StatsAuditLogger,
    classify_daily_maturity,
    compute_effective_n,
    compute_missingness_pct,
    enforce_maturity_audit,
)

HABIT_DURABILITY_COLUMNS: list[str] = [
    "row_type",
    "period_id",
    "start_date",
    "display_start_date",
    "end_date",
    "display_end_date",
    "glp1_regime",
    "active_dose_label",
    "transition_trigger_reason",
    "post_glp1_week_num",
    "days_in_window",
    "nutrition_logged_days",
    "nutrition_coverage_pct",
    "protein_g_mean",
    "protein_target_consistency_pct",
    "clean_eating_score",
    "eating_window_hr_mean",
    "late_eating_after_21_pct",
    "steps_mean",
    "steps_target_consistency_pct",
    "active_hours_mean",
    "strength_sessions_cnt",
    "asleep_hr_mean",
    "bedtime_sd_hr",
    "earlier_bedtime_recovery_pct",
    "hunger_mean",
    "food_noise_mean",
    "weight_drift_kg_wk",
    "delta_protein_vs_on_glp1",
    "delta_clean_eating_vs_on_glp1",
    "delta_steps_vs_on_glp1",
    "delta_asleep_hr_vs_on_glp1",
    "delta_hunger_vs_on_glp1",
    "delta_protein_vs_pre_glp1",
    "delta_steps_vs_pre_glp1",
    "delta_asleep_hr_vs_pre_glp1",
    "raw_n",
    "valid_n",
    "effective_n",
    "missingness_pct",
    "predictor_count",
    "stability_flag",
    "maturity_level",
    "epistemic_label",
    "caveat_note",
    "epistemic_class",
]


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _compute_daily_clean_eating_score(df: pd.DataFrame) -> pd.Series:
    """0-100 composite clean-eating score strictly on logged nutrition days (NaN when Not_Logged)."""
    prot = _num_series(df, "protein_g")
    fib = _num_series(df, "fiber_g")
    sug = _num_series(df, "sugar_g")
    late_kcal = _num_series(df, "late_kcal_after_21").fillna(0.0)
    nutr_present = (
        df["nutrition_status"].astype(str) == "Present"
        if "nutrition_status" in df.columns
        else _num_series(df, "calories_kcal").notna()
    )

    p_score = (prot / 95.0).clip(0.0, 1.0) * 40.0
    f_score = (fib / 25.0).clip(0.0, 1.0) * 25.0
    # Lower sugar scores higher: <=25g gets full 20 pts, >=75g gets 0 pts
    s_score = (1.0 - ((sug.fillna(25.0) - 25.0) / 50.0).clip(0.0, 1.0)) * 20.0
    # No heavy post-21:00 eating (>=300 kcal) gets 15 pts
    l_score = np.where(late_kcal < 300.0, 15.0, np.where(late_kcal < 500.0, 7.5, 0.0))

    raw_score = p_score.fillna(0.0) + f_score.fillna(0.0) + s_score.fillna(0.0) + l_score
    return pd.Series(np.where(nutr_present, np.round(raw_score, 2), np.nan), index=df.index, dtype=float)


def _detect_sustained_taper_dates(silver_shots: pd.DataFrame | None) -> set[str]:
    """Identify dates where Tirzepatide dose stepped down below steady-state across >=2 consecutive cycles."""
    if silver_shots is None or silver_shots.empty or "date" not in silver_shots.columns:
        return set()
    sh = silver_shots.copy()
    sh["_dose"] = pd.to_numeric(sh.get("dose_mg", np.nan), errors="coerce")
    sh["_dt"] = sh["date"].astype(str).str.slice(0, 10)
    sh = sh.loc[sh["_dose"].notna()].sort_values("_dt").reset_index(drop=True)
    if len(sh) < 3:
        return set()

    # Established steady-state peak requires at least 2 shots at or above that dose
    taper_start_dates: set[str] = set()
    dose_history: list[float] = []
    steady_peak = 0.0
    for idx, row in sh.iterrows():
        d_val = float(row["_dose"])
        # Check if prior dose_history had >=2 shots at >= current steady_peak
        if len(dose_history) >= 2:
            # Highest dose achieved at least twice (or established for >=2 shots)
            counts: dict[float, int] = {}
            for dh in dose_history:
                counts[dh] = counts.get(dh, 0) + 1
            established = [k for k, v in counts.items() if v >= 2]
            if established:
                steady_peak = max(steady_peak, max(established))

        if steady_peak > 0 and d_val < steady_peak:
            # Sustained check: next scheduled shot also remains < steady_peak
            if idx + 1 < len(sh) and float(sh.loc[idx + 1, "_dose"]) < steady_peak:
                taper_start_dates.add(str(row["_dt"]))
        dose_history.append(d_val)

    return taper_start_dates


def classify_glp1_regimes(
    gold_daily: pd.DataFrame,
    silver_shots: pd.DataFrame | None = None,
    glp1_start_date: str = "2026-10-21",
    missed_shot_grace_days: int = 10,
) -> pd.DataFrame:
    """Annotate every spine day with `glp1_regime`, `active_dose_label`, and `transition_trigger_reason`."""
    df = gold_daily.sort_values("date").reset_index(drop=True).copy()
    sustained_taper_dates = _detect_sustained_taper_dates(silver_shots)
    earliest_taper_dt = min(sustained_taper_dates) if sustained_taper_dates else None

    regimes: list[str] = []
    dose_labels: list[str] = []
    reasons: list[str] = []

    dss_series = _num_series(df, "days_since_shot")
    dose_series = _num_series(df, "active_tirz_dose_mg")
    explicit_stop_col = (
        df["taper_or_stop_flag"].fillna(False).astype(bool)
        if "taper_or_stop_flag" in df.columns
        else pd.Series(False, index=df.index)
    )

    for idx in range(len(df)):
        ds = str(df.loc[idx, "date"])[:10]
        dss = float(dss_series.iloc[idx]) if pd.notna(dss_series.iloc[idx]) else np.nan
        dose = float(dose_series.iloc[idx]) if pd.notna(dose_series.iloc[idx]) else np.nan
        explicit_flag = bool(explicit_stop_col.iloc[idx])

        if ds < glp1_start_date:
            regimes.append("Pre_GLP1")
            dose_labels.append("None")
            reasons.append("Pre_Treatment_Baseline")
            continue

        if explicit_flag:
            regimes.append("Off_GLP1_Transition")
            dose_labels.append(f"Taper_{dose:g}mg" if np.isfinite(dose) and dose > 0 else "Off_GLP1")
            reasons.append("Explicit_Stop_Or_Taper_Flag")
        elif np.isfinite(dss) and dss > float(missed_shot_grace_days):
            regimes.append("Off_GLP1_Transition")
            dose_labels.append("Off_GLP1_Gap")
            reasons.append(f"Missed_Or_Delayed_Shot_Gt_{missed_shot_grace_days}d")
        elif earliest_taper_dt is not None and ds >= earliest_taper_dt:
            regimes.append("Off_GLP1_Transition")
            dose_labels.append(f"Taper_{dose:g}mg" if np.isfinite(dose) and dose > 0 else "Taper")
            reasons.append("Sustained_Dose_Stepdown")
        else:
            regimes.append("On_GLP1")
            dose_labels.append(f"Tirz_{dose:g}mg" if np.isfinite(dose) and dose > 0 else "On_GLP1")
            reasons.append("Active_Scheduled_GLP1")

    df["glp1_regime"] = regimes
    df["active_dose_label"] = dose_labels
    df["transition_trigger_reason"] = reasons
    return df


def _summarize_window_habits(
    sub: pd.DataFrame,
    clean_score: pd.Series,
    *,
    protein_target_g: float = 90.0,
    steps_target: float = 7500.0,
) -> dict[str, Any]:
    n_days = int(len(sub))
    nutr_present = (
        sub["nutrition_status"].astype(str) == "Present"
        if "nutrition_status" in sub.columns
        else _num_series(sub, "calories_kcal").notna()
    )
    nutr_logged = int(nutr_present.sum())
    nutr_cov = round(nutr_logged / n_days, 4) if n_days > 0 else np.nan

    prot_s = _num_series(sub, "protein_g").loc[nutr_present].dropna()
    prot_mean = round(float(prot_s.mean()), 2) if not prot_s.empty else np.nan
    prot_cons = (
        round(float((prot_s >= protein_target_g).mean() * 100.0), 1)
        if not prot_s.empty
        else np.nan
    )

    cs_s = clean_score.loc[sub.index].dropna()
    cs_mean = round(float(cs_s.mean()), 1) if not cs_s.empty else np.nan

    ew_s = _num_series(sub, "eating_window_hr").loc[nutr_present].dropna()
    ew_mean = round(float(ew_s.mean()), 2) if not ew_s.empty else np.nan

    late_s = _num_series(sub, "late_kcal_after_21").loc[nutr_present].dropna()
    late_pct = (
        round(float((late_s >= 300.0).mean() * 100.0), 1) if not late_s.empty else np.nan
    )

    steps_s = _num_series(sub, "steps").dropna()
    steps_mean = round(float(steps_s.mean()), 1) if not steps_s.empty else np.nan
    steps_cons = (
        round(float((steps_s >= steps_target).mean() * 100.0), 1)
        if not steps_s.empty
        else np.nan
    )

    ah_s = _num_series(sub, "active_hours").dropna()
    ah_mean = round(float(ah_s.mean()), 2) if not ah_s.empty else np.nan

    str_s = _num_series(sub, "strength_sessions_cnt").dropna()
    str_cnt = int(str_s.sum()) if not str_s.empty else 0

    asleep_s = _num_series(sub, "asleep_min").dropna()
    asleep_hr = round(float(asleep_s.mean() / 60.0), 2) if not asleep_s.empty else np.nan

    bed_s = _num_series(sub, "bedtime_hr_rel_mid").dropna()
    bed_sd = round(float(bed_s.std(ddof=1)), 2) if len(bed_s) >= 2 else np.nan
    early_pct = round(float((bed_s <= 1.0).mean() * 100.0), 1) if not bed_s.empty else np.nan

    hun_s = _num_series(sub, "hunger").dropna()
    hun_mean = round(float(hun_s.mean()), 2) if not hun_s.empty else np.nan

    fn_s = _num_series(sub, "food_noise").dropna()
    fn_mean = round(float(fn_s.mean()), 2) if not fn_s.empty else np.nan

    wt_s = _num_series(sub, "weight_kg").dropna()
    wt_drift = np.nan
    if len(wt_s) >= 2:
        d_first = date.fromisoformat(str(sub.loc[wt_s.index[0], "date"])[:10])
        d_last = date.fromisoformat(str(sub.loc[wt_s.index[-1], "date"])[:10])
        span = (d_last - d_first).days
        if span >= 2:
            wt_drift = round(float((wt_s.iloc[-1] - wt_s.iloc[0]) / (span / 7.0)), 3)

    return {
        "days_in_window": n_days,
        "nutrition_logged_days": nutr_logged,
        "nutrition_coverage_pct": nutr_cov,
        "protein_g_mean": prot_mean,
        "protein_target_consistency_pct": prot_cons,
        "clean_eating_score": cs_mean,
        "eating_window_hr_mean": ew_mean,
        "late_eating_after_21_pct": late_pct,
        "steps_mean": steps_mean,
        "steps_target_consistency_pct": steps_cons,
        "active_hours_mean": ah_mean,
        "strength_sessions_cnt": str_cnt,
        "asleep_hr_mean": asleep_hr,
        "bedtime_sd_hr": bed_sd,
        "earlier_bedtime_recovery_pct": early_pct,
        "hunger_mean": hun_mean,
        "food_noise_mean": fn_mean,
        "weight_drift_kg_wk": wt_drift,
    }


def _safe_diff(a: Any, b: Any) -> float:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return np.nan
    return round(float(a) - float(b), 3)


def build_gold_habit_durability(
    gold_daily: pd.DataFrame,
    silver_shots: pd.DataFrame | None = None,
    thresholds_cfg: dict[str, Any] | None = None,
    audit_logger: StatsAuditLogger | None = None,
) -> pd.DataFrame:
    """Build `Gold_Habit_Durability` with weekly rows, Peak vs Trough carryover, and Post-GLP-1 weeks."""
    if gold_daily is None or gold_daily.empty or "date" not in gold_daily.columns:
        return pd.DataFrame(columns=HABIT_DURABILITY_COLUMNS)

    glp1_start = (
        str(thresholds_cfg.get("phase_anchors", {}).get("active_glp1_phase", {}).get("start_date", "2026-10-21"))
        if thresholds_cfg
        else "2026-10-21"
    )
    df = classify_glp1_regimes(gold_daily, silver_shots=silver_shots, glp1_start_date=glp1_start)
    clean_score = _compute_daily_clean_eating_score(df)

    pre_df = df.loc[df["glp1_regime"] == "Pre_GLP1"]
    on_df = df.loc[df["glp1_regime"] == "On_GLP1"]
    off_df = df.loc[df["glp1_regime"] == "Off_GLP1_Transition"]

    pre_base = _summarize_window_habits(pre_df, clean_score) if not pre_df.empty else {}
    on_base = _summarize_window_habits(on_df, clean_score) if not on_df.empty else {}

    rows: list[dict[str, Any]] = []

    # 1. Peak Suppression (Days 0-2) vs Late-Cycle Trough (Days 4-6) Carryover while On_GLP1
    if not on_df.empty and "shot_day_int" in on_df.columns:
        sdi = _num_series(on_df, "shot_day_int")
        for label, mask in [
            ("Peak_Suppression_Days_0_2", sdi.isin([0, 1, 2])),
            ("Late_Cycle_Trough_Days_4_6", sdi.isin([4, 5, 6])),
        ]:
            sub = on_df.loc[mask]
            if sub.empty:
                continue
            stats_d = _summarize_window_habits(sub, clean_score)
            valid_n = int(max(stats_d["nutrition_logged_days"], _num_series(sub, "steps").notna().sum()))
            eff_n = compute_effective_n(_num_series(sub, "protein_g")) if stats_d["nutrition_logged_days"] >= 5 else valid_n
            _, mat, _ = classify_daily_maturity(valid_n)
            rows.append(
                {
                    "row_type": "On_GLP1_Peak_vs_Trough",
                    "period_id": label,
                    "start_date": str(sub["date"].min())[:10],
                    "display_start_date": iso_to_display_date(str(sub["date"].min())[:10]),
                    "end_date": str(sub["date"].max())[:10],
                    "display_end_date": iso_to_display_date(str(sub["date"].max())[:10]),
                    "glp1_regime": "On_GLP1",
                    "active_dose_label": str(sub["active_dose_label"].mode().iloc[0]) if not sub["active_dose_label"].empty else "On_GLP1",
                    "transition_trigger_reason": "Peak_vs_Trough_Carryover_Audit",
                    "post_glp1_week_num": np.nan,
                    **stats_d,
                    "delta_protein_vs_on_glp1": _safe_diff(stats_d["protein_g_mean"], on_base.get("protein_g_mean")),
                    "delta_clean_eating_vs_on_glp1": _safe_diff(stats_d["clean_eating_score"], on_base.get("clean_eating_score")),
                    "delta_steps_vs_on_glp1": _safe_diff(stats_d["steps_mean"], on_base.get("steps_mean")),
                    "delta_asleep_hr_vs_on_glp1": _safe_diff(stats_d["asleep_hr_mean"], on_base.get("asleep_hr_mean")),
                    "delta_hunger_vs_on_glp1": _safe_diff(stats_d["hunger_mean"], on_base.get("hunger_mean")),
                    "delta_protein_vs_pre_glp1": _safe_diff(stats_d["protein_g_mean"], pre_base.get("protein_g_mean")),
                    "delta_steps_vs_pre_glp1": _safe_diff(stats_d["steps_mean"], pre_base.get("steps_mean")),
                    "delta_asleep_hr_vs_pre_glp1": _safe_diff(stats_d["asleep_hr_mean"], pre_base.get("asleep_hr_mean")),
                    "raw_n": len(sub),
                    "valid_n": valid_n,
                    "effective_n": eff_n,
                    "missingness_pct": compute_missingness_pct(len(sub), valid_n),
                    "predictor_count": 0,
                    "stability_flag": "Descriptive_Only" if valid_n >= 5 else "Insufficient_N",
                    "maturity_level": mat,
                    "epistemic_label": "Descriptive",
                    "caveat_note": "Compares late-cycle GLP-1 trough (Days 4-6) against peak suppression (Days 0-2) to track habit durability.",
                }
            )

    # 2. Weekly Durability Ledger (Thu-Wed aligned weeks from latest 16 weeks + all GLP-1/Post-GLP-1 weeks)
    max_d = date.fromisoformat(str(df["date"].max())[:10])
    min_d_all = date.fromisoformat(str(df["date"].min())[:10])
    glp1_d = date.fromisoformat(glp1_start)
    # Include up to 8 Pre-GLP-1 weeks before 2026-10-21 plus all weeks since
    start_floor = max(min_d_all, glp1_d - timedelta(days=56))
    cur_thu = start_floor - timedelta(days=(start_floor.weekday() - 3) % 7)

    while cur_thu <= max_d:
        w_end = min(max_d, cur_thu + timedelta(days=6))
        ws_iso, we_iso = cur_thu.isoformat(), w_end.isoformat()
        sub = df.loc[(df["date"].astype(str) >= ws_iso) & (df["date"].astype(str) <= we_iso)]
        if not sub.empty:
            stats_d = _summarize_window_habits(sub, clean_score)
            regime = str(sub["glp1_regime"].iloc[-1])
            dose_lbl = str(sub["active_dose_label"].iloc[-1])
            trig = str(sub["transition_trigger_reason"].iloc[-1])
            valid_n = int(max(stats_d["nutrition_logged_days"], _num_series(sub, "steps").notna().sum()))
            rows.append(
                {
                    "row_type": "Weekly_Durability",
                    "period_id": f"HAB-WK-{ws_iso.replace('-', '')}",
                    "start_date": ws_iso,
                    "display_start_date": iso_to_display_date(ws_iso),
                    "end_date": we_iso,
                    "display_end_date": iso_to_display_date(we_iso),
                    "glp1_regime": regime,
                    "active_dose_label": dose_lbl,
                    "transition_trigger_reason": trig,
                    "post_glp1_week_num": np.nan,
                    **stats_d,
                    "delta_protein_vs_on_glp1": _safe_diff(stats_d["protein_g_mean"], on_base.get("protein_g_mean")),
                    "delta_clean_eating_vs_on_glp1": _safe_diff(stats_d["clean_eating_score"], on_base.get("clean_eating_score")),
                    "delta_steps_vs_on_glp1": _safe_diff(stats_d["steps_mean"], on_base.get("steps_mean")),
                    "delta_asleep_hr_vs_on_glp1": _safe_diff(stats_d["asleep_hr_mean"], on_base.get("asleep_hr_mean")),
                    "delta_hunger_vs_on_glp1": _safe_diff(stats_d["hunger_mean"], on_base.get("hunger_mean")),
                    "delta_protein_vs_pre_glp1": _safe_diff(stats_d["protein_g_mean"], pre_base.get("protein_g_mean")),
                    "delta_steps_vs_pre_glp1": _safe_diff(stats_d["steps_mean"], pre_base.get("steps_mean")),
                    "delta_asleep_hr_vs_pre_glp1": _safe_diff(stats_d["asleep_hr_mean"], pre_base.get("asleep_hr_mean")),
                    "raw_n": len(sub),
                    "valid_n": valid_n,
                    "effective_n": valid_n,
                    "missingness_pct": compute_missingness_pct(len(sub), valid_n),
                    "predictor_count": 0,
                    "stability_flag": "Descriptive_Only",
                    "maturity_level": "Level 0 (Baseline)",
                    "epistemic_label": "Measured",
                    "caveat_note": "Weekly descriptive habit ledger; unlogged nutrition days remain NaN.",
                }
            )
        cur_thu = cur_thu + timedelta(days=7)

    # 3. Automatic Post_GLP1_Week_1, Week_2, ... rows whenever Off_GLP1_Transition is active
    if not off_df.empty:
        off_start = date.fromisoformat(str(off_df["date"].min())[:10])
        off_end_max = date.fromisoformat(str(off_df["date"].max())[:10])
        wk_num = 1
        cur_s = off_start
        while cur_s <= off_end_max:
            cur_e = min(off_end_max, cur_s + timedelta(days=6))
            s_iso, e_iso = cur_s.isoformat(), cur_e.isoformat()
            sub = off_df.loc[(off_df["date"].astype(str) >= s_iso) & (off_df["date"].astype(str) <= e_iso)]
            if not sub.empty:
                stats_d = _summarize_window_habits(sub, clean_score)
                valid_n = int(max(stats_d["nutrition_logged_days"], _num_series(sub, "steps").notna().sum()))
                rows.append(
                    {
                        "row_type": "Post_GLP1_Week",
                        "period_id": f"Post_GLP1_Week_{wk_num}",
                        "start_date": s_iso,
                        "display_start_date": iso_to_display_date(s_iso),
                        "end_date": e_iso,
                        "display_end_date": iso_to_display_date(e_iso),
                        "glp1_regime": "Off_GLP1_Transition",
                        "active_dose_label": str(sub["active_dose_label"].iloc[-1]),
                        "transition_trigger_reason": str(sub["transition_trigger_reason"].iloc[0]),
                        "post_glp1_week_num": wk_num,
                        **stats_d,
                        "delta_protein_vs_on_glp1": _safe_diff(stats_d["protein_g_mean"], on_base.get("protein_g_mean")),
                        "delta_clean_eating_vs_on_glp1": _safe_diff(stats_d["clean_eating_score"], on_base.get("clean_eating_score")),
                        "delta_steps_vs_on_glp1": _safe_diff(stats_d["steps_mean"], on_base.get("steps_mean")),
                        "delta_asleep_hr_vs_on_glp1": _safe_diff(stats_d["asleep_hr_mean"], on_base.get("asleep_hr_mean")),
                        "delta_hunger_vs_on_glp1": _safe_diff(stats_d["hunger_mean"], on_base.get("hunger_mean")),
                        "delta_protein_vs_pre_glp1": _safe_diff(stats_d["protein_g_mean"], pre_base.get("protein_g_mean")),
                        "delta_steps_vs_pre_glp1": _safe_diff(stats_d["steps_mean"], pre_base.get("steps_mean")),
                        "delta_asleep_hr_vs_pre_glp1": _safe_diff(stats_d["asleep_hr_mean"], pre_base.get("asleep_hr_mean")),
                        "raw_n": len(sub),
                        "valid_n": valid_n,
                        "effective_n": valid_n,
                        "missingness_pct": compute_missingness_pct(len(sub), valid_n),
                        "predictor_count": 0,
                        "stability_flag": "Insufficient_N" if wk_num < 4 else "Descriptive_Only",
                        "maturity_level": "Level 0 (Baseline)",
                        "epistemic_label": "Descriptive",
                        "caveat_note": f"Post_GLP1_Week_{wk_num} durability check vs On_GLP1 and Pre_GLP1 baselines; descriptive until >=12 transition weeks.",
                    }
                )
            wk_num += 1
            cur_s = cur_s + timedelta(days=7)

    raw_out = pd.DataFrame(rows)
    n_off = int(len(off_df))
    post_wk_cnt = int((raw_out["row_type"] == "Post_GLP1_Week").sum()) if not raw_out.empty else 0
    out_df = enforce_maturity_audit(
        raw_out,
        module_name="habit_durability",
        analysis_id="HABIT_DURABILITY_GLP1_TRANSITION",
        audit_logger=audit_logger,
        default_epistemic_label="Descriptive",
        summary_cv_or_ci=f"rows={len(raw_out)}, off_glp1_days={n_off}, post_glp1_weeks={post_wk_cnt}",
        summary_caveat="Tracks Peak vs Trough carryover on GLP-1 and auto-activates Post_GLP1_Week_1..N on sustained taper or >10d shot gap.",
    )
    for col in HABIT_DURABILITY_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = np.nan
    return out_df[HABIT_DURABILITY_COLUMNS]
