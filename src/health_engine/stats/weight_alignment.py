"""Module A: 3-Grain Weight-Loss Alignment Engine (`weight_alignment.py`).

Builds:
1. Canonical deduplicated weigh-in series (`Fitdays BIA` > `HealthConnect` > `Glapp`) with
   cross-source >0.2 kg discrepancy QA logging and dynamic data-driven fasted weigh-in
   classification (`Fasted`, `Post_Meal`, `Fasted_Assumed_Morning`, `Midday_Unverified`,
   `Evening_Unverified`).
2. Grain A — `Gold_Weight_Intervals` & `Gold_Weight_Interval_Stats` (Interval-to-Interval
   velocity, BIA fat-share of loss, aligned interval predictors, scale-noise confounders,
   and Weigh-In Interval Maturity Gate `<12` / `12..23` / `>=24`).
3. Grain B — `Gold_Weight_Trend` (Time-decayed EMA `alpha = 1 - exp(-dt / 10.0)` prioritizing
   fasted anchors + rolling 28d co-movement with vitals, cardiac efficiency, visceral fat,
   and cycle length).
4. Grain C — `Gold_Weight_Clinical_Link` (Cumulative % weight lost from `2026-10-07` `108.3 kg`
   vs `7%` and `10%` MASLD steatosis-reversal milestones, BMI thresholds at `163 cm`, and
   longitudinal lab draws).
"""

from __future__ import annotations

from datetime import date, timedelta
import math
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from health_engine.config_loader import add_display_date_col, iso_to_display_date, to_iso_date_str
from health_engine.silver.qa_logger import SilverQALogger
from health_engine.stats.maturity import (
    StatsAuditLogger,
    bootstrap_bivariate_ci,
    classify_daily_maturity,
    classify_interval_maturity,
    compute_effective_n,
    compute_missingness_pct,
    enforce_maturity_audit,
)

NUTRITION_INTERVAL_PREDICTORS: frozenset[str] = frozenset(
    {
        "kcal_mean",
        "tdee_mean",
        "est_deficit_mean",
        "protein_g_mean",
        "protein_per_kg_ffm",
        "carbs_g_mean",
        "fiber_g_mean",
        "sugar_g_mean",
        "hydration_l_mean",
    }
)

ALL_INTERVAL_PREDICTORS: tuple[str, ...] = (
    "kcal_mean",
    "tdee_mean",
    "est_deficit_mean",
    "protein_g_mean",
    "protein_per_kg_ffm",
    "carbs_g_mean",
    "fiber_g_mean",
    "sugar_g_mean",
    "hydration_l_mean",
    "steps_mean",
    "zone_mins_total",
    "workout_mins_total",
    "strength_sessions_cnt",
    "asleep_hr_mean",
    "bedtime_sd_hr",
    "hrv_mean",
    "sleep_hr_mean",
    "avg_hr_mean",
    "rhr_mean",
    "spikes_per_asleep_hr_mean",
    "active_tirz_dose_mg",
    "tirz_pk_mean",
    "hunger_mean",
    "food_noise_mean",
    "gi_severity_mean",
    "meds_adherence_pct",
)


def _parse_hhmm_decimal(val: Any) -> float:
    if val is None or pd.isna(val):
        return np.nan
    s = str(val).strip()
    m = re.search(r"(\d{1,2}):(\d{2})", s)
    if not m:
        return np.nan
    return float(int(m.group(1))) + float(int(m.group(2))) / 60.0


def _num_scalar(val: Any) -> float:
    if val is None or pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def classify_fasted_weighin(
    weighin_time_ist: Any,
    first_meal_time_ist: Any,
    has_logged_meals: bool,
    morning_cutoff_hr: float = 14.0,
    evening_cutoff_hr: float = 18.0,
) -> tuple[str, bool]:
    """Classify weigh-in dynamically against logged first-meal time (no naive clock cutoff).

    Returns `(weighin_status, is_fasted_anchor)`.
    """
    w_hr = _parse_hhmm_decimal(weighin_time_ist)
    m_hr = _parse_hhmm_decimal(first_meal_time_ist)

    if has_logged_meals and np.isfinite(m_hr):
        if np.isfinite(w_hr):
            status = "Fasted" if w_hr < m_hr else "Post_Meal"
        else:
            status = "Fasted_Assumed_Morning"
    else:
        if not np.isfinite(w_hr) or w_hr < morning_cutoff_hr:
            status = "Fasted_Assumed_Morning"
        elif w_hr < evening_cutoff_hr:
            status = "Midday_Unverified"
        else:
            status = "Evening_Unverified"

    is_fasted = status in ("Fasted", "Fasted_Assumed_Morning")
    return status, is_fasted


def build_canonical_weighins(
    silver_body_comp: pd.DataFrame,
    silver_weight: pd.DataFrame,
    silver_glapp_weight: pd.DataFrame,
    silver_nutr_log: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    qa: SilverQALogger | None = None,
) -> pd.DataFrame:
    """Build canonical deduplicated weigh-in series unifying Fitdays BIA, HealthConnect, and Glapp."""
    wcfg = thresholds_cfg.get("weight_alignment", {})
    disc_thresh = float(wcfg.get("cross_source_discrepancy_kg", 0.2))
    morn_cut = float(wcfg.get("fallback_morning_cutoff_hr", 14.0))
    eve_cut = float(wcfg.get("fallback_evening_cutoff_hr", 18.0))

    # Extract earliest logged meal timestamp per date from Silver_Nutrition_Log
    first_meal_by_date: dict[str, str] = {}
    if not silver_nutr_log.empty and "date" in silver_nutr_log.columns:
        nl = silver_nutr_log.copy()
        nl["date"] = nl["date"].map(to_iso_date_str)
        nl["_mhr"] = nl["time_ist"].map(_parse_hhmm_decimal) if "time_ist" in nl.columns else np.nan
        valid_nl = nl.loc[nl["date"].notna() & nl["_mhr"].notna()].sort_values("_mhr")
        for ds, grp in valid_nl.groupby("date", sort=True):
            first_meal_by_date[str(ds)] = str(grp.iloc[0]["time_ist"]).strip()

    # Collect per-date source records
    by_date_sources: dict[str, dict[str, dict[str, Any]]] = {}

    if not silver_glapp_weight.empty and "date" in silver_glapp_weight.columns:
        for r in silver_glapp_weight.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date"))
            w = _num_scalar(r.get("weight_kg"))
            if ds and np.isfinite(w) and w > 0:
                by_date_sources.setdefault(ds, {})["Glapp"] = {
                    "weight_kg": w,
                    "time_ist": str(r.get("time_ist") or r.get("time") or "").strip(),
                    "has_bia": False,
                }

    if not silver_weight.empty and "date" in silver_weight.columns:
        for r in silver_weight.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date"))
            w = _num_scalar(r.get("weight_kg"))
            if ds and np.isfinite(w) and w > 0:
                by_date_sources.setdefault(ds, {})["HealthConnect"] = {
                    "weight_kg": w,
                    "time_ist": str(r.get("time_ist") or "").strip(),
                    "has_bia": False,
                }

    if not silver_body_comp.empty and "date" in silver_body_comp.columns:
        for r in silver_body_comp.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date"))
            w = _num_scalar(r.get("weight_kg"))
            if not ds or not np.isfinite(w) or w <= 0:
                continue
            fat_kg = _num_scalar(r.get("fat_mass_kg"))
            ffm_direct = _num_scalar(r.get("fat_free_body_weight_kg"))
            ffm_kg = (
                ffm_direct
                if np.isfinite(ffm_direct) and ffm_direct > 0
                else (round(w - fat_kg, 4) if np.isfinite(fat_kg) else np.nan)
            )
            has_bia = bool(np.isfinite(fat_kg) or np.isfinite(_num_scalar(r.get("body_fat_pct"))))
            by_date_sources.setdefault(ds, {})["Fitdays_BIA"] = {
                "weight_kg": w,
                "time_ist": str(r.get("time_ist") or "").strip(),
                "has_bia": has_bia,
                "bmi": _num_scalar(r.get("bmi")),
                "body_fat_pct": _num_scalar(r.get("body_fat_pct")),
                "fat_mass_kg": fat_kg,
                "ffm_kg": ffm_kg,
                "muscle_mass_kg": _num_scalar(r.get("muscle_mass_kg")),
                "skeletal_muscle_pct": _num_scalar(r.get("skeletal_muscle_pct")),
                "visceral_fat": _num_scalar(r.get("visceral_fat")),
                "bone_mass_kg": _num_scalar(r.get("bone_mass_kg")),
            }

    cols = [
        "date",
        "display_date",
        "time_ist",
        "canonical_source",
        "weight_kg",
        "has_bia",
        "bmi",
        "body_fat_pct",
        "fat_mass_kg",
        "ffm_kg",
        "muscle_mass_kg",
        "skeletal_muscle_pct",
        "visceral_fat",
        "bone_mass_kg",
        "first_meal_time_ist",
        "weighin_status",
        "is_fasted_anchor",
        "cross_source_max_diff_kg",
        "cross_source_discrepancy_flag",
        "epistemic_label",
    ]
    if not by_date_sources:
        return pd.DataFrame(columns=cols)

    out_rows: list[dict[str, Any]] = []
    discrepancy_notes: list[str] = []

    for ds in sorted(by_date_sources.keys()):
        srcs = by_date_sources[ds]
        weights = [float(v["weight_kg"]) for v in srcs.values()]
        max_diff = round(max(weights) - min(weights), 4) if len(weights) >= 2 else 0.0
        disc_flag = bool(len(weights) >= 2 and max_diff > (disc_thresh + 1e-6))

        if disc_flag:
            detail_pairs = ", ".join(f"{k}={v['weight_kg']}" for k, v in sorted(srcs.items()))
            discrepancy_notes.append(f"{ds} ({detail_pairs}, diff={max_diff}kg)")

        # Priority: Fitdays_BIA > HealthConnect > Glapp
        if "Fitdays_BIA" in srcs:
            chosen_src = "Fitdays_BIA"
            chosen = srcs["Fitdays_BIA"]
        elif "HealthConnect" in srcs:
            chosen_src = "HealthConnect"
            chosen = srcs["HealthConnect"]
        else:
            chosen_src = "Glapp"
            chosen = srcs["Glapp"]

        fmeal = first_meal_by_date.get(ds, "")
        has_meals = ds in first_meal_by_date
        w_status, is_fasted = classify_fasted_weighin(
            chosen.get("time_ist", ""),
            fmeal,
            has_logged_meals=has_meals,
            morning_cutoff_hr=morn_cut,
            evening_cutoff_hr=eve_cut,
        )

        out_rows.append(
            {
                "date": ds,
                "time_ist": chosen.get("time_ist", ""),
                "canonical_source": chosen_src,
                "weight_kg": round(float(chosen["weight_kg"]), 4),
                "has_bia": bool(chosen.get("has_bia", False)),
                "bmi": chosen.get("bmi", np.nan),
                "body_fat_pct": chosen.get("body_fat_pct", np.nan),
                "fat_mass_kg": chosen.get("fat_mass_kg", np.nan),
                "ffm_kg": chosen.get("ffm_kg", np.nan),
                "muscle_mass_kg": chosen.get("muscle_mass_kg", np.nan),
                "skeletal_muscle_pct": chosen.get("skeletal_muscle_pct", np.nan),
                "visceral_fat": chosen.get("visceral_fat", np.nan),
                "bone_mass_kg": chosen.get("bone_mass_kg", np.nan),
                "first_meal_time_ist": fmeal,
                "weighin_status": w_status,
                "is_fasted_anchor": is_fasted,
                "cross_source_max_diff_kg": max_diff,
                "cross_source_discrepancy_flag": disc_flag,
                "epistemic_label": "Measured",
            }
        )

    if qa is not None:
        qa.log(
            domain="Weight",
            source_tab="Body_Composition+Weight+Glapp!weight",
            rule_id="WEIGHT_CROSS_SOURCE_DISCREPANCY_GT_0_2KG",
            total_input_rows=len(out_rows),
            flagged_or_excluded_rows=len(discrepancy_notes),
            valid_output_rows=len(out_rows),
            detail_note=(
                f"Discrepancies >{disc_thresh}kg: {'; '.join(discrepancy_notes)}"
                if discrepancy_notes
                else f"All multi-source weigh-ins matched within {disc_thresh} kg"
            ),
        )

    canon_df = add_display_date_col(pd.DataFrame(out_rows), "date")
    return canon_df[cols]


def _extract_constipation_dates(
    silver_symptoms: pd.DataFrame,
    silver_glapp_se: pd.DataFrame,
) -> set[str]:
    constip_dates: set[str] = set()
    if not silver_symptoms.empty and "date" in silver_symptoms.columns:
        for r in silver_symptoms.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date"))
            txt = f"{r.get('Normalized_Symptom', '')} {r.get('Symptom', '')}".lower()
            if ds and "constipat" in txt:
                constip_dates.add(ds)
    if not silver_glapp_se.empty and "date" in silver_glapp_se.columns:
        for r in silver_glapp_se.to_dict(orient="records"):
            ds = to_iso_date_str(r.get("date"))
            txt = str(r.get("side_effect") or "").lower()
            if ds and "constipat" in txt:
                constip_dates.add(ds)
    return constip_dates


def _rank_partial_corr(
    x: np.ndarray,
    y: np.ndarray,
    covs: np.ndarray,
) -> tuple[float, float]:
    """Compute rank-based partial correlation residualizing rank(X) and rank(Y) on covariates."""
    mask = np.isfinite(x) & np.isfinite(y)
    if covs.ndim == 1:
        covs = covs.reshape(-1, 1)
    if covs.shape[1] > 0:
        mask &= np.all(np.isfinite(covs), axis=1)
    n = int(mask.sum())
    if n < 6:
        return np.nan, np.nan

    rx = sp_stats.rankdata(x[mask]).astype(float)
    ry = sp_stats.rankdata(y[mask]).astype(float)
    c_sub = covs[mask].astype(float)

    # Keep only non-constant covariate columns
    keep_cols = [j for j in range(c_sub.shape[1]) if np.std(c_sub[:, j]) > 1e-10]
    if not keep_cols or n <= len(keep_cols) + 3:
        res = sp_stats.spearmanr(x[mask], y[mask])
        return float(res.statistic), float(res.pvalue)

    design = np.column_stack([np.ones(n, dtype=float), c_sub[:, keep_cols]])
    bx, _, _, _ = np.linalg.lstsq(design, rx, rcond=None)
    by, _, _, _ = np.linalg.lstsq(design, ry, rcond=None)
    ex = rx - design @ bx
    ey = ry - design @ by

    if np.std(ex) <= 1e-12 or np.std(ey) <= 1e-12:
        return np.nan, np.nan
    r = float(np.corrcoef(ex, ey)[0, 1])
    df_deg = max(1, n - 2 - len(keep_cols))
    if abs(r) >= 1.0 - 1e-12:
        p = 0.0
    else:
        t_val = r * math.sqrt(df_deg / max(1e-12, 1.0 - r * r))
        p = float(2.0 * sp_stats.t.sf(abs(t_val), df=df_deg))
    return round(r, 4), round(p, 6)


def build_gold_weight_intervals(
    canonical_weighins: pd.DataFrame,
    gold_daily: pd.DataFrame,
    gold_sleep_spikes: pd.DataFrame,
    gold_event_windows: pd.DataFrame,
    silver_symptoms: pd.DataFrame,
    silver_glapp_se: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    *,
    all_weighins: bool = True,
    bia_anchors_only: bool = False,
    audit_logger: StatsAuditLogger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Grain A: (Gold_Weight_Intervals, Gold_Weight_Interval_Stats)."""
    wcfg = thresholds_cfg.get("weight_alignment", {})
    min_loss_fat_share = float(wcfg.get("min_loss_for_fat_share_kg", -0.3))
    igates = thresholds_cfg.get("v20_maturity_gates", {}).get("weight_intervals", {})
    lvl1_min = int(igates.get("level1_early_signal_min_intervals", 12))
    lvl2_min = int(igates.get("level2_modeling_min_intervals", 24))
    min_nutr_cov = float(igates.get("min_nutrition_coverage_pct", 0.60))

    interval_cols = [
        "interval_id",
        "start_date",
        "display_start_date",
        "end_date",
        "display_end_date",
        "in_all_weighins",
        "in_bia_anchors_only",
        "both_anchors_bia",
        "interval_days",
        "start_kg",
        "end_kg",
        "delta_kg",
        "loss_rate_kg_per_wk",
        "pct_wt_lost_per_wk",
        "fat_delta_kg",
        "ffm_delta_kg",
        "muscle_delta_kg",
        "visceral_fat_delta",
        "fat_share_of_loss",
        "kcal_mean",
        "tdee_mean",
        "est_deficit_mean",
        "protein_g_mean",
        "protein_per_kg_ffm",
        "carbs_g_mean",
        "fiber_g_mean",
        "sugar_g_mean",
        "hydration_l_mean",
        "nutrition_logged_days",
        "nutrition_coverage_pct",
        "steps_mean",
        "zone_mins_total",
        "workout_mins_total",
        "strength_sessions_cnt",
        "asleep_hr_mean",
        "bedtime_sd_hr",
        "hrv_mean",
        "sleep_hr_mean",
        "avg_hr_mean",
        "rhr_mean",
        "spikes_per_asleep_hr_mean",
        "active_tirz_dose_mg",
        "tirz_pk_mean",
        "hunger_mean",
        "food_noise_mean",
        "gi_severity_mean",
        "meds_adherence_pct",
        "start_weighin_status",
        "end_weighin_status",
        "is_fasted_interval",
        "period_at_weighin_flag",
        "period_days_in_interval",
        "constipation_48h_before_weighin_flag",
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
    stats_cols = [
        "target_metric",
        "predictor_name",
        "interval_filter_mode",
        "faster_weeks_mean",
        "slower_weeks_mean",
        "faster_minus_slower_diff",
        "group_diff_p_value",
        "spearman_rho",
        "ci_95_low",
        "ci_95_high",
        "p_value",
        "partial_rho",
        "partial_p_value",
        "ridge_std_coef",
        "ridge_cv_r2",
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

    if canonical_weighins.empty or len(canonical_weighins) < 2:
        return (
            enforce_maturity_audit(
                pd.DataFrame(columns=interval_cols),
                module_name="weight_alignment",
                analysis_id="WEIGHT_INTERVALS_LEDGER",
                audit_logger=audit_logger,
            ),
            enforce_maturity_audit(
                pd.DataFrame(columns=stats_cols),
                module_name="weight_alignment",
                analysis_id="WEIGHT_INTERVAL_STATS",
                audit_logger=audit_logger,
            ),
        )

    cw = canonical_weighins.sort_values("date").reset_index(drop=True)
    cw_records = cw.to_dict(orient="records")
    bia_records = [r for r in cw_records if bool(r.get("has_bia", False))]

    # Assemble unique anchor pairs based on requested flags
    pair_flags: dict[tuple[str, str], dict[str, Any]] = {}
    if all_weighins:
        for idx in range(len(cw_records) - 1):
            r0, r1 = cw_records[idx], cw_records[idx + 1]
            key = (str(r0["date"]), str(r1["date"]))
            pair_flags.setdefault(key, {"r0": r0, "r1": r1, "in_all": False, "in_bia": False})
            pair_flags[key]["in_all"] = True

    if bia_anchors_only or all_weighins:
        # When all_weighins is True (pipeline default), also preserve spanning BIA-to-BIA chains;
        # when bia_anchors_only=True & all_weighins=False, emit ONLY BIA-to-BIA intervals.
        for idx in range(len(bia_records) - 1):
            r0, r1 = bia_records[idx], bia_records[idx + 1]
            key = (str(r0["date"]), str(r1["date"]))
            if not bia_anchors_only and not all_weighins:
                continue
            pair_flags.setdefault(key, {"r0": r0, "r1": r1, "in_all": False, "in_bia": False})
            pair_flags[key]["in_bia"] = True

    if bia_anchors_only and not all_weighins:
        pair_flags = {k: v for k, v in pair_flags.items() if v["in_bia"]}

    # Enrich daily frame with spikes_per_asleep_hr and GI severity
    daily = gold_daily.sort_values("date").copy() if not gold_daily.empty else pd.DataFrame(columns=["date"])
    if not gold_sleep_spikes.empty and "spikes_per_asleep_hr" in gold_sleep_spikes.columns:
        sp_sub = gold_sleep_spikes[["date", "spikes_per_asleep_hr"]].drop_duplicates("date")
        daily = daily.merge(sp_sub, on="date", how="left")
    else:
        daily["spikes_per_asleep_hr"] = np.nan

    if not gold_event_windows.empty and "out_day_t_gi_symptom_score" in gold_event_windows.columns:
        gi_sub = gold_event_windows[["date", "out_day_t_gi_symptom_score"]].drop_duplicates("date")
        daily = daily.merge(gi_sub, on="date", how="left")
    else:
        daily["out_day_t_gi_symptom_score"] = daily.get("glapp_gi_score", np.nan)

    constip_dates = _extract_constipation_dates(silver_symptoms, silver_glapp_se)
    period_dates = set(
        daily.loc[daily.get("is_period_day", pd.Series(False, index=daily.index)).fillna(False).astype(bool), "date"].astype(str)
    ) if not daily.empty else set()

    # Known FFM fallback
    known_ffm_vals = cw["ffm_kg"].dropna().to_numpy(dtype=float)
    fallback_ffm = float(np.median(known_ffm_vals)) if len(known_ffm_vals) > 0 else np.nan

    sorted_keys = sorted(pair_flags.keys(), key=lambda x: (x[0], x[1]))
    raw_interval_cnt = len(sorted_keys)

    all_chain_cnt = sum(1 for v in pair_flags.values() if v["in_all"])
    bia_chain_cnt = sum(1 for v in pair_flags.values() if v["in_bia"])
    active_chain_cnt = bia_chain_cnt if (bia_anchors_only and not all_weighins) else all_chain_cnt
    _, ledger_mat_lbl, _ = classify_interval_maturity(active_chain_cnt, lvl1_min, lvl2_min)

    interval_rows: list[dict[str, Any]] = []
    for idx, (s_ds, e_ds) in enumerate(sorted_keys, start=1):
        meta = pair_flags[(s_ds, e_ds)]
        r0, r1 = meta["r0"], meta["r1"]
        d0 = date.fromisoformat(s_ds)
        d1 = date.fromisoformat(e_ds)
        int_days = max(1, (d1 - d0).days)

        start_kg = float(r0["weight_kg"])
        end_kg = float(r1["weight_kg"])
        delta_kg = round(end_kg - start_kg, 4)
        wks = int_days / 7.0
        loss_rate = round(-(delta_kg) / wks, 4)
        pct_lost_wk = round(-((delta_kg / start_kg) * 100.0) / wks, 4) if start_kg > 0 else np.nan

        both_bia = bool(r0.get("has_bia", False) and r1.get("has_bia", False))
        f0, f1 = _num_scalar(r0.get("fat_mass_kg")), _num_scalar(r1.get("fat_mass_kg"))
        ffm0, ffm1 = _num_scalar(r0.get("ffm_kg")), _num_scalar(r1.get("ffm_kg"))
        m0, m1 = _num_scalar(r0.get("muscle_mass_kg")), _num_scalar(r1.get("muscle_mass_kg"))
        vf0, vf1 = _num_scalar(r0.get("visceral_fat")), _num_scalar(r1.get("visceral_fat"))

        fat_delta = round(f1 - f0, 4) if (both_bia and np.isfinite(f0) and np.isfinite(f1)) else np.nan
        ffm_delta = round(ffm1 - ffm0, 4) if (both_bia and np.isfinite(ffm0) and np.isfinite(ffm1)) else np.nan
        mus_delta = round(m1 - m0, 4) if (both_bia and np.isfinite(m0) and np.isfinite(m1)) else np.nan
        vf_delta = round(vf1 - vf0, 4) if (both_bia and np.isfinite(vf0) and np.isfinite(vf1)) else np.nan

        if both_bia and np.isfinite(fat_delta) and delta_kg <= min_loss_fat_share:
            fat_share = round(fat_delta / delta_kg, 4)
        else:
            fat_share = np.nan

        # Slice daily features strictly over (start_date, end_date]
        sub = daily.loc[(daily["date"] > s_ds) & (daily["date"] <= e_ds)].copy() if not daily.empty else pd.DataFrame()

        def _col_mean(cname: str) -> float:
            if sub.empty or cname not in sub.columns:
                return np.nan
            s = pd.to_numeric(sub[cname], errors="coerce").dropna()
            return round(float(s.mean()), 4) if not s.empty else np.nan

        def _col_sum(cname: str) -> float:
            if sub.empty or cname not in sub.columns:
                return np.nan
            s = pd.to_numeric(sub[cname], errors="coerce").dropna()
            return round(float(s.sum()), 4) if not s.empty else np.nan

        kcal_m = _col_mean("calories_kcal")
        tdee_m = _col_mean("activity_total_kcal")
        if not sub.empty and "activity_total_kcal" in sub.columns and "calories_kcal" in sub.columns:
            td_s = pd.to_numeric(sub["activity_total_kcal"], errors="coerce")
            kc_s = pd.to_numeric(sub["calories_kcal"], errors="coerce")
            both_kc = (td_s - kc_s).dropna()
            deficit_m = round(float(both_kc.mean()), 4) if not both_kc.empty else np.nan
            nutr_days = int(kc_s.notna().sum())
        else:
            deficit_m = np.nan
            nutr_days = 0

        nutr_cov_pct = round(float(np.clip(nutr_days / float(int_days), 0.0, 1.0)), 4)
        prot_m = _col_mean("protein_g")
        ffm_ref = (
            float(np.nanmean([ffm0, ffm1]))
            if (np.isfinite(ffm0) or np.isfinite(ffm1))
            else fallback_ffm
        )
        prot_per_ffm = (
            round(prot_m / ffm_ref, 4)
            if (np.isfinite(prot_m) and np.isfinite(ffm_ref) and ffm_ref > 0)
            else np.nan
        )

        asleep_m = _col_mean("asleep_min")
        asleep_hr_m = round(asleep_m / 60.0, 4) if np.isfinite(asleep_m) else np.nan
        if not sub.empty and "bedtime_hr_rel_mid" in sub.columns:
            bt_s = pd.to_numeric(sub["bedtime_hr_rel_mid"], errors="coerce").dropna()
            bed_sd = round(float(bt_s.std(ddof=1)), 4) if len(bt_s) >= 2 else np.nan
        else:
            bed_sd = np.nan

        if not sub.empty and "active_tirz_dose_mg" in sub.columns:
            d_s = pd.to_numeric(sub["active_tirz_dose_mg"], errors="coerce").dropna()
            tirz_dose = float(d_s.iloc[-1]) if not d_s.empty else np.nan
        else:
            tirz_dose = np.nan

        if not sub.empty and "meds_taken_cnt" in sub.columns and "meds_skipped_cnt" in sub.columns:
            mt = pd.to_numeric(sub["meds_taken_cnt"], errors="coerce")
            ms = pd.to_numeric(sub["meds_skipped_cnt"], errors="coerce")
            tot_m = (mt + ms).dropna()
            denom_m = float(tot_m.sum())
            med_adh = round(float(mt.dropna().sum()) / denom_m, 4) if denom_m > 0 else np.nan
        else:
            med_adh = np.nan

        # Mandatory Scale-Noise Confounder Flags
        s_stat = str(r0["weighin_status"])
        e_stat = str(r1["weighin_status"])
        is_fasted_int = bool(r0["is_fasted_anchor"] and r1["is_fasted_anchor"])
        e_minus_1 = (d1 - timedelta(days=1)).isoformat()
        e_minus_2 = (d1 - timedelta(days=2)).isoformat()
        period_at_end = bool((e_ds in period_dates) or (e_minus_1 in period_dates))
        period_days_cnt = sum(
            1 for k in range(1, int_days + 1) if (d0 + timedelta(days=k)).isoformat() in period_dates
        )
        constip_48h = bool(
            (e_ds in constip_dates) or (e_minus_1 in constip_dates) or (e_minus_2 in constip_dates)
        )

        interval_rows.append(
            {
                "interval_id": f"WINT-{idx:03d}",
                "start_date": s_ds,
                "display_start_date": iso_to_display_date(s_ds),
                "end_date": e_ds,
                "display_end_date": iso_to_display_date(e_ds),
                "in_all_weighins": bool(meta["in_all"]),
                "in_bia_anchors_only": bool(meta["in_bia"]),
                "both_anchors_bia": both_bia,
                "interval_days": int_days,
                "start_kg": start_kg,
                "end_kg": end_kg,
                "delta_kg": delta_kg,
                "loss_rate_kg_per_wk": loss_rate,
                "pct_wt_lost_per_wk": pct_lost_wk,
                "fat_delta_kg": fat_delta,
                "ffm_delta_kg": ffm_delta,
                "muscle_delta_kg": mus_delta,
                "visceral_fat_delta": vf_delta,
                "fat_share_of_loss": fat_share,
                "kcal_mean": kcal_m,
                "tdee_mean": tdee_m,
                "est_deficit_mean": deficit_m,
                "protein_g_mean": prot_m,
                "protein_per_kg_ffm": prot_per_ffm,
                "carbs_g_mean": _col_mean("carbs_g"),
                "fiber_g_mean": _col_mean("fiber_g"),
                "sugar_g_mean": _col_mean("sugar_g"),
                "hydration_l_mean": _col_mean("hydration_liters"),
                "nutrition_logged_days": nutr_days,
                "nutrition_coverage_pct": nutr_cov_pct,
                "steps_mean": _col_mean("steps"),
                "zone_mins_total": _col_sum("zone_mins"),
                "workout_mins_total": _col_sum("workout_mins"),
                "strength_sessions_cnt": _col_sum("strength_sessions_cnt"),
                "asleep_hr_mean": asleep_hr_m,
                "bedtime_sd_hr": bed_sd,
                "hrv_mean": _col_mean("hrv_rmssd_ms"),
                "sleep_hr_mean": _col_mean("sleep_hr_avg"),
                "avg_hr_mean": _col_mean("avg_hr"),
                "rhr_mean": _col_mean("resting_hr"),
                "spikes_per_asleep_hr_mean": _col_mean("spikes_per_asleep_hr"),
                "active_tirz_dose_mg": tirz_dose,
                "tirz_pk_mean": _col_mean("tirz_pk_est"),
                "hunger_mean": _col_mean("hunger"),
                "food_noise_mean": _col_mean("food_noise"),
                "gi_severity_mean": _col_mean("out_day_t_gi_symptom_score"),
                "meds_adherence_pct": med_adh,
                "start_weighin_status": s_stat,
                "end_weighin_status": e_stat,
                "is_fasted_interval": is_fasted_int,
                "period_at_weighin_flag": period_at_end,
                "period_days_in_interval": int(period_days_cnt),
                "constipation_48h_before_weighin_flag": constip_48h,
                "raw_n": active_chain_cnt,
                "valid_n": active_chain_cnt,
                "effective_n": 0,  # filled below across chain
                "missingness_pct": round((1.0 - nutr_cov_pct) * 100.0, 2),
                "predictor_count": len(ALL_INTERVAL_PREDICTORS),
                "stability_flag": "Descriptive_Only" if active_chain_cnt >= lvl1_min else "Insufficient_N",
                "maturity_level": ledger_mat_lbl,
                "epistemic_label": "Descriptive",
                "caveat_note": (
                    f"Interval ({s_ds}, {e_ds}] ({int_days}d): end={e_stat}, "
                    f"fasted_pair={is_fasted_int}, period_flag={period_at_end}, constip_48h={constip_48h}."
                ),
            }
        )

    intervals_df = pd.DataFrame(interval_rows, columns=interval_cols)
    primary_sub = (
        intervals_df.loc[intervals_df["in_bia_anchors_only"]]
        if (bia_anchors_only and not all_weighins)
        else intervals_df.loc[intervals_df["in_all_weighins"]]
    )
    eff_int_n = compute_effective_n(primary_sub["loss_rate_kg_per_wk"])
    intervals_df["effective_n"] = eff_int_n

    intervals_df = enforce_maturity_audit(
        intervals_df,
        module_name="weight_alignment",
        analysis_id="WEIGHT_INTERVALS_LEDGER",
        audit_logger=audit_logger,
        summary_cv_or_ci=f"all_intervals={all_chain_cnt}, bia_intervals={bia_chain_cnt}",
    )

    # Build Gold_Weight_Interval_Stats enforcing Weigh-In Interval Maturity Gate
    stat_rows: list[dict[str, Any]] = []
    target_specs = [
        ("loss_rate_kg_per_wk", "all_weighins", intervals_df.loc[intervals_df["in_all_weighins"]].copy()),
        ("fat_share_of_loss", "bia_anchors_only", intervals_df.loc[intervals_df["in_bia_anchors_only"]].copy()),
    ]
    if bia_anchors_only and not all_weighins:
        target_specs = [target_specs[1]]

    for target_col, mode_lbl, base_sub in target_specs:
        raw_scope = len(base_sub)
        if raw_scope == 0:
            continue

        # Fit RidgeCV at Level 2 (>= 24 valid intervals) across predictors with sufficient coverage
        ridge_coefs: dict[str, float] = {}
        ridge_r2_val: float = np.nan
        y_all = pd.to_numeric(base_sub[target_col], errors="coerce")
        if int(y_all.notna().sum()) >= lvl2_min:
            cand_preds = [
                p
                for p in ALL_INTERVAL_PREDICTORS
                if pd.to_numeric(base_sub[p], errors="coerce").notna().sum() >= lvl2_min
            ]
            if cand_preds:
                sub_reg = base_sub[[target_col, *cand_preds]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(sub_reg) >= lvl2_min:
                    x_mat = StandardScaler().fit_transform(sub_reg[cand_preds].to_numpy(dtype=float))
                    y_vec = sub_reg[target_col].to_numpy(dtype=float)
                    if np.std(y_vec) > 1e-12:
                        rcv = RidgeCV(alphas=np.logspace(-2, 3, 25))
                        rcv.fit(x_mat, y_vec)
                        ridge_r2_val = round(float(rcv.score(x_mat, y_vec)), 4)
                        for pname, cval in zip(cand_preds, rcv.coef_):
                            ridge_coefs[pname] = round(float(cval), 4)

        for pred in ALL_INTERVAL_PREDICTORS:
            work = base_sub.copy()
            cov_note_extra = ""
            if pred in NUTRITION_INTERVAL_PREDICTORS:
                work = work.loc[pd.to_numeric(work["nutrition_coverage_pct"], errors="coerce") >= min_nutr_cov]
                cov_note_extra = f" (filtered to nutrition_coverage_pct >= {min_nutr_cov:.2f})"

            x_s = pd.to_numeric(work[pred], errors="coerce")
            y_s = pd.to_numeric(work[target_col], errors="coerce")
            valid_mask = x_s.notna() & y_s.notna()
            v_n = int(valid_mask.sum())
            lvl_num, mat_lbl, _ = classify_interval_maturity(v_n, lvl1_min, lvl2_min)
            eff_n = compute_effective_n(y_s.loc[valid_mask])

            # Faster vs Slower week medians/means (unlocked at Level >= 1)
            fast_mean = np.nan
            slow_mean = np.nan
            fs_diff = np.nan
            grp_p = np.nan
            rho = np.nan
            ci_lo = np.nan
            ci_hi = np.nan
            pval = np.nan
            part_rho = np.nan
            part_p = np.nan
            r_coef = np.nan
            r_r2 = np.nan
            stab = "Insufficient_N"
            ep_lbl = "Descriptive"

            if lvl_num >= 1 and v_n >= lvl1_min:
                xv = x_s.loc[valid_mask].to_numpy(dtype=float)
                yv = y_s.loc[valid_mask].to_numpy(dtype=float)
                med_y = float(np.median(yv))
                faster_x = xv[yv > med_y]
                slower_x = xv[yv <= med_y]
                if len(faster_x) > 0 and len(slower_x) > 0:
                    fast_mean = round(float(np.mean(faster_x)), 4)
                    slow_mean = round(float(np.mean(slower_x)), 4)
                    fs_diff = round(fast_mean - slow_mean, 4)
                    if np.std(xv) > 1e-12:
                        mwu = sp_stats.mannwhitneyu(faster_x, slower_x, alternative="two-sided")
                        grp_p = round(float(mwu.pvalue), 6)

                if np.std(xv) > 1e-12 and np.std(yv) > 1e-12:
                    sp = sp_stats.spearmanr(xv, yv)
                    rho = round(float(sp.statistic), 4)
                    pval = round(float(sp.pvalue), 6)
                    ci_lo, ci_hi, _, stab = bootstrap_bivariate_ci(
                        xv,
                        yv,
                        stat_fn=lambda a, b: float(sp_stats.spearmanr(a, b).statistic),
                        point_est=rho,
                        min_required_n=lvl1_min,
                    )
                    cov_df = pd.DataFrame(
                        {
                            "fasted": work.loc[valid_mask, "is_fasted_interval"].astype(float),
                            "period": work.loc[valid_mask, "period_at_weighin_flag"].astype(float),
                        }
                    )
                    part_rho, part_p = _rank_partial_corr(xv, yv, cov_df.to_numpy(dtype=float))
                ep_lbl = "Correlation"

            if lvl_num >= 2 and pred in ridge_coefs:
                r_coef = ridge_coefs[pred]
                r_r2 = ridge_r2_val

            caveat = (
                f"Locked at Level 0 (valid_intervals={v_n} < {lvl1_min}); correlations & regressions suppressed{cov_note_extra}."
                if lvl_num == 0
                else f"Interval association ({mode_lbl}) controlling for weigh-in status & period flags{cov_note_extra}; observational, not causal proof."
            )

            stat_rows.append(
                {
                    "target_metric": target_col,
                    "predictor_name": pred,
                    "interval_filter_mode": mode_lbl,
                    "faster_weeks_mean": fast_mean,
                    "slower_weeks_mean": slow_mean,
                    "faster_minus_slower_diff": fs_diff,
                    "group_diff_p_value": grp_p,
                    "spearman_rho": rho,
                    "ci_95_low": ci_lo,
                    "ci_95_high": ci_hi,
                    "p_value": pval,
                    "partial_rho": part_rho,
                    "partial_p_value": part_p,
                    "ridge_std_coef": r_coef,
                    "ridge_cv_r2": r_r2,
                    "raw_n": raw_scope,
                    "valid_n": v_n,
                    "effective_n": eff_n,
                    "missingness_pct": compute_missingness_pct(raw_scope, v_n),
                    "predictor_count": 3 if lvl_num >= 1 else 1,
                    "stability_flag": stab,
                    "maturity_level": mat_lbl,
                    "epistemic_label": ep_lbl,
                    "caveat_note": caveat,
                }
            )

    stats_df = enforce_maturity_audit(
        pd.DataFrame(stat_rows, columns=stats_cols),
        module_name="weight_alignment",
        analysis_id="WEIGHT_INTERVAL_STATS",
        audit_logger=audit_logger,
        suppressed_at_level0=(
            "faster_weeks_mean",
            "slower_weeks_mean",
            "faster_minus_slower_diff",
            "group_diff_p_value",
            "spearman_rho",
            "ci_95_low",
            "ci_95_high",
            "p_value",
            "partial_rho",
            "partial_p_value",
            "ridge_std_coef",
            "ridge_cv_r2",
        ),
        summary_caveat=f"Weigh-In Interval Maturity Gate enforced (min {lvl1_min} intervals for correlations, {lvl2_min} for Ridge).",
    )
    return intervals_df, stats_df


def _ols_slope_per_day(vals: np.ndarray, min_pts: int = 5) -> float:
    mask = np.isfinite(vals)
    if int(mask.sum()) < min_pts:
        return np.nan
    x = np.nonzero(mask)[0].astype(float)
    y = vals[mask].astype(float)
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0.0:
        return np.nan
    return float(np.dot(xc, y - y.mean()) / denom)


def build_gold_weight_trend(
    canonical_weighins: pd.DataFrame,
    gold_daily: pd.DataFrame,
    cardiac_daily: pd.DataFrame,
    silver_cycles: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    audit_logger: StatsAuditLogger | None = None,
) -> pd.DataFrame:
    """Build Grain B: Gold_Weight_Trend (irregular time-decayed EMA + 28d co-movement)."""
    cols = [
        "date",
        "display_date",
        "measured_weight_kg",
        "weighin_status",
        "is_fasted_anchor",
        "ema_weight_kg",
        "pct_weight_lost_from_oct7",
        "trend_loss_kg_per_wk",
        "hrv_28d_mean",
        "sleep_hr_28d_mean",
        "avg_hr_28d_mean",
        "cardiac_efficiency_residual_28d_mean",
        "visceral_fat_28d_mean",
        "cycle_length_days_latest",
        "overlap_confounder_note",
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
            module_name="weight_alignment",
            analysis_id="WEIGHT_TREND_28D",
            audit_logger=audit_logger,
        )

    wcfg = thresholds_cfg.get("weight_alignment", {})
    tau = float(wcfg.get("ema_tau_days", 10.0))
    non_fasted_disc = float(wcfg.get("non_fasted_ema_discount", 0.35))
    oct7_kg = float(thresholds_cfg["phase_anchors"]["oct7_baseline_weight_kg"])
    oct7_dt = str(thresholds_cfg["phase_anchors"].get("oct7_baseline_date", "2026-10-07"))

    anchors = [
        (str(a["anchor_id"]), date.fromisoformat(str(a["anchor_date"])))
        for a in thresholds_cfg.get("intervention_anchors", [])
        if a.get("anchor_date")
    ]

    df = gold_daily.sort_values("date").reset_index(drop=True).copy()
    if not cardiac_daily.empty and "cardiac_efficiency_residual" in cardiac_daily.columns:
        ce_sub = cardiac_daily[["date", "cardiac_efficiency_residual"]].drop_duplicates("date")
        df = df.merge(ce_sub, on="date", how="left")
    else:
        df["cardiac_efficiency_residual"] = np.nan

    # Map cycle_length_days by cycle_start from Silver_Cycles
    cyc_len_map: dict[str, float] = {}
    if not silver_cycles.empty and "cycle_start" in silver_cycles.columns:
        for r in silver_cycles.to_dict(orient="records"):
            cs = to_iso_date_str(r.get("cycle_start"))
            cl = _num_scalar(r.get("cycle_length_days"))
            if cs and np.isfinite(cl):
                cyc_len_map[cs] = cl

    # Compute time-decayed EMA across irregular weigh-ins prioritizing fasted anchors
    cw_map: dict[str, dict[str, Any]] = {}
    if not canonical_weighins.empty:
        for r in canonical_weighins.sort_values("date").to_dict(orient="records"):
            cw_map[str(r["date"])] = r

    ema_state: float = np.nan
    last_w_dt: date | None = None
    ema_series: list[float] = []
    meas_w_series: list[float] = []
    w_stat_series: list[str] = []
    w_fast_series: list[Any] = []
    cyc_latest: float = np.nan
    cyc_series: list[float] = []
    overlap_notes: list[str] = []

    for ds in df["date"].astype(str):
        d_obj = date.fromisoformat(ds)
        if ds in cyc_len_map:
            cyc_latest = cyc_len_map[ds]
        cyc_series.append(cyc_latest)

        if ds in cw_map:
            wr = cw_map[ds]
            w_val = float(wr["weight_kg"])
            is_fast = bool(wr["is_fasted_anchor"])
            w_st = str(wr["weighin_status"])
            if not np.isfinite(ema_state) or last_w_dt is None:
                ema_state = w_val
            else:
                dt_days = max(1.0, float((d_obj - last_w_dt).days))
                alpha_raw = 1.0 - math.exp(-dt_days / tau)
                alpha = alpha_raw if is_fast else (non_fasted_disc * alpha_raw)
                ema_state = ema_state + alpha * (w_val - ema_state)
            last_w_dt = d_obj
            meas_w_series.append(w_val)
            w_stat_series.append(w_st)
            w_fast_series.append(is_fast)
        else:
            meas_w_series.append(np.nan)
            w_stat_series.append("")
            w_fast_series.append(np.nan)

        ema_series.append(round(ema_state, 4) if np.isfinite(ema_state) else np.nan)

        # Overlapping clinical anchors inside trailing 28d window [d_obj - 28d, d_obj]
        in_win = [aid for aid, adt in anchors if 0 <= (d_obj - adt).days <= 28]
        overlap_notes.append(
            f"Overlapping 28d clinical anchors: {', '.join(in_win)}" if in_win else ""
        )

    ema_arr = np.asarray(ema_series, dtype=float)
    trend_loss_wk = np.full(len(df), np.nan, dtype=float)
    for i in range(len(ema_arr)):
        lo = max(0, i - 27)
        sl = _ols_slope_per_day(ema_arr[lo : i + 1], min_pts=7)
        if np.isfinite(sl):
            trend_loss_wk[i] = round(-sl * 7.0, 4)

    pct_lost = np.where(
        (df["date"].astype(str) >= oct7_dt) & np.isfinite(ema_arr),
        np.round(((oct7_kg - ema_arr) / oct7_kg) * 100.0, 4),
        np.nan,
    )

    roll_cfg = thresholds_cfg.get("rolling_maturity_gates", {}).get("min_periods", {})
    mp28 = int(roll_cfg.get(28, roll_cfg.get("28", 10)))

    def _roll28(cname: str, min_p: int = mp28) -> np.ndarray:
        if cname not in df.columns:
            return np.full(len(df), np.nan, dtype=float)
        s = pd.to_numeric(df[cname], errors="coerce")
        return np.round(s.rolling(28, min_periods=min_p).mean().to_numpy(dtype=float), 4)

    raw_n = len(df)
    valid_w_n = int(np.isfinite(np.asarray(meas_w_series, dtype=float)).sum())
    eff_n = compute_effective_n(meas_w_series, dates=df["date"])
    _, mat_lbl, _ = classify_daily_maturity(valid_w_n)

    out = pd.DataFrame(
        {
            "date": df["date"].astype(str),
            "measured_weight_kg": meas_w_series,
            "weighin_status": w_stat_series,
            "is_fasted_anchor": w_fast_series,
            "ema_weight_kg": ema_arr,
            "pct_weight_lost_from_oct7": pct_lost,
            "trend_loss_kg_per_wk": trend_loss_wk,
            "hrv_28d_mean": _roll28("hrv_rmssd_ms"),
            "sleep_hr_28d_mean": _roll28("sleep_hr_avg"),
            "avg_hr_28d_mean": _roll28("avg_hr"),
            "cardiac_efficiency_residual_28d_mean": _roll28("cardiac_efficiency_residual"),
            "visceral_fat_28d_mean": _roll28("visceral_fat", min_p=1),
            "cycle_length_days_latest": cyc_series,
            "overlap_confounder_note": overlap_notes,
            "raw_n": raw_n,
            "valid_n": valid_w_n,
            "effective_n": eff_n,
            "missingness_pct": compute_missingness_pct(raw_n, valid_w_n),
            "predictor_count": 0,
            "stability_flag": "Descriptive_Only" if valid_w_n >= 15 else "Insufficient_N",
            "maturity_level": mat_lbl,
            "epistemic_label": "Descriptive",
            "caveat_note": "Smoothed EMA weight co-movement is strictly descriptive; multiple Oct-Nov 2026 therapies overlap.",
        }
    )
    out = add_display_date_col(out, "date")
    return enforce_maturity_audit(
        out[cols],
        module_name="weight_alignment",
        analysis_id="WEIGHT_TREND_28D",
        audit_logger=audit_logger,
    )


def build_gold_weight_clinical_link(
    canonical_weighins: pd.DataFrame,
    silver_labs: pd.DataFrame,
    gold_clinical_indices: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    audit_logger: StatsAuditLogger | None = None,
) -> pd.DataFrame:
    """Build Grain C: Gold_Weight_Clinical_Link (MASLD zones, BMI classes, & lab draws)."""
    cols = [
        "record_type",
        "date",
        "display_date",
        "milestone_or_biomarker",
        "baseline_oct7_kg",
        "reference_weight_kg",
        "pct_weight_lost_from_oct7",
        "target_weight_kg",
        "kg_remaining_to_target",
        "milestone_reached",
        "biomarker_value",
        "biomarker_unit",
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
    oct7_kg = float(thresholds_cfg["phase_anchors"]["oct7_baseline_weight_kg"])
    oct7_dt = str(thresholds_cfg["phase_anchors"].get("oct7_baseline_date", "2026-10-07"))
    height_m = float(thresholds_cfg["patient_profile"]["height_cm"]) / 100.0
    wcfg = thresholds_cfg.get("weight_alignment", {})
    masld_pcts = [float(x) for x in wcfg.get("masld_milestones_pct", [7.0, 10.0])]
    bmi_cuts = [float(x) for x in wcfg.get("bmi_thresholds", [40.0, 35.0, 30.0, 25.0])]

    cw = canonical_weighins.sort_values("date").reset_index(drop=True) if not canonical_weighins.empty else pd.DataFrame()
    valid_w_n = len(cw)
    latest_date = str(cw.iloc[-1]["date"]) if valid_w_n > 0 else oct7_dt
    latest_kg = float(cw.iloc[-1]["weight_kg"]) if valid_w_n > 0 else oct7_kg
    latest_pct_lost = round(((oct7_kg - latest_kg) / oct7_kg) * 100.0, 4)

    _, mat_lbl, _ = classify_daily_maturity(valid_w_n)
    eff_n = compute_effective_n(cw["weight_kg"]) if valid_w_n > 0 else 0
    rows: list[dict[str, Any]] = []

    # 1. MASLD Steatosis-Reversal Zones (7% and 10%)
    for pct in masld_pcts:
        tgt_kg = round(oct7_kg * (1.0 - pct / 100.0), 2)
        rem_kg = round(max(0.0, latest_kg - tgt_kg), 2)
        reached = bool(latest_kg <= tgt_kg)
        rows.append(
            {
                "record_type": "MASLD_Milestone",
                "date": latest_date,
                "milestone_or_biomarker": f"MASLD_{int(pct)}pct_Steatosis_Zone",
                "baseline_oct7_kg": oct7_kg,
                "reference_weight_kg": latest_kg,
                "pct_weight_lost_from_oct7": latest_pct_lost,
                "target_weight_kg": tgt_kg,
                "kg_remaining_to_target": rem_kg,
                "milestone_reached": reached,
                "biomarker_value": np.nan,
                "biomarker_unit": "%_wt_loss",
                "raw_n": max(1, valid_w_n),
                "valid_n": valid_w_n,
                "effective_n": eff_n,
                "missingness_pct": 0.0 if valid_w_n > 0 else 100.0,
                "predictor_count": 0,
                "stability_flag": "Descriptive_Only",
                "maturity_level": mat_lbl,
                "epistemic_label": "Screening_Context",
                "caveat_note": f"Clinical hepatology reference zone ({int(pct)}% sustained weight loss from 108.3 kg baseline).",
            }
        )

    # 2. BMI Class Thresholds at height = 163.0 cm
    for bmi_t in bmi_cuts:
        tgt_kg = round(bmi_t * (height_m**2), 2)
        rem_kg = round(max(0.0, latest_kg - tgt_kg), 2)
        reached = bool(latest_kg <= tgt_kg)
        rows.append(
            {
                "record_type": "BMI_Threshold",
                "date": latest_date,
                "milestone_or_biomarker": f"BMI_Threshold_{bmi_t:.0f}",
                "baseline_oct7_kg": oct7_kg,
                "reference_weight_kg": latest_kg,
                "pct_weight_lost_from_oct7": latest_pct_lost,
                "target_weight_kg": tgt_kg,
                "kg_remaining_to_target": rem_kg,
                "milestone_reached": reached,
                "biomarker_value": round(latest_kg / (height_m**2), 2),
                "biomarker_unit": "kg/m^2",
                "raw_n": max(1, valid_w_n),
                "valid_n": valid_w_n,
                "effective_n": eff_n,
                "missingness_pct": 0.0 if valid_w_n > 0 else 100.0,
                "predictor_count": 0,
                "stability_flag": "Descriptive_Only",
                "maturity_level": mat_lbl,
                "epistemic_label": "Screening_Context",
                "caveat_note": "WHO BMI class boundary at height 163.0 cm (interpret alongside waist/WHtR and Asian-Indian risk context).",
            }
        )

    # Helper to find nearest canonical weight to a lab date
    def _nearest_weight(lab_ds: str) -> tuple[float, float]:
        if cw.empty:
            return np.nan, np.nan
        ld = date.fromisoformat(lab_ds)
        best_w = np.nan
        best_gap = 999999
        for r in cw.to_dict(orient="records"):
            wd = date.fromisoformat(str(r["date"]))
            gap = abs((ld - wd).days)
            if gap < best_gap:
                best_gap = gap
                best_w = float(r["weight_kg"])
        pct_l = round(((oct7_kg - best_w) / oct7_kg) * 100.0, 4) if np.isfinite(best_w) else np.nan
        return best_w, pct_l

    # 3. Longitudinal Lab Draws (HbA1c, Fasting Glucose, Ferritin, TSH) + Clinical Indices (HOMA-IR, TyG, TG/HDL)
    target_lab_map = {
        "hba1c": "HbA1c",
        "glycated hemoglobin (hba1c)": "HbA1c",
        "fasting blood glucose (fbs)": "Fasting_Glucose",
        "fasting plasma glucose": "Fasting_Glucose",
        "ferritin": "Ferritin",
        "serum ferritin": "Ferritin",
        "tsh": "TSH",
        "thyroid stimulating hormone (tsh)": "TSH",
    }
    if not silver_labs.empty and "Collection_Date" in silver_labs.columns:
        valid_l = silver_labs.loc[
            (~silver_labs.get("data_corruption_flag", pd.Series(False, index=silver_labs.index)).fillna(False))
            & silver_labs["Normalized_Value"].notna()
            & silver_labs["Collection_Date"].notna()
        ]
        for r in valid_l.sort_values("Collection_Date").to_dict(orient="records"):
            std_low = str(r.get("Standardized_Test_Name") or "").strip().lower()
            if std_low not in target_lab_map:
                continue
            ds = str(r["Collection_Date"])
            ref_w, ref_pct = _nearest_weight(ds)
            rows.append(
                {
                    "record_type": "Longitudinal_Lab",
                    "date": ds,
                    "milestone_or_biomarker": target_lab_map[std_low],
                    "baseline_oct7_kg": oct7_kg,
                    "reference_weight_kg": ref_w,
                    "pct_weight_lost_from_oct7": ref_pct,
                    "target_weight_kg": np.nan,
                    "kg_remaining_to_target": np.nan,
                    "milestone_reached": np.nan,
                    "biomarker_value": round(float(r["Normalized_Value"]), 4),
                    "biomarker_unit": str(r.get("Normalized_Unit") or r.get("Standardized_Unit") or ""),
                    "raw_n": len(valid_l),
                    "valid_n": len(valid_l),
                    "effective_n": len(valid_l),
                    "missingness_pct": 0.0,
                    "predictor_count": 0,
                    "stability_flag": "Descriptive_Only",
                    "maturity_level": mat_lbl,
                    "epistemic_label": "Measured",
                    "caveat_note": f"Measured lab draw on {ds} paired with nearest canonical weigh-in.",
                }
            )

    target_idx_set = {"HOMA_IR", "TyG_Index", "TG_HDL_Ratio"}
    if not gold_clinical_indices.empty and "index_name" in gold_clinical_indices.columns:
        sub_ci = gold_clinical_indices.loc[gold_clinical_indices["index_name"].isin(target_idx_set)]
        for r in sub_ci.sort_values("date").to_dict(orient="records"):
            ds = str(r["date"])
            ref_w, ref_pct = _nearest_weight(ds)
            rows.append(
                {
                    "record_type": "Longitudinal_Index",
                    "date": ds,
                    "milestone_or_biomarker": str(r["index_name"]),
                    "baseline_oct7_kg": oct7_kg,
                    "reference_weight_kg": ref_w,
                    "pct_weight_lost_from_oct7": ref_pct,
                    "target_weight_kg": np.nan,
                    "kg_remaining_to_target": np.nan,
                    "milestone_reached": np.nan,
                    "biomarker_value": round(float(r["index_value"]), 4),
                    "biomarker_unit": str(r.get("unit") or ""),
                    "raw_n": len(sub_ci),
                    "valid_n": len(sub_ci),
                    "effective_n": len(sub_ci),
                    "missingness_pct": 0.0,
                    "predictor_count": 0,
                    "stability_flag": "Descriptive_Only",
                    "maturity_level": mat_lbl,
                    "epistemic_label": "Screening_Context",
                    "caveat_note": str(r.get("clinical_caveat") or "Longitudinal metabolic index."),
                }
            )

    out_df = add_display_date_col(pd.DataFrame(rows), "date")
    return enforce_maturity_audit(
        out_df[cols],
        module_name="weight_alignment",
        analysis_id="WEIGHT_CLINICAL_LINK",
        audit_logger=audit_logger,
        default_epistemic_label="Screening_Context",
    )
