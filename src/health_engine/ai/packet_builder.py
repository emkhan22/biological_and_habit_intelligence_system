"""Strictly Smoking-Free `AI_Briefing_Packet` Builder (`packet_builder.py`).

Assembles a compact, pre-computed JSON-serializable briefing packet from validated
Silver, V1.5 Gold, V2.0 Statistical Inference, and V2.5 Progressive ML tables so Gemini:
1. NEVER performs arithmetic on raw rows, and
2. NEVER sees any data or metadata from `Micro Journal!Smoking_Log` or `Smoking_Private_Store`.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import iso_to_display_date, now_ist_str
from health_engine.io_adapter import PrivacyBoundaryViolationError

FORBIDDEN_PACKET_TOKENS: tuple[str, ...] = (
    "smoking",
    "smoke_log",
    "smoking_private",
    "cigarette",
    "craving_intensity_onset",
    "count_smoked_post",
    "smk-",
)


def _clean_scalar(val: Any) -> Any:
    """Convert numpy/pandas scalars safely to JSON-native Python types (NaN -> None)."""
    if val is None:
        return None
    if isinstance(val, (float, np.floating)):
        if math.isnan(float(val)) or math.isinf(float(val)):
            return None
        return round(float(val), 4)
    if isinstance(val, (np.integer, int)) and not isinstance(val, bool):
        return int(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s.lower() in {"nan", "nat", "none", "<na>"}:
        return None
    return s


def _df_records(df: pd.DataFrame | None, cols: list[str] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    sub = df.copy()
    if cols is not None:
        keep = [c for c in cols if c in sub.columns]
        sub = sub[keep]
    if limit is not None:
        sub = sub.tail(limit)
    out: list[dict[str, Any]] = []
    for rec in sub.to_dict(orient="records"):
        out.append({str(k): _clean_scalar(v) for k, v in rec.items()})
    return out


def assert_packet_smoking_free(obj: Any, path: str = "root") -> None:
    """Recursively verify zero keys or values contain isolated smoking tokens."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            kl = ks.lower()
            for tok in FORBIDDEN_PACKET_TOKENS:
                if tok in kl:
                    raise PrivacyBoundaryViolationError(
                        f"AGENTS.md Rule 2 violation in AI_Briefing_Packet key '{path}.{ks}' (matched '{tok}')."
                    )
            assert_packet_smoking_free(v, f"{path}.{ks}")
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            assert_packet_smoking_free(item, f"{path}[{idx}]")
    elif isinstance(obj, str):
        sl = obj.lower()
        for tok in FORBIDDEN_PACKET_TOKENS:
            if tok in sl:
                raise PrivacyBoundaryViolationError(
                    f"AGENTS.md Rule 2 violation in AI_Briefing_Packet value at '{path}' (matched '{tok}')."
                )


def _clamp_score(val: float | None) -> float | None:
    if val is None or not np.isfinite(val):
        return None
    return round(float(max(0.0, min(100.0, val))), 1)


def _compute_5_pillar_readiness(
    latest_daily: dict[str, Any],
    latest_z: dict[str, Any],
    latest_spikes_per_hr: float | None,
) -> dict[str, float | None]:
    """Deterministically compute 0-100 5-pillar readiness scores without zero-filling missing streams."""
    # 1. Sleep & Circadian Pillar
    asleep = latest_daily.get("asleep_min")
    bed_z = latest_z.get("bedtime_robust_z")
    sleep_eff = latest_daily.get("sleep_eff_pct")
    if asleep is None:
        sleep_score = None
    else:
        dur_component = min(100.0, (float(asleep) / 450.0) * 100.0)
        eff_component = float(sleep_eff) if sleep_eff is not None else 85.0
        bed_penalty = max(-20.0, min(15.0, -8.0 * float(bed_z))) if bed_z is not None else 0.0
        sleep_score = _clamp_score(0.65 * dur_component + 0.35 * eff_component + bed_penalty)

    # 2. Autonomic Recovery Pillar
    hrv_z = latest_z.get("hrv_robust_z")
    shr_z = latest_z.get("sleep_hr_robust_z")
    if latest_daily.get("hrv_rmssd_ms") is None and latest_daily.get("sleep_hr_avg") is None:
        autonomic_score = None
    else:
        base = 72.0
        if hrv_z is not None:
            base += 10.0 * float(hrv_z)
        if shr_z is not None:
            base -= 8.0 * float(shr_z)
        if latest_spikes_per_hr is not None:
            base -= min(15.0, 2.5 * float(latest_spikes_per_hr))
        autonomic_score = _clamp_score(base)

    # 3. Activity & Movement Pillar
    steps = latest_daily.get("steps")
    active_hrs = latest_daily.get("active_hours")
    if steps is None:
        activity_score = None
    else:
        step_comp = min(100.0, (float(steps) / 8500.0) * 100.0)
        ah_comp = min(100.0, (float(active_hrs) / 9.0) * 100.0) if active_hrs is not None else step_comp
        activity_score = _clamp_score(0.7 * step_comp + 0.3 * ah_comp)

    # 4. Nutrition & Metabolic Pillar (strictly None if Not_Logged)
    if str(latest_daily.get("nutrition_status", "")) != "Present" or latest_daily.get("calories_kcal") is None:
        nutrition_score = None
    else:
        prot = float(latest_daily.get("protein_g") or 0.0)
        fib = float(latest_daily.get("fiber_g") or 0.0)
        hyd = float(latest_daily.get("hydration_liters") or 0.0)
        p_sc = min(100.0, (prot / 95.0) * 100.0)
        f_sc = min(100.0, (fib / 25.0) * 100.0)
        h_sc = min(100.0, (hyd / 2.5) * 100.0) if hyd > 0 else 65.0
        nutrition_score = _clamp_score(0.5 * p_sc + 0.3 * f_sc + 0.2 * h_sc)

    # 5. Regimen Consistency Pillar
    taken = latest_daily.get("meds_taken_cnt")
    skipped = latest_daily.get("meds_skipped_cnt")
    if taken is not None and (float(taken) + float(skipped or 0)) > 0:
        adh_ratio = float(taken) / (float(taken) + float(skipped or 0))
        regimen_score = _clamp_score(adh_ratio * 100.0)
    elif latest_daily.get("active_tirz_dose_mg") is not None:
        regimen_score = 85.0
    else:
        regimen_score = None

    return {
        "sleep_circadian_score": sleep_score,
        "autonomic_recovery_score": autonomic_score,
        "activity_movement_score": activity_score,
        "nutrition_metabolic_score": nutrition_score,
        "regimen_consistency_score": regimen_score,
    }


def build_ai_briefing_packet(
    tables: dict[str, pd.DataFrame],
    silver_tables: dict[str, pd.DataFrame] | None = None,
    *,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """Build a compact, smoking-free `AI_Briefing_Packet` from Silver/Gold/Stats/ML tables."""
    silver = silver_tables or {}
    # Guard input table names immediately
    for tname in list(tables.keys()) + list(silver.keys()):
        assert_packet_smoking_free(tname, path="input_table_name")

    gold_daily = tables.get("Gold_Daily_Unified", pd.DataFrame()).copy()
    if not gold_daily.empty and "date" in gold_daily.columns:
        gold_daily = gold_daily.sort_values("date").reset_index(drop=True)
        if as_of_date:
            gold_daily = gold_daily.loc[gold_daily["date"].astype(str) <= str(as_of_date)].reset_index(drop=True)

    report_iso = (
        str(gold_daily["date"].iloc[-1])[:10]
        if not gold_daily.empty and "date" in gold_daily.columns
        else (as_of_date or "2026-11-27")
    )
    report_disp = iso_to_display_date(report_iso) or report_iso

    tail_7d = gold_daily.tail(7) if not gold_daily.empty else pd.DataFrame()
    latest_row = _df_records(gold_daily, limit=1)[0] if not gold_daily.empty else {}

    # 1. Meta & Maturity Summary
    stats_audit = tables.get("Stats_Run_Audit", pd.DataFrame())
    ml_audit = tables.get("Model_Run_Audit", pd.DataFrame())
    meta_section = {
        "report_date_iso": report_iso,
        "report_date_display": report_disp,
        "sync_timestamp_ist": now_ist_str(),
        "total_spine_days": int(len(gold_daily)),
        "coverage_7d": {
            "sleep_present_days": int((tail_7d.get("sleep_status", pd.Series(dtype=str)) == "Present").sum()) if not tail_7d.empty else 0,
            "vitals_present_days": int((tail_7d.get("vitals_status", pd.Series(dtype=str)) == "Present").sum()) if not tail_7d.empty else 0,
            "nutrition_present_days": int((tail_7d.get("nutrition_status", pd.Series(dtype=str)) == "Present").sum()) if not tail_7d.empty else 0,
        },
        "active_stats_maturity": _df_records(stats_audit, ["analysis_id", "maturity_level", "valid_n", "epistemic_label"], limit=12),
        "active_ml_maturity": _df_records(ml_audit, ["model_id", "algorithm", "maturity_level", "valid_n", "epistemic_label"], limit=10),
    }

    # 2. Today & 7-Day Phenotype, Baselines Z-scores & Spikes
    pheno_df = tables.get("Gold_Daily_Phenotypes", pd.DataFrame())
    if not pheno_df.empty and as_of_date and "date" in pheno_df.columns:
        pheno_df = pheno_df.loc[pheno_df["date"].astype(str) <= str(as_of_date)]
    latest_pheno_rec = _df_records(pheno_df, limit=1)[0] if not pheno_df.empty else {}
    pheno_7d = pheno_df.tail(7) if not pheno_df.empty else pd.DataFrame()
    pheno_mix = (
        {str(k): int(v) for k, v in pheno_7d["phenotype_label"].value_counts(dropna=True).items()}
        if not pheno_7d.empty and "phenotype_label" in pheno_7d.columns
        else {}
    )

    base_df = tables.get("Gold_Personal_Baselines", pd.DataFrame())
    if not base_df.empty and as_of_date and "date" in base_df.columns:
        base_df = base_df.loc[base_df["date"].astype(str) <= str(as_of_date)]
    latest_base = _df_records(base_df, limit=1)[0] if not base_df.empty else {}
    latest_z = {
        "hrv_classical_z": latest_base.get("hrv_z_classical_pre_oct"),
        "hrv_robust_z": latest_base.get("hrv_z_robust_pre_oct"),
        "sleep_hr_classical_z": latest_base.get("sleep_hr_z_classical_pre_oct"),
        "sleep_hr_robust_z": latest_base.get("sleep_hr_z_robust_pre_oct"),
        "daytime_hr_robust_z": latest_base.get("daytime_hr_z_robust_pre_oct"),
        "sleep_dur_robust_z": latest_base.get("sleep_dur_z_robust_pre_oct"),
        "bedtime_robust_z": latest_base.get("bedtime_z_robust_pre_oct"),
        "steps_robust_z": latest_base.get("steps_z_robust_pre_oct"),
    }

    spikes_df = tables.get("Gold_Sleep_HR_Spikes", pd.DataFrame())
    if not spikes_df.empty and as_of_date and "date" in spikes_df.columns:
        spikes_df = spikes_df.loc[spikes_df["date"].astype(str) <= str(as_of_date)]
    latest_spike_rec = _df_records(spikes_df, limit=1)[0] if not spikes_df.empty else {}
    spikes_per_hr = _clean_scalar(latest_spike_rec.get("spikes_per_asleep_hr"))

    readiness_pillars = _compute_5_pillar_readiness(latest_row, latest_z, spikes_per_hr)
    today_7d_section = {
        "latest_date_display": report_disp,
        "latest_phenotype": latest_pheno_rec.get("phenotype_label"),
        "phenotype_mix_7d": pheno_mix,
        "readiness_pillars": readiness_pillars,
        "latest_z_scores": latest_z,
        "latest_sleep_hr_spikes_per_hr": spikes_per_hr,
        "latest_biometrics": {
            k: latest_row.get(k)
            for k in (
                "steps",
                "active_hours",
                "zone_mins",
                "avg_hr",
                "daytime_avg_hr",
                "resting_hr",
                "hrv_rmssd_ms",
                "asleep_min",
                "deep_min",
                "rem_min",
                "bedtime_hr_rel_mid",
                "sleep_hr_avg",
            )
        },
    }

    # 3. Tirzepatide & Dual-Baseline Treatment Scorecards
    wave_df = tables.get("Gold_Tirz_Folded_Wave", pd.DataFrame())
    overall_wave = (
        wave_df.loc[wave_df["stratum_type"].astype(str) == "Overall"]
        if not wave_df.empty and "stratum_type" in wave_df.columns
        else pd.DataFrame()
    )
    sc_df = tables.get("Gold_Treatment_Scorecards", pd.DataFrame())
    tirz_section = {
        "shot_day_int": latest_row.get("shot_day_int"),
        "days_since_shot": latest_row.get("days_since_shot"),
        "active_tirz_dose_mg": latest_row.get("active_tirz_dose_mg"),
        "tirz_pk_est": latest_row.get("tirz_pk_est"),
        "tirz_pk_epistemic_class": latest_row.get("tirz_pk_epistemic_class", "Descriptive"),
        "folded_7d_wave_overall": _df_records(
            overall_wave,
            ["metric_name", "shot_day_int", "mean", "median", "valid_n", "epistemic_label"],
            limit=28,
        ),
        "treatment_scorecards": _df_records(
            sc_df,
            [
                "treatment_id",
                "metric_name",
                "pre_treatment_baseline_mean",
                "prior_dose_baseline_mean",
                "current_full_phase_mean",
                "delta_vs_pre_treatment",
                "delta_vs_prior_dose",
                "overlap_warning",
                "clinical_window_status",
                "maturity_level",
                "epistemic_label",
            ],
            limit=30,
        ),
    }

    # 4. 3-Grain Weight Alignment
    wint_df = tables.get("Gold_Weight_Intervals", pd.DataFrame())
    if not wint_df.empty and as_of_date and "end_date" in wint_df.columns:
        wint_df = wint_df.loc[wint_df["end_date"].astype(str) <= str(as_of_date)]
    wtrend_df = tables.get("Gold_Weight_Trend", pd.DataFrame())
    if not wtrend_df.empty and as_of_date and "date" in wtrend_df.columns:
        wtrend_df = wtrend_df.loc[wtrend_df["date"].astype(str) <= str(as_of_date)]
    wclin_df = tables.get("Gold_Weight_Clinical_Link", pd.DataFrame())

    latest_trend = _df_records(wtrend_df, limit=1)[0] if not wtrend_df.empty else {}
    masld_rows = (
        wclin_df.loc[wclin_df["record_type"].astype(str).str.contains("Milestone", case=False, na=False)]
        if not wclin_df.empty and "record_type" in wclin_df.columns
        else wclin_df
    )
    weight_section = {
        "latest_measured_weight_kg": latest_row.get("latest_weight_kg") or latest_trend.get("measured_weight_kg"),
        "latest_ema_weight_kg": latest_trend.get("ema_weight_kg"),
        "ema_trend_loss_kg_per_wk": latest_trend.get("trend_loss_kg_per_wk"),
        "pct_weight_lost_from_oct7": latest_trend.get("pct_weight_lost_from_oct7"),
        "recent_intervals": _df_records(
            wint_df,
            [
                "interval_id",
                "display_start_date",
                "display_end_date",
                "interval_days",
                "delta_kg",
                "loss_rate_kg_per_wk",
                "fat_share_of_loss",
                "nutrition_coverage_pct",
                "end_weighin_status",
                "is_fasted_interval",
                "period_at_weighin_flag",
                "constipation_48h_before_weighin_flag",
                "maturity_level",
            ],
            limit=4,
        ),
        "masld_progress": _df_records(
            masld_rows,
            [
                "milestone_or_biomarker",
                "baseline_oct7_kg",
                "reference_weight_kg",
                "pct_weight_lost_from_oct7",
                "target_weight_kg",
                "kg_remaining_to_target",
                "milestone_reached",
            ],
            limit=6,
        ),
    }

    # 5. Nutrition, Episode Chains & Recent Meal Descriptions
    def _mean_7d(col: str) -> float | None:
        if tail_7d.empty or col not in tail_7d.columns:
            return None
        s = pd.to_numeric(tail_7d[col], errors="coerce").dropna()
        return round(float(s.mean()), 2) if not s.empty else None

    nutr_log_df = silver.get("Silver_Nutrition_Log", pd.DataFrame())
    if not nutr_log_df.empty and as_of_date and "date" in nutr_log_df.columns:
        nutr_log_df = nutr_log_df.loc[nutr_log_df["date"].astype(str) <= str(as_of_date)]
    med_bio_df = tables.get("Gold_Med_Bioavailability", pd.DataFrame())
    ep_chains_df = tables.get("Gold_Episode_Chains", pd.DataFrame())
    ep_rules_df = tables.get("Gold_Episode_Rules", pd.DataFrame())

    nutrition_section = {
        "nutrition_logged_days_7d": int((tail_7d.get("nutrition_status", pd.Series(dtype=str)) == "Present").sum()) if not tail_7d.empty else 0,
        "calories_kcal_7d_mean": _mean_7d("calories_kcal"),
        "protein_g_7d_mean": _mean_7d("protein_g"),
        "fiber_g_7d_mean": _mean_7d("fiber_g"),
        "hydration_l_7d_mean": _mean_7d("hydration_liters"),
        "episode_chains": _df_records(
            ep_chains_df,
            ["chain_metric_id", "condition_group", "rate_or_mean_value", "valid_n", "epistemic_label", "caveat_note"],
            limit=10,
        ),
        "episode_rules": _df_records(
            ep_rules_df,
            ["rule_id", "antecedent", "consequent", "support", "confidence", "lift", "maturity_level"],
            limit=8,
        ),
        "recent_med_bioavailability": _df_records(
            med_bio_df,
            [
                "display_date",
                "med_or_supplement",
                "status",
                "thyronorm_fasting_gap_min",
                "thyronorm_fasting_status",
                "fat_cofactor_meal_fat_g",
                "fat_cofactor_valid_10g",
            ],
            limit=12,
        ),
        "recent_meals": _df_records(
            nutr_log_df,
            ["date", "time_ist", "meal_type", "meal_name", "calories_kcal", "protein_g", "fat_g", "fiber_g"],
            limit=20,
        ),
    }

    # 6. Verified Correlations & ML Feature Drivers
    corr_df = tables.get("Gold_Stats_Correlations", pd.DataFrame())
    if not corr_df.empty and "spearman_rho" in corr_df.columns:
        corr_sorted = corr_df.loc[pd.to_numeric(corr_df["spearman_rho"], errors="coerce").notna()].copy()
        corr_sorted["_abs"] = pd.to_numeric(corr_sorted["spearman_rho"], errors="coerce").abs()
        corr_sorted = corr_sorted.sort_values("_abs", ascending=False)
    else:
        corr_sorted = pd.DataFrame()

    ml_sum_df = tables.get("Gold_ML_Model_Summary", pd.DataFrame())
    ml_drv_df = tables.get("Gold_ML_Feature_Drivers", pd.DataFrame())
    top_drv = (
        ml_drv_df.loc[pd.to_numeric(ml_drv_df["importance_rank"], errors="coerce") <= 3]
        if not ml_drv_df.empty and "importance_rank" in ml_drv_df.columns
        else ml_drv_df
    )
    stats_ml_section = {
        "top_correlations": _df_records(
            corr_sorted,
            [
                "pair_id",
                "x_variable",
                "y_variable",
                "spearman_rho",
                "partial_rho",
                "valid_n",
                "effective_n",
                "reverse_causality_risk",
                "stability_flag",
                "epistemic_label",
            ],
            limit=10,
        ),
        "ml_models": _df_records(
            ml_sum_df,
            [
                "model_id",
                "target_metric",
                "status",
                "selected_algorithm",
                "valid_n",
                "effective_n",
                "linear_cv_r2",
                "ml_cv_r2",
                "stability_flag",
                "maturity_level",
                "epistemic_label",
            ],
            limit=8,
        ),
        "top_ml_feature_drivers": _df_records(
            top_drv,
            [
                "model_id",
                "algorithm",
                "importance_rank",
                "feature_name",
                "feature_weight",
                "direction",
                "feature_stability_flag",
                "epistemic_label",
                "ui_phrasing",
            ],
            limit=18,
        ),
    }

    # 7. Clinical Context (Indices with Mandatory Caveats, Labs, Active Meds, Open Retest Windows)
    clin_idx_df = tables.get("Gold_Clinical_Indices", pd.DataFrame())
    labs_df = silver.get("Silver_Lab_Results", tables.get("Silver_Lab_Results", pd.DataFrame()))
    meds_reg_df = silver.get("Silver_Medications", pd.DataFrame())
    procs_df = silver.get("Silver_Procedures", pd.DataFrame())

    open_windows: list[dict[str, Any]] = []
    if not sc_df.empty and "clinical_window_status" in sc_df.columns:
        retest_rows = sc_df.loc[
            sc_df["clinical_window_status"].astype(str).str.contains("Retest|Equilibration", case=False, na=False)
        ]
        seen_t: set[str] = set()
        for r in _df_records(retest_rows, ["treatment_id", "display_name", "days_on_treatment", "clinical_window_status"]):
            tid = str(r.get("treatment_id", ""))
            if tid and tid not in seen_t:
                seen_t.add(tid)
                open_windows.append(r)

    clinical_section = {
        "clinical_indices": _df_records(
            clin_idx_df,
            [
                "display_date",
                "index_name",
                "index_value",
                "unit",
                "fasting_verified",
                "age_at_test_years",
                "clinical_caveat",
                "epistemic_class",
            ],
            limit=15,
        ),
        "recent_labs": _df_records(
            labs_df,
            [
                "display_date",
                "Standardized_Test_Name",
                "Normalized_Value",
                "Normalized_Unit",
                "Specimen_Type",
            ],
            limit=20,
        ),
        "active_medications": _df_records(
            meds_reg_df,
            ["Drug_Name_Generic", "Brand_Name", "Strength", "Prescription_Date", "Status"],
            limit=15,
        ),
        "procedures_history": _df_records(
            procs_df,
            ["Procedure_Name", "Category", "Status", "Date_Performed_Or_Planned"],
            limit=10,
        ),
        "open_retest_windows": open_windows,
    }

    # 8. V3.0 Behavioral-Metabolic Audits (Habit Durability, Sweet -> Hunger, BMR/TDEE Calorie Audit)
    hab_df = tables.get("Gold_Habit_Durability", pd.DataFrame())
    sw_df = tables.get("Gold_Sweet_Hunger_Analysis", pd.DataFrame())
    cal_df = tables.get("Gold_Calorie_Balance_Audit", pd.DataFrame())

    habit_durability_section = {
        "peak_vs_trough_carryover": _df_records(
            hab_df.loc[hab_df["row_type"].astype(str) == "On_GLP1_Peak_vs_Trough"]
            if not hab_df.empty and "row_type" in hab_df.columns
            else pd.DataFrame(),
            [
                "period_id",
                "protein_g_mean",
                "clean_eating_score",
                "steps_mean",
                "hunger_mean",
                "food_noise_mean",
                "valid_n",
            ],
            limit=4,
        ),
        "post_glp1_weeks": _df_records(
            hab_df.loc[hab_df["row_type"].astype(str) == "Post_GLP1_Week"]
            if not hab_df.empty and "row_type" in hab_df.columns
            else pd.DataFrame(),
            [
                "period_id",
                "display_start_date",
                "display_end_date",
                "transition_trigger_reason",
                "protein_g_mean",
                "clean_eating_score",
                "steps_mean",
                "asleep_hr_mean",
                "hunger_mean",
                "delta_protein_vs_on_glp1",
                "delta_hunger_vs_on_glp1",
            ],
            limit=6,
        ),
        "recent_weekly_durability": _df_records(
            hab_df.loc[hab_df["row_type"].astype(str) == "Weekly_Durability"]
            if not hab_df.empty and "row_type" in hab_df.columns
            else pd.DataFrame(),
            [
                "period_id",
                "display_start_date",
                "glp1_regime",
                "protein_target_consistency_pct",
                "clean_eating_score",
                "steps_target_consistency_pct",
                "asleep_hr_mean",
            ],
            limit=4,
        ),
    }

    sweet_hunger_section = _df_records(
        sw_df,
        [
            "window_alignment",
            "outcome_metric",
            "sweet_days_n",
            "non_sweet_logged_days_n",
            "nutrition_coverage_pct",
            "sweet_window_mean",
            "non_sweet_window_mean",
            "sweet_minus_nonsweet_diff",
            "spearman_rho",
            "reverse_causality_risk",
            "pct_sweet_preceded_by_hunger_or_sleep_debt",
            "maturity_level",
            "epistemic_label",
        ],
        limit=6,
    )

    calorie_audit_section = _df_records(
        cal_df,
        [
            "window_type",
            "window_id",
            "display_start_date",
            "display_end_date",
            "window_days",
            "bmr_kcal_mean",
            "tdee_est_kcal_mean",
            "logged_kcal_mean",
            "nutrition_coverage_pct",
            "expected_delta_kg",
            "actual_delta_kg",
            "discrepancy_kcal_per_day",
            "audit_classification",
            "confounder_note",
        ],
        limit=6,
    )

    packet: dict[str, Any] = {
        "meta": meta_section,
        "today_and_7d": today_7d_section,
        "tirzepatide_and_treatments": tirz_section,
        "weight_alignment": weight_section,
        "nutrition_and_adherence": nutrition_section,
        "stats_and_ml_verified": stats_ml_section,
        "clinical_context": clinical_section,
        "habit_durability": habit_durability_section,
        "sweet_hunger": sweet_hunger_section,
        "calorie_balance_audit": calorie_audit_section,
    }

    # Final runtime deep privacy assertion
    assert_packet_smoking_free(packet)
    return packet


def serialize_briefing_packet(packet: dict[str, Any]) -> str:
    """Validate smoking-free invariant and serialize `AI_Briefing_Packet` to compact JSON."""
    assert_packet_smoking_free(packet)
    return json.dumps(packet, ensure_ascii=False, indent=2)
