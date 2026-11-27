"""Supervised Predictive Regression Suite (`regressors.py` -> `Gold_ML_Model_Summary` & `Gold_ML_Feature_Drivers`).

Models (Prompt Section 2 & AGENTS.md Rules 1, 3, 4):
1. `ML_HRV_1A_FULL`: Overnight HRV Full Predictor (`target = hrv_rmssd_ms`).
2. `ML_HRV_1B_MODIFIABLE`: Overnight HRV Modifiable-Behavior Predictor (`target = hrv_rmssd_ms`),
   strictly excluding `sleep_hr_avg` (autonomic twin) to highlight actionable habits.
3. `ML_SLEEP_2A_CORE` & `ML_SLEEP_2B_NUTRITION_OVERLAP`: Clean Sleep Duration Predictor
   (`target = asleep_min`, Fitbit staged nights >= 120m).
4. `ML_GI_3_NEXT_DAY`: Event-Window Next-Day GI Severity Predictor (`target = gi_severity_sum`),
   strictly forbidding same-day calories or same-day fat (hard anti-reverse-causality guard).
5. `ML_WEIGHT_4_VELOCITY`: Weekly Weight-Loss Velocity Predictor (`target = loss_rate_kg_per_wk`)
   on fasted intervals with `nutrition_coverage_pct >= 0.60` (gated at `valid_intervals >= 24`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from health_engine.ml.evaluator import (
    ModelAuditLogger,
    evaluate_progressive_regressor,
)

FORBIDDEN_HRV_MODIFIABLE_COLS: frozenset[str] = frozenset(
    {
        "sleep_hr_avg",
        "sleep_hr_median",
        "sleep_hr_p95",
        "sleep_hr_max",
    }
)

FORBIDDEN_GI_SAMEDAY_NUTRITION_COLS: frozenset[str] = frozenset(
    {
        "calories_kcal",
        "fat_g",
        "protein_g",
        "carbs_g",
        "fiber_g",
        "sugar_g",
        "evening_fat_after_18",
        "evening_kcal_after_18",
        "late_fat_after_21",
        "late_kcal_after_21",
    }
)

MODEL_SUMMARY_COLUMNS: list[str] = [
    "model_id",
    "model_label",
    "target_metric",
    "status",
    "selected_algorithm",
    "tree_unlock_status",
    "shap_unlock_status",
    "train_date_range",
    "validation_method",
    "raw_n",
    "valid_n",
    "effective_n",
    "predictor_count",
    "missingness_pct",
    "dummy_baseline_mae",
    "ridge_cv_r2",
    "ridge_cv_mae",
    "elasticnet_cv_r2",
    "elasticnet_cv_mae",
    "linear_cv_r2",
    "linear_cv_mae",
    "ml_cv_r2",
    "ml_cv_mae",
    "stability_score",
    "stability_flag",
    "maturity_level",
    "epistemic_label",
    "epistemic_class",
    "ui_phrasing",
]

FEATURE_DRIVER_COLUMNS: list[str] = [
    "model_id",
    "target_metric",
    "algorithm",
    "importance_rank",
    "feature_name",
    "weight_metric_type",
    "feature_weight",
    "linear_std_coef",
    "mean_abs_shap",
    "direction",
    "pct_folds_same_sign",
    "pct_boot_same_sign",
    "stability_score",
    "feature_stability_flag",
    "maturity_level",
    "epistemic_label",
    "epistemic_class",
    "ui_phrasing",
]


def assert_modifiable_hrv_features(feature_cols: list[str] | tuple[str, ...]) -> None:
    """Raise ValueError if autonomic-twin sleep HR columns enter Model 1B."""
    bad = [c for c in feature_cols if c in FORBIDDEN_HRV_MODIFIABLE_COLS]
    if bad:
        raise ValueError(
            f"Model 1B (Modifiable HRV) forbids autonomic-twin sleep HR features: {bad}"
        )


def assert_gi_no_sameday_nutrition(feature_cols: list[str] | tuple[str, ...]) -> None:
    """Raise ValueError if same-day nutrition columns enter Model 3 (prevents reverse causality)."""
    bad = [c for c in feature_cols if c in FORBIDDEN_GI_SAMEDAY_NUTRITION_COLS]
    if bad:
        raise ValueError(
            f"Reverse-causality guard: Model 3 (Next-Day GI) forbids same-day nutrition features: {bad}"
        )


def prepare_supervised_daily_matrix(
    gold_daily: pd.DataFrame,
    gold_event_windows: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Construct strictly time-ordered daily feature matrix without future or same-day leakage."""
    if gold_daily.empty:
        return pd.DataFrame()

    glp1_start = str(thresholds_cfg["phase_anchors"]["active_glp1_phase"]["start_date"])
    df = gold_daily.sort_values("date", kind="mergesort").reset_index(drop=True).copy()

    if not gold_event_windows.empty:
        ev_cols = [c for c in gold_event_windows.columns if c == "date" or c not in df.columns]
        df = df.merge(gold_event_windows[ev_cols], on="date", how="left")

    # Preceding-day (t-1) activity & meal-gap features for night ending morning t
    # Verify contiguous 1-day spacing before shifting so calendar gaps never misalign lags
    dates_dt = pd.to_datetime(df["date"], errors="coerce")
    is_contig_1d = dates_dt.diff().dt.days.eq(1)

    for src_c, lag_c in (
        ("steps", "prev_day_steps"),
        ("zone_mins", "prev_day_zone_mins"),
        ("workout_mins", "prev_day_workout_mins"),
        ("active_hours", "prev_day_active_hours"),
        ("avg_hr", "prev_day_avg_hr"),
        ("last_meal_to_bed_min", "prev_day_last_meal_to_bed_min"),
    ):
        if src_c in df.columns:
            shifted = pd.to_numeric(df[src_c], errors="coerce").shift(1)
            df[lag_c] = np.where(is_contig_1d, shifted, np.nan)
        else:
            df[lag_c] = np.nan

    # True_Zero pharmacokinetic load before GLP-1 initiation (2026-10-21)
    is_pre_glp1 = df["date"].astype(str) < glp1_start
    pk_raw = pd.to_numeric(df.get("tirz_pk_est", pd.Series(np.nan, index=df.index)), errors="coerce")
    dose_raw = pd.to_numeric(df.get("active_tirz_dose_mg", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["tirz_pk_clean"] = np.where(is_pre_glp1, 0.0, pk_raw)
    df["active_tirz_dose_clean"] = np.where(is_pre_glp1, 0.0, dose_raw)

    # Work mode one-hot (Weekend as reference)
    wm = df.get("work_mode", pd.Series("", index=df.index)).fillna("").astype(str)
    df["work_mode_office"] = wm.eq("Office_Mon_Wed").astype(float)
    df["work_mode_wfh"] = wm.eq("WFH_Tue_Thu_Fri").astype(float)

    # Cycle phase one-hot (Follicular_D1_14 / Unknown as reference)
    cp = df.get("cycle_phase", pd.Series("", index=df.index)).fillna("").astype(str)
    df["cycle_menstrual"] = cp.eq("Menstrual").astype(float)
    df["cycle_mid_extended"] = cp.eq("Mid_Extended").astype(float)
    df["cycle_late_luteal"] = cp.eq("Late_Luteal_PreMenst").astype(float)

    # Late work 00:05 numeric strictly where logged (NaN when Micro Journal unlogged!)
    if "late_work_00_05_flag" in df.columns:
        df["late_work_00_05_num"] = df["late_work_00_05_flag"].map(
            lambda v: 1.0 if v is True else (0.0 if v is False else np.nan)
        )
    else:
        df["late_work_00_05_num"] = np.nan

    # Preceding exposure aliases for Model 3 (Next-Day GI Severity)
    df["prev_evening_fat_g"] = pd.to_numeric(
        df.get("exp_t_minus_1_evening_fat_after_18", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    df["prev_evening_kcal"] = pd.to_numeric(
        df.get("exp_t_minus_1_evening_kcal_after_18", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    df["rolling_48h_fiber_g"] = pd.to_numeric(
        df.get("exp_rolling_48h_fiber_avg_g", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    df["prev_day_hydration_l"] = pd.to_numeric(
        df.get("exp_t_minus_1_hydration_liters", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    df["gi_severity_sum"] = pd.to_numeric(
        df.get("out_day_t_gi_symptom_score", df.get("glapp_gi_score", pd.Series(np.nan, index=df.index))),
        errors="coerce",
    )

    return df


def run_gi_severity_model(
    daily_matrix: pd.DataFrame,
    feature_cols: list[str],
    thresholds_cfg: dict[str, Any],
    audit_logger: ModelAuditLogger | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run Model 3 (`ML_GI_3_NEXT_DAY`) after enforcing strict anti-reverse-causality check."""
    assert_gi_no_sameday_nutrition(feature_cols)
    glp1_start = str(thresholds_cfg["phase_anchors"]["active_glp1_phase"]["start_date"])
    gi_scope = daily_matrix.loc[daily_matrix["date"].astype(str) >= glp1_start].copy()
    return evaluate_progressive_regressor(
        model_id="ML_GI_3_NEXT_DAY",
        model_label="Event-Window Next-Day GI Severity Predictor",
        df_scope=gi_scope,
        date_col="date",
        target_col="gi_severity_sum",
        feature_cols=feature_cols,
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
        is_interval_grain=False,
    )


def build_gold_ml_regressors(
    gold_daily: pd.DataFrame,
    gold_event_windows: pd.DataFrame,
    gold_weight_intervals: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    audit_logger: ModelAuditLogger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Models 1A, 1B, 2A, 2B, 3, and 4 -> (Gold_ML_Model_Summary, Gold_ML_Feature_Drivers)."""
    if gold_daily.empty:
        return (
            pd.DataFrame(columns=MODEL_SUMMARY_COLUMNS),
            pd.DataFrame(columns=FEATURE_DRIVER_COLUMNS),
        )

    dm = prepare_supervised_daily_matrix(gold_daily, gold_event_windows, thresholds_cfg)
    summaries: list[dict[str, Any]] = []
    drivers: list[dict[str, Any]] = []

    # 1. Model 1A — Overnight HRV Full Predictor
    m1a_features = [
        "sleep_hr_avg",
        "bedtime_hr_rel_mid",
        "asleep_min",
        "prev_day_steps",
        "prev_day_zone_mins",
        "tirz_pk_clean",
        "active_tirz_dose_clean",
        "work_mode_office",
        "work_mode_wfh",
        "cycle_menstrual",
        "cycle_mid_extended",
        "cycle_late_luteal",
    ]
    s1a, d1a = evaluate_progressive_regressor(
        model_id="ML_HRV_1A_FULL",
        model_label="Overnight HRV Full Predictor (With Sleep HR)",
        df_scope=dm,
        date_col="date",
        target_col="hrv_rmssd_ms",
        feature_cols=m1a_features,
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
    )
    summaries.append(s1a)
    drivers.extend(d1a)

    # 2. Model 1B — Overnight HRV Modifiable-Behavior Predictor (strictly excluding sleep_hr_avg)
    m1b_features = [
        "bedtime_hr_rel_mid",
        "asleep_min",
        "prev_day_steps",
        "prev_day_zone_mins",
        "prev_day_workout_mins",
        "prev_day_last_meal_to_bed_min",
        "tirz_pk_clean",
        "active_tirz_dose_clean",
        "work_mode_office",
        "work_mode_wfh",
        "cycle_menstrual",
        "cycle_mid_extended",
        "cycle_late_luteal",
    ]
    assert_modifiable_hrv_features(m1b_features)
    s1b, d1b = evaluate_progressive_regressor(
        model_id="ML_HRV_1B_MODIFIABLE",
        model_label="Overnight HRV Modifiable-Behavior Predictor (No Sleep HR)",
        df_scope=dm,
        date_col="date",
        target_col="hrv_rmssd_ms",
        feature_cols=m1b_features,
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
    )
    summaries.append(s1b)
    drivers.extend(d1b)

    # 3. Model 2A — Clean Sleep Duration Predictor (Fitbit staged nights >= 120m)
    asleep_s = pd.to_numeric(
        dm.get("asleep_min", pd.Series(np.nan, index=dm.index)),
        errors="coerce",
    )
    sleep_scope = dm.loc[asleep_s >= 120.0].copy()
    m2a_features = [
        "bedtime_hr_rel_mid",
        "prev_day_steps",
        "prev_day_active_hours",
        "prev_day_avg_hr",
        "work_mode_office",
        "work_mode_wfh",
        "tirz_pk_clean",
        "late_work_00_05_num",
    ]
    s2a, d2a = evaluate_progressive_regressor(
        model_id="ML_SLEEP_2A_CORE",
        model_label="Clean Nocturnal Sleep Duration Predictor",
        df_scope=sleep_scope,
        date_col="date",
        target_col="asleep_min",
        feature_cols=m2a_features,
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
    )
    summaries.append(s2a)
    drivers.extend(d2a)

    # Model 2B — Sleep Duration + Meal-to-Bed Overlap Sub-Model (gated at valid_n >= 30)
    nutr_sleep_scope = sleep_scope.loc[sleep_scope["prev_day_last_meal_to_bed_min"].notna()].copy()
    m2b_features = [
        "bedtime_hr_rel_mid",
        "prev_day_last_meal_to_bed_min",
        "prev_day_steps",
        "prev_day_avg_hr",
        "work_mode_office",
        "tirz_pk_clean",
    ]
    s2b, d2b = evaluate_progressive_regressor(
        model_id="ML_SLEEP_2B_MEAL_GAP",
        model_label="Clean Sleep Duration Predictor (Meal-Timing Overlap)",
        df_scope=nutr_sleep_scope,
        date_col="date",
        target_col="asleep_min",
        feature_cols=m2b_features,
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
    )
    summaries.append(s2b)
    drivers.extend(d2b)

    # 4. Model 3 — Event-Window Next-Day GI Severity Predictor (strictly preceding features ONLY)
    m3_features = [
        "prev_evening_fat_g",
        "prev_evening_kcal",
        "rolling_48h_fiber_g",
        "prev_day_hydration_l",
        "tirz_pk_clean",
        "shot_day_int",
        "active_tirz_dose_clean",
    ]
    s3, d3 = run_gi_severity_model(dm, m3_features, thresholds_cfg, audit_logger=audit_logger)
    summaries.append(s3)
    drivers.extend(d3)

    # 5. Model 4 — Weekly Weight-Loss Velocity Predictor (fasted intervals & nutrition_coverage_pct >= 0.60)
    min_nutr_cov = float(
        thresholds_cfg.get("v25_ml_gates", {})
        .get("weight_intervals", {})
        .get("min_nutrition_coverage_pct", 0.60)
    )
    if not gold_weight_intervals.empty:
        wi = gold_weight_intervals.copy()
        if "in_all_weighins" in wi.columns:
            wi = wi.loc[wi["in_all_weighins"].fillna(False).astype(bool)].copy()
        wi_filtered = wi.loc[
            wi["is_fasted_interval"].fillna(False).astype(bool)
            & (pd.to_numeric(wi["nutrition_coverage_pct"], errors="coerce") >= min_nutr_cov)
        ].copy()
        wi_filtered["period_flag_num"] = wi_filtered["period_at_weighin_flag"].fillna(False).astype(float)
        wi_filtered["constip_48h_num"] = wi_filtered["constipation_48h_before_weighin_flag"].fillna(False).astype(float)
        wi_filtered["tirz_dose_clean"] = pd.to_numeric(wi_filtered["active_tirz_dose_mg"], errors="coerce").fillna(0.0)
    else:
        wi_filtered = pd.DataFrame()

    m4_features = [
        "est_deficit_mean",
        "protein_g_mean",
        "carbs_g_mean",
        "steps_mean",
        "asleep_hr_mean",
        "tirz_dose_clean",
        "period_flag_num",
        "constip_48h_num",
    ]
    s4, d4 = evaluate_progressive_regressor(
        model_id="ML_WEIGHT_4_VELOCITY",
        model_label="Fasted Weekly Weight-Loss Velocity Predictor",
        df_scope=wi_filtered,
        date_col="end_date",
        target_col="loss_rate_kg_per_wk",
        feature_cols=m4_features,
        thresholds_cfg=thresholds_cfg,
        audit_logger=audit_logger,
        is_interval_grain=True,
    )
    summaries.append(s4)
    drivers.extend(d4)

    summary_df = pd.DataFrame(summaries, columns=MODEL_SUMMARY_COLUMNS)
    drivers_df = (
        pd.DataFrame(drivers, columns=FEATURE_DRIVER_COLUMNS)
        if drivers
        else pd.DataFrame(columns=FEATURE_DRIVER_COLUMNS)
    )
    return summary_df, drivers_df
