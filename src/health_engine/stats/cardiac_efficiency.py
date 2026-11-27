"""Exertion-Adjusted Heart Rate (`cardiac_efficiency.py` -> `Gold_Cardiac_Efficiency`).

Rules (Prompt Section 4.4 & AGENTS.md Rules 1, 3, 4, 5):
- Join `Silver_Heart_Rate_Hourly` and `Silver_Steps_Hourly` on `(date, hour_ist)`.
- Compute:
  (a) `Gold_Cardiac_Efficiency_Buckets`: mean/median HR per standardized hourly step bucket
      (`0`, `1-249`, `250-499`, `500-999`, `1000+` steps/hr) by month and treatment phase.
  (b) `Gold_Cardiac_Efficiency`: daily `cardiac_efficiency_residual` (actual daytime `avg_hr`
      minus expected HR predicted from that day's hourly step distribution calibrated on
      `Pre_Oct_Baseline` daytime hours), plus 14d/28d rolling trends.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import add_display_date_col, to_iso_date_str
from health_engine.stats.maturity import (
    StatsAuditLogger,
    classify_daily_maturity,
    compute_effective_n,
    compute_missingness_pct,
    enforce_maturity_audit,
)

STEP_BUCKET_ORDER: tuple[str, ...] = (
    "0",
    "1-249",
    "250-499",
    "500-999",
    "1000+",
)


def _step_bucket_label(steps_val: float) -> str:
    if not np.isfinite(steps_val):
        return ""
    if steps_val <= 0.0:
        return "0"
    if steps_val <= 249.0:
        return "1-249"
    if steps_val <= 499.0:
        return "250-499"
    if steps_val <= 999.0:
        return "500-999"
    return "1000+"


def _assign_treatment_phase(ds: str, thresholds_cfg: dict[str, Any]) -> str:
    anchors = thresholds_cfg["phase_anchors"]
    pre_start = str(anchors["pre_oct_baseline"]["start_date"])
    pre_end = str(anchors["pre_oct_baseline"]["end_date"])
    glp1_start = str(anchors["active_glp1_phase"]["start_date"])

    tirz_5_start = "2026-11-19"
    for item in thresholds_cfg.get("intervention_anchors", []):
        if str(item.get("anchor_id")) == "Tirzepatide_5.0mg":
            tirz_5_start = str(item.get("anchor_date") or tirz_5_start)

    if pre_start <= ds <= pre_end:
        return "Pre_Oct_Baseline"
    if "2026-10-01" <= ds < glp1_start:
        return "Oct_VitD_Iron_Pre_GLP1"
    if glp1_start <= ds < tirz_5_start:
        return "Tirzepatide_2.5mg_Phase"
    if ds >= tirz_5_start:
        return "Tirzepatide_5.0mg_Phase"
    return "Historical_Pre_May2026"


def build_gold_cardiac_efficiency(
    gold_daily: pd.DataFrame,
    silver_hr_hourly: pd.DataFrame,
    silver_steps_hourly: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    audit_logger: StatsAuditLogger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (Gold_Cardiac_Efficiency, Gold_Cardiac_Efficiency_Buckets)."""
    daily_cols = [
        "date",
        "display_date",
        "month_yyyy_mm",
        "treatment_phase",
        "vitals_valid_coverage",
        "joined_daytime_hours",
        "actual_daytime_hr",
        "expected_daytime_hr",
        "cardiac_efficiency_residual",
        "residual_roll14d_mean",
        "residual_roll28d_mean",
        "hr_bucket_0_mean",
        "hr_bucket_1_249_mean",
        "hr_bucket_250_499_mean",
        "hr_bucket_500_999_mean",
        "hr_bucket_1000_plus_mean",
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
    bucket_cols = [
        "aggregation_scope",
        "scope_value",
        "step_bucket",
        "hours_count",
        "days_represented",
        "mean_hr_bpm",
        "median_hr_bpm",
        "iqr_hr_bpm",
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
        return (
            enforce_maturity_audit(
                pd.DataFrame(columns=daily_cols),
                module_name="cardiac_efficiency",
                analysis_id="CARDIAC_EFFICIENCY_DAILY",
                audit_logger=audit_logger,
            ),
            enforce_maturity_audit(
                pd.DataFrame(columns=bucket_cols),
                module_name="cardiac_efficiency",
                analysis_id="CARDIAC_EFFICIENCY_BUCKETS",
                audit_logger=audit_logger,
            ),
        )

    vcfg = thresholds_cfg["silver_gates"]["vitals"]
    day_start = int(vcfg.get("daytime_start_hour_ist", 8))
    day_end = int(vcfg.get("daytime_end_hour_ist", 21))
    min_day_hrs = int(vcfg.get("min_daytime_hours", 6))

    # Prepare HR hourly
    joined = pd.DataFrame()
    if not silver_hr_hourly.empty and not silver_steps_hourly.empty:
        hrh = silver_hr_hourly.copy()
        hrh["date"] = hrh["date"].map(to_iso_date_str)
        hrh["hour_int"] = pd.to_numeric(hrh["hour_ist"], errors="coerce")
        hrh["avg_hr_val"] = pd.to_numeric(hrh.get("avg_hr", pd.Series(dtype=str)).replace("", np.nan), errors="coerce")
        hrh = hrh.loc[hrh["date"].notna() & hrh["hour_int"].between(0, 23) & hrh["avg_hr_val"].notna()].copy()
        hrh["hour_int"] = hrh["hour_int"].astype(int)

        sth = silver_steps_hourly.copy()
        sth["date"] = sth["date"].map(to_iso_date_str)
        sth["hour_int"] = pd.to_numeric(sth["hour_ist"], errors="coerce")
        step_col = "fitbit_steps" if "fitbit_steps" in sth.columns else ("steps" if "steps" in sth.columns else None)
        if step_col is not None:
            sth["steps_hr_val"] = pd.to_numeric(sth[step_col].replace("", np.nan), errors="coerce")
        else:
            sth["steps_hr_val"] = np.nan
        sth = sth.loc[sth["date"].notna() & sth["hour_int"].between(0, 23) & sth["steps_hr_val"].notna()].copy()
        sth["hour_int"] = sth["hour_int"].astype(int)

        joined = hrh[["date", "hour_int", "avg_hr_val"]].merge(
            sth[["date", "hour_int", "steps_hr_val"]],
            on=["date", "hour_int"],
            how="inner",
        )

    # Filter joined hours to dates passing Silver vitals coverage when known
    valid_cov_map: dict[str, bool] = {}
    if "vitals_valid_coverage" in gold_daily.columns:
        for ds, vc in zip(gold_daily["date"].astype(str), gold_daily["vitals_valid_coverage"]):
            valid_cov_map[ds] = bool(vc) if pd.notna(vc) else False

    if not joined.empty:
        joined["vitals_valid"] = joined["date"].map(lambda d: valid_cov_map.get(str(d), True))
        joined = joined.loc[joined["vitals_valid"]].copy()
        joined["step_bucket"] = [
            _step_bucket_label(v) for v in joined["steps_hr_val"].to_numpy(dtype=float)
        ]
        joined["month_yyyy_mm"] = joined["date"].str.slice(0, 7)
        joined["treatment_phase"] = [
            _assign_treatment_phase(str(d), thresholds_cfg) for d in joined["date"]
        ]

    # (a) Build Step-Bucket Profiles by Treatment Phase and Month
    bucket_rows: list[dict[str, Any]] = []
    total_joined_hrs = len(joined)
    if not joined.empty:
        for scope_name, group_col in (("Treatment_Phase", "treatment_phase"), ("Month", "month_yyyy_mm")):
            for scope_val, sgrp in joined.groupby(group_col, sort=True):
                scope_total_hrs = len(sgrp)
                for b_lbl in STEP_BUCKET_ORDER:
                    bsub = sgrp.loc[sgrp["step_bucket"] == b_lbl]
                    hrs_n = len(bsub)
                    days_n = int(bsub["date"].nunique()) if hrs_n > 0 else 0
                    if hrs_n > 0:
                        vals = bsub["avg_hr_val"].to_numpy(dtype=float)
                        mean_hr = round(float(np.mean(vals)), 3)
                        med_hr = round(float(np.median(vals)), 3)
                        q75, q25 = np.percentile(vals, [75, 25])
                        iqr_hr = round(float(q75 - q25), 3)
                        neff = compute_effective_n(
                            bsub.groupby("date")["avg_hr_val"].mean(),
                            dates=sorted(bsub["date"].unique()),
                        )
                    else:
                        mean_hr = np.nan
                        med_hr = np.nan
                        iqr_hr = np.nan
                        neff = 0

                    _, mat_lbl, _ = classify_daily_maturity(days_n)
                    bucket_rows.append(
                        {
                            "aggregation_scope": scope_name,
                            "scope_value": str(scope_val),
                            "step_bucket": b_lbl,
                            "hours_count": int(hrs_n),
                            "days_represented": days_n,
                            "mean_hr_bpm": mean_hr,
                            "median_hr_bpm": med_hr,
                            "iqr_hr_bpm": iqr_hr,
                            "raw_n": int(scope_total_hrs),
                            "valid_n": int(hrs_n),
                            "effective_n": int(neff),
                            "missingness_pct": compute_missingness_pct(scope_total_hrs, hrs_n),
                            "predictor_count": 1,
                            "stability_flag": "Descriptive_Only" if days_n >= 15 else "Insufficient_N",
                            "maturity_level": mat_lbl,
                            "epistemic_label": "Descriptive",
                            "caveat_note": "Exertion-stratified hourly HR profile on coverage-valid days (hr_hours_covered >= 18).",
                        }
                    )

    buckets_df = enforce_maturity_audit(
        pd.DataFrame(bucket_rows, columns=bucket_cols),
        module_name="cardiac_efficiency",
        analysis_id="CARDIAC_EFFICIENCY_BUCKETS",
        audit_logger=audit_logger,
        summary_caveat="Step-bucket HR profiles stratified by month and treatment phase.",
    )

    # (b) Calibrate Daytime Expected HR per Step Bucket & Compute Daily Residuals
    daytime_joined = (
        joined.loc[joined["hour_int"].between(day_start, day_end)].copy()
        if not joined.empty
        else pd.DataFrame()
    )

    ref_bucket_hr: dict[str, float] = {}
    ref_source_note = "Pre_Oct_Baseline daytime calibration"
    if not daytime_joined.empty:
        pre_day = daytime_joined.loc[daytime_joined["treatment_phase"] == "Pre_Oct_Baseline"]
        global_means = daytime_joined.groupby("step_bucket")["avg_hr_val"].mean().to_dict()
        pre_means = pre_day.groupby("step_bucket")["avg_hr_val"].mean().to_dict()
        pre_counts = pre_day.groupby("step_bucket")["avg_hr_val"].count().to_dict()

        used_fallback = False
        overall_fallback = float(daytime_joined["avg_hr_val"].mean())
        for b_lbl in STEP_BUCKET_ORDER:
            if pre_counts.get(b_lbl, 0) >= 5 and b_lbl in pre_means:
                ref_bucket_hr[b_lbl] = float(pre_means[b_lbl])
            elif b_lbl in global_means:
                ref_bucket_hr[b_lbl] = float(global_means[b_lbl])
                used_fallback = True
            else:
                ref_bucket_hr[b_lbl] = overall_fallback
                used_fallback = True
        if used_fallback:
            ref_source_note = "Pre_Oct_Baseline daytime calibration (with global bucket fallback where pre-Oct hours < 5)"
    else:
        ref_source_note = "No joined Heart_Rate_Hourly x Steps_Hourly rows available; residuals remain NaN."

    # Aggregate daytime joined rows per date
    per_day_stats: dict[str, dict[str, Any]] = {}
    if not daytime_joined.empty:
        daytime_joined["expected_hr_hour"] = daytime_joined["step_bucket"].map(ref_bucket_hr)
        for ds, dgrp in daytime_joined.groupby("date", sort=True):
            n_hrs = int(dgrp["hour_int"].nunique())
            b_means = dgrp.groupby("step_bucket")["avg_hr_val"].mean().to_dict()
            if n_hrs >= min_day_hrs:
                act_hr = float(dgrp["avg_hr_val"].mean())
                exp_hr = float(dgrp["expected_hr_hour"].mean())
                resid = act_hr - exp_hr
            else:
                act_hr = np.nan
                exp_hr = np.nan
                resid = np.nan
            per_day_stats[str(ds)] = {
                "joined_daytime_hours": n_hrs,
                "actual_daytime_hr": round(act_hr, 4) if np.isfinite(act_hr) else np.nan,
                "expected_daytime_hr": round(exp_hr, 4) if np.isfinite(exp_hr) else np.nan,
                "cardiac_efficiency_residual": round(resid, 4) if np.isfinite(resid) else np.nan,
                "hr_bucket_0_mean": round(float(b_means["0"]), 3) if "0" in b_means else np.nan,
                "hr_bucket_1_249_mean": round(float(b_means["1-249"]), 3) if "1-249" in b_means else np.nan,
                "hr_bucket_250_499_mean": round(float(b_means["250-499"]), 3) if "250-499" in b_means else np.nan,
                "hr_bucket_500_999_mean": round(float(b_means["500-999"]), 3) if "500-999" in b_means else np.nan,
                "hr_bucket_1000_plus_mean": round(float(b_means["1000+"]), 3) if "1000+" in b_means else np.nan,
            }

    spine_sorted = gold_daily.sort_values("date").reset_index(drop=True)
    raw_days_n = len(spine_sorted)
    daily_records: list[dict[str, Any]] = []
    for ds in spine_sorted["date"].astype(str):
        p = per_day_stats.get(ds, {})
        daily_records.append(
            {
                "date": ds,
                "month_yyyy_mm": ds[:7],
                "treatment_phase": _assign_treatment_phase(ds, thresholds_cfg),
                "vitals_valid_coverage": valid_cov_map.get(ds, False),
                "joined_daytime_hours": int(p.get("joined_daytime_hours", 0)),
                "actual_daytime_hr": p.get("actual_daytime_hr", np.nan),
                "expected_daytime_hr": p.get("expected_daytime_hr", np.nan),
                "cardiac_efficiency_residual": p.get("cardiac_efficiency_residual", np.nan),
                "hr_bucket_0_mean": p.get("hr_bucket_0_mean", np.nan),
                "hr_bucket_1_249_mean": p.get("hr_bucket_1_249_mean", np.nan),
                "hr_bucket_250_499_mean": p.get("hr_bucket_250_499_mean", np.nan),
                "hr_bucket_500_999_mean": p.get("hr_bucket_500_999_mean", np.nan),
                "hr_bucket_1000_plus_mean": p.get("hr_bucket_1000_plus_mean", np.nan),
            }
        )

    daily_df = pd.DataFrame(daily_records)
    resid_s = pd.to_numeric(daily_df["cardiac_efficiency_residual"], errors="coerce")
    roll_cfg = thresholds_cfg.get("rolling_maturity_gates", {}).get("min_periods", {})
    mp14 = int(roll_cfg.get(14, roll_cfg.get("14", 5)))
    mp28 = int(roll_cfg.get(28, roll_cfg.get("28", 10)))

    daily_df["residual_roll14d_mean"] = np.round(resid_s.rolling(14, min_periods=mp14).mean(), 4)
    daily_df["residual_roll28d_mean"] = np.round(resid_s.rolling(28, min_periods=mp28).mean(), 4)

    valid_days_n = int(resid_s.notna().sum())
    eff_days_n = compute_effective_n(resid_s, dates=daily_df["date"])
    miss_pct = compute_missingness_pct(raw_days_n, valid_days_n)
    _, mat_lbl, _ = classify_daily_maturity(valid_days_n)

    daily_df["raw_n"] = raw_days_n
    daily_df["valid_n"] = valid_days_n
    daily_df["effective_n"] = eff_days_n
    daily_df["missingness_pct"] = miss_pct
    daily_df["predictor_count"] = 1
    daily_df["stability_flag"] = "Descriptive_Only" if valid_days_n >= 15 else "Insufficient_N"
    daily_df["maturity_level"] = mat_lbl
    daily_df["epistemic_label"] = "Descriptive"
    daily_df["caveat_note"] = (
        f"Negative residual indicates lower daytime HR than expected for hourly step load ({ref_source_note})."
    )

    daily_df = add_display_date_col(daily_df, "date")
    daily_df = enforce_maturity_audit(
        daily_df[daily_cols],
        module_name="cardiac_efficiency",
        analysis_id="CARDIAC_EFFICIENCY_DAILY",
        audit_logger=audit_logger,
        summary_cv_or_ci=f"valid_days={valid_days_n}, joined_hours={total_joined_hrs}",
        summary_caveat=ref_source_note,
    )
    return daily_df, buckets_df
