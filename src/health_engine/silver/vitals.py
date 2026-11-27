"""Silver Gate 2: Coverage-Based Vitals Filtering.

Rules:
- NEVER apply a hardcoded `hr_sample_count >= 20000` cutoff (sampling frequency jumped ~25x in May 2026).
- Compute `hr_hours_covered` per date from `Heart_Rate_Hourly` (`nunique(hour_ist)`).
- Flag `vitals_valid_coverage = True` when `hr_hours_covered >= 18`.
- Retain `hr_sample_count` and `sampling_era` (`pre_may2026_1min` vs `may2026_high_freq`) for auditability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import add_display_date_col, to_iso_date_str
from health_engine.silver.qa_logger import SilverQALogger


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", np.nan), errors="coerce")


def clean_vitals_tables(
    vitals_raw: pd.DataFrame,
    hr_hourly_raw: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    qa: SilverQALogger,
) -> pd.DataFrame:
    """Build Silver_Vitals_Daily with hr_hours_covered, vitals_valid_coverage, daytime_avg_hr, and sampling_era."""
    vcfg = thresholds_cfg["silver_gates"]["vitals"]
    min_hours = int(vcfg["min_hr_hours_covered"])
    era_cutoff = str(vcfg["high_freq_era_start_date"])
    day_start = int(vcfg.get("daytime_start_hour_ist", 8))
    day_end = int(vcfg.get("daytime_end_hour_ist", 21))
    min_day_hrs = int(vcfg.get("min_daytime_hours", 6))

    # 1. Aggregate Heart_Rate_Hourly coverage & daytime HR
    if not hr_hourly_raw.empty:
        hrh = hr_hourly_raw.copy()
        hrh["date"] = hrh["date"].map(to_iso_date_str)
        hrh["hour_num"] = _to_num(hrh["hour_ist"])
        hrh["avg_hr_num"] = _to_num(hrh["avg_hr"]) if "avg_hr" in hrh.columns else np.nan
        hrh["sample_cnt_num"] = (
            _to_num(hrh["sample_count"]) if "sample_count" in hrh.columns else np.nan
        )
        valid_hr_row = (
            hrh["date"].notna()
            & hrh["hour_num"].between(0, 23)
            & (hrh["avg_hr_num"].notna() | (hrh["sample_cnt_num"].fillna(0) > 0))
        )
        hrh_valid = hrh.loc[valid_hr_row].copy()
        hrh_valid["hour_int"] = hrh_valid["hour_num"].astype(int)

        cov_grp = (
            hrh_valid.groupby("date", as_index=False)
            .agg(
                hr_hours_covered=("hour_int", "nunique"),
                hourly_sample_sum=("sample_cnt_num", "sum"),
            )
        )

        day_sub = hrh_valid.loc[
            hrh_valid["hour_int"].between(day_start, day_end) & hrh_valid["avg_hr_num"].notna()
        ]
        if not day_sub.empty:
            day_grp = (
                day_sub.groupby("date", as_index=False)
                .agg(
                    daytime_hr_hours=("hour_int", "nunique"),
                    daytime_avg_hr_raw=("avg_hr_num", "mean"),
                )
            )
            day_grp["daytime_avg_hr"] = np.where(
                day_grp["daytime_hr_hours"] >= min_day_hrs,
                day_grp["daytime_avg_hr_raw"],
                np.nan,
            )
            cov_grp = cov_grp.merge(
                day_grp[["date", "daytime_hr_hours", "daytime_avg_hr"]],
                on="date",
                how="left",
            )
        else:
            cov_grp["daytime_hr_hours"] = 0
            cov_grp["daytime_avg_hr"] = np.nan
    else:
        cov_grp = pd.DataFrame(
            columns=["date", "hr_hours_covered", "hourly_sample_sum", "daytime_hr_hours", "daytime_avg_hr"]
        )

    # 2. Parse daily Vitals without overwriting original columns
    if not vitals_raw.empty:
        vit = vitals_raw.copy()
        vit["date"] = vit["date"].map(to_iso_date_str)
        vit = vit.loc[vit["date"].notna()].drop_duplicates(subset=["date"], keep="last").copy()
    else:
        vit = pd.DataFrame(columns=["date"])

    merged = vit.merge(cov_grp, on="date", how="outer").sort_values("date").reset_index(drop=True)
    if merged.empty:
        qa.log("Vitals", "HealthConnectDaily!Vitals", "VITALS_LOW_HOUR_COVERAGE", 0, 0, 0, "Empty Vitals")
        return merged

    merged["hr_hours_covered"] = merged["hr_hours_covered"].fillna(0).astype(int)
    merged["vitals_valid_coverage"] = merged["hr_hours_covered"] >= min_hours
    merged["sampling_era"] = np.where(
        merged["date"] < era_cutoff,
        "pre_may2026_1min",
        "may2026_high_freq",
    )

    # Numeric typed columns alongside preserved raw columns
    for col in [
        "avg_hr",
        "min_hr",
        "max_hr",
        "hr_sample_count",
        "p95_hr",
        "resting_hr",
        "hrv_rmssd_ms",
        "breathing_rate_rpm",
        "spo2_pct",
    ]:
        if col in merged.columns:
            merged[f"{col}_val"] = _to_num(merged[col])
        else:
            merged[col] = ""
            merged[f"{col}_val"] = np.nan

    # Mask daytime_avg_hr if overall daily coverage < 18h
    merged["daytime_avg_hr"] = np.where(
        merged["vitals_valid_coverage"], merged["daytime_avg_hr"], np.nan
    )
    merged["epistemic_class"] = "Measured"

    low_cov_cnt = int((~merged["vitals_valid_coverage"]).sum())
    valid_cov_cnt = int(merged["vitals_valid_coverage"].sum())
    qa.log(
        domain="Vitals",
        source_tab="HealthConnectDaily!Vitals+Heart_Rate_Hourly",
        rule_id="VITALS_LOW_HOUR_COVERAGE",
        total_input_rows=len(merged),
        flagged_or_excluded_rows=low_cov_cnt,
        valid_output_rows=valid_cov_cnt,
        detail_note=f"hr_hours_covered >= {min_hours} required across both pre/post-May 2026 sampling eras",
    )

    return add_display_date_col(merged, "date")
