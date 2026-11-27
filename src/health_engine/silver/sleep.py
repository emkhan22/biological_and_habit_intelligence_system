"""Silver Gate 1: Sleep Source & Duration Filtering.

Rules:
- Filter staged sleep strictly to `source_app == 'Fitbit'` AND `stage_coverage_pct == 100`.
  Exclude unstaged Google `Fit` (`stage_coverage_pct == 0`) or partial-stage sessions, logging counts.
- `asleep_min = light_min + deep_min + rem_min + unspecified_sleep_min`.
- Exclude `< 120 min` sessions from nocturnal sleep metrics (flag as `nap_or_partial_wear`).
- Select primary nocturnal session per wake-up `date` (longest valid nocturnal session) and derive:
  - `bedtime_hr_rel_mid` (hours relative to midnight, e.g., 23:00 = -1.0, 02:00 = +2.0)
  - `wake_hr`
  - `sleep_eff_pct`
  - `sleep_fragmentation_per_hr = awake_episode_count / (asleep_min / 60)`
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from health_engine.config_loader import add_display_date_col, to_iso_date_str
from health_engine.silver.qa_logger import SilverQALogger


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", np.nan), errors="coerce")


def _parse_ist_decimal_hour(ts_val: Any) -> float:
    if ts_val is None or pd.isna(ts_val):
        return np.nan
    s = str(ts_val).strip()
    if not s:
        return np.nan
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return np.nan
    return float(dt.hour) + float(dt.minute) / 60.0 + float(dt.second) / 3600.0


def _bedtime_rel_midnight(decimal_hr: float, evening_pivot_hr: float = 18.0) -> float:
    if pd.isna(decimal_hr):
        return np.nan
    return float(decimal_hr - 24.0) if decimal_hr >= evening_pivot_hr else float(decimal_hr)


def clean_sleep_tables(
    sleep_sessions_raw: pd.DataFrame,
    sleep_daily_raw: pd.DataFrame,
    stage_timeline_raw: pd.DataFrame,
    thresholds_cfg: dict[str, Any],
    qa: SilverQALogger,
) -> dict[str, pd.DataFrame]:
    """Return Silver_Sleep_Sessions_Audited, Silver_Sleep_Nocturnal_Daily, and Silver_Sleep_Stages."""
    sleep_cfg = thresholds_cfg["silver_gates"]["sleep"]
    allowed_src = str(sleep_cfg["allowed_staged_source"])
    req_cov = float(sleep_cfg["required_stage_coverage_pct"])
    min_asleep = float(sleep_cfg["min_nocturnal_asleep_min"])
    pivot_hr = float(sleep_cfg.get("bedtime_evening_pivot_hr", 18.0))

    total_in = len(sleep_sessions_raw)
    if sleep_sessions_raw.empty:
        qa.log(
            "Sleep",
            "HealthConnectDaily!Sleep_Sessions",
            "SLEEP_UNSTAGED_EXCLUDED",
            0,
            0,
            0,
            "Empty Sleep_Sessions input",
        )
        empty = pd.DataFrame()
        return {
            "Silver_Sleep_Sessions": empty,
            "Silver_Sleep_Daily": empty,
            "Silver_Sleep_Stage_Timeline": empty,
        }

    df = sleep_sessions_raw.copy()
    df["date"] = df["date"].map(to_iso_date_str)
    df = df.loc[df["date"].notna()].copy()

    num_cols = [
        "session_duration_min",
        "sleep_onset_latency_min_est",
        "awake_episode_count",
        "awake_min",
        "longest_awake_episode_min",
        "light_min",
        "deep_min",
        "rem_min",
        "unspecified_sleep_min",
        "sleep_hr_avg",
        "sleep_hr_median",
        "sleep_hr_p95",
        "sleep_hr_max",
        "sleep_hr_sample_count",
        "stage_coverage_pct",
    ]
    for c in num_cols:
        if c in df.columns:
            df[f"{c}_num"] = _to_num(df[c])
        else:
            df[f"{c}_num"] = np.nan

    src_clean = df["source_app"].astype(str).str.strip()
    is_staged_fitbit = (src_clean == allowed_src) & np.isclose(
        df["stage_coverage_pct_num"].fillna(-1.0), req_cov
    )
    excluded_unstaged_cnt = int((~is_staged_fitbit).sum())

    stage_sum_any = df[
        ["light_min_num", "deep_min_num", "rem_min_num", "unspecified_sleep_min_num"]
    ].notna().any(axis=1)
    df["asleep_min"] = np.where(
        stage_sum_any,
        df["light_min_num"].fillna(0.0)
        + df["deep_min_num"].fillna(0.0)
        + df["rem_min_num"].fillna(0.0)
        + df["unspecified_sleep_min_num"].fillna(0.0),
        np.nan,
    )

    df["staged_fitbit_valid"] = is_staged_fitbit
    df["nap_or_partial_wear"] = is_staged_fitbit & (
        df["asleep_min"].isna() | (df["asleep_min"] < min_asleep)
    )
    short_cnt = int(df["nap_or_partial_wear"].sum())

    df["nocturnal_valid"] = df["staged_fitbit_valid"] & (~df["nap_or_partial_wear"])

    start_dec_hr = df["start_time_ist"].map(_parse_ist_decimal_hour)
    end_dec_hr = df["end_time_ist"].map(_parse_ist_decimal_hour)
    df["bedtime_hr_rel_mid"] = [
        _bedtime_rel_midnight(h, pivot_hr) for h in start_dec_hr.to_numpy()
    ]
    df["wake_hr"] = end_dec_hr
    df["sleep_eff_pct"] = np.where(
        df["session_duration_min_num"] > 0,
        (df["asleep_min"] / df["session_duration_min_num"]) * 100.0,
        np.nan,
    )
    df["sleep_fragmentation_per_hr"] = np.where(
        df["asleep_min"] > 0,
        df["awake_episode_count_num"] / (df["asleep_min"] / 60.0),
        np.nan,
    )
    df["epistemic_class"] = "Measured"

    qa.log(
        domain="Sleep",
        source_tab="HealthConnectDaily!Sleep_Sessions",
        rule_id="SLEEP_UNSTAGED_EXCLUDED",
        total_input_rows=total_in,
        flagged_or_excluded_rows=excluded_unstaged_cnt,
        valid_output_rows=int(is_staged_fitbit.sum()),
        detail_note=(
            f"Strict filter source_app=='{allowed_src}' & stage_coverage_pct=={int(req_cov)}"
        ),
    )
    qa.log(
        domain="Sleep",
        source_tab="HealthConnectDaily!Sleep_Sessions",
        rule_id="SLEEP_SHORT_SESSION_EXCLUDED",
        total_input_rows=int(is_staged_fitbit.sum()),
        flagged_or_excluded_rows=short_cnt,
        valid_output_rows=int(df["nocturnal_valid"].sum()),
        detail_note=f"Excluded staged sessions with asleep_min < {int(min_asleep)} (nap_or_partial_wear)",
    )

    # Also audit daily Sleep summary rows < 120m if Sleep tab was supplied
    if not sleep_daily_raw.empty and "sleep_mins" in sleep_daily_raw.columns:
        sd_mins = _to_num(sleep_daily_raw["sleep_mins"])
        sd_short = int((sd_mins.isna() | (sd_mins < min_asleep)).sum())
        qa.log(
            domain="Sleep",
            source_tab="HealthConnectDaily!Sleep",
            rule_id="SLEEP_DAILY_SHORT_OR_EMPTY",
            total_input_rows=len(sleep_daily_raw),
            flagged_or_excluded_rows=sd_short,
            valid_output_rows=int((sd_mins >= min_asleep).sum()),
            detail_note="Daily summary comparison audit (Gold consumes Fitbit staged Sleep_Sessions)",
        )

    # Select primary nocturnal session per wake-up date:
    # Prefer non-afternoon starts (not in [12:00, 18:00)) when available, then longest asleep_min, then latest end_time_ist
    noc = df.loc[df["nocturnal_valid"]].copy()
    if not noc.empty:
        noc["_is_overnight_window"] = ~((start_dec_hr.loc[noc.index] >= 12.0) & (start_dec_hr.loc[noc.index] < pivot_hr))
        noc["_end_sort"] = noc["end_time_ist"].astype(str)
        noc = noc.sort_values(
            by=["date", "_is_overnight_window", "asleep_min", "_end_sort"],
            ascending=[True, False, False, False],
        )
        noc["is_primary_nocturnal"] = ~noc.duplicated(subset=["date"], keep="first")
        multi_noc_cnt = int((~noc["is_primary_nocturnal"]).sum())
    else:
        noc["is_primary_nocturnal"] = pd.Series(dtype=bool)
        multi_noc_cnt = 0

    qa.log(
        domain="Sleep",
        source_tab="HealthConnectDaily!Sleep_Sessions",
        rule_id="SLEEP_PRIMARY_SESSION_SELECTED",
        total_input_rows=len(noc),
        flagged_or_excluded_rows=multi_noc_cnt,
        valid_output_rows=int(noc["is_primary_nocturnal"].sum()) if not noc.empty else 0,
        detail_note="Selected longest valid nocturnal Fitbit session per wake-up date",
    )

    df["is_primary_nocturnal"] = False
    if not noc.empty:
        primary_ids = set(noc.loc[noc["is_primary_nocturnal"]].index)
        df.loc[list(primary_ids), "is_primary_nocturnal"] = True

    # Build clean primary nocturnal daily table (preserving original session columns + derived)
    helper_drop = [c for c in df.columns if c.endswith("_num")]
    silver_sessions = add_display_date_col(df.drop(columns=helper_drop), "date")

    if not noc.empty:
        primary_df = df.loc[df["is_primary_nocturnal"]].copy()
        daily_out = pd.DataFrame(
            {
                "date": primary_df["date"],
                "session_id": primary_df["session_id"].astype(str),
                "start_time_ist": primary_df["start_time_ist"].astype(str),
                "end_time_ist": primary_df["end_time_ist"].astype(str),
                "session_duration_min": primary_df["session_duration_min_num"],
                "asleep_min": primary_df["asleep_min"],
                "light_min": primary_df["light_min_num"],
                "deep_min": primary_df["deep_min_num"],
                "rem_min": primary_df["rem_min_num"],
                "unspecified_sleep_min": primary_df["unspecified_sleep_min_num"],
                "awake_min": primary_df["awake_min_num"],
                "awake_episode_count": primary_df["awake_episode_count_num"],
                "longest_awake_episode_min": primary_df["longest_awake_episode_min_num"],
                "sleep_onset_latency_min_est": primary_df["sleep_onset_latency_min_est_num"],
                "bedtime_hr_rel_mid": primary_df["bedtime_hr_rel_mid"],
                "wake_hr": primary_df["wake_hr"],
                "sleep_eff_pct": primary_df["sleep_eff_pct"],
                "sleep_fragmentation_per_hr": primary_df["sleep_fragmentation_per_hr"],
                "sleep_hr_avg": primary_df["sleep_hr_avg_num"],
                "sleep_hr_median": primary_df["sleep_hr_median_num"],
                "sleep_hr_p95": primary_df["sleep_hr_p95_num"],
                "sleep_hr_max": primary_df["sleep_hr_max_num"],
                "sleep_hr_sample_count": primary_df["sleep_hr_sample_count_num"],
                "source_app": primary_df["source_app"].astype(str),
                "stage_coverage_pct": primary_df["stage_coverage_pct_num"],
                "epistemic_class": "Measured",
            }
        ).sort_values("date").reset_index(drop=True)
        daily_out = add_display_date_col(daily_out, "date")
    else:
        daily_out = pd.DataFrame(
            columns=[
                "date",
                "display_date",
                "session_id",
                "start_time_ist",
                "end_time_ist",
                "session_duration_min",
                "asleep_min",
                "light_min",
                "deep_min",
                "rem_min",
                "unspecified_sleep_min",
                "awake_min",
                "awake_episode_count",
                "longest_awake_episode_min",
                "sleep_onset_latency_min_est",
                "bedtime_hr_rel_mid",
                "wake_hr",
                "sleep_eff_pct",
                "sleep_fragmentation_per_hr",
                "sleep_hr_avg",
                "sleep_hr_median",
                "sleep_hr_p95",
                "sleep_hr_max",
                "sleep_hr_sample_count",
                "source_app",
                "stage_coverage_pct",
                "epistemic_class",
            ]
        )

    # Clean Sleep_Stage_Timeline restricted to valid staged Fitbit session keys (date, session_id)
    if not stage_timeline_raw.empty:
        st = stage_timeline_raw.copy()
        st["date"] = st["date"].map(to_iso_date_str)
        valid_keys = set(
            zip(
                df.loc[df["nocturnal_valid"], "date"].astype(str),
                df.loc[df["nocturnal_valid"], "session_id"].astype(str),
            )
        )
        key_series = list(zip(st["date"].astype(str), st["session_id"].astype(str)))
        keep_mask = pd.Series([k in valid_keys for k in key_series], index=st.index)
        st_valid = st.loc[keep_mask].copy()
        if "duration_min" in st_valid.columns:
            dur = _to_num(st_valid["duration_min"])
        else:
            dur = pd.Series(np.nan, index=st_valid.index)
        t0 = pd.to_datetime(st_valid["stage_start_ist"], errors="coerce")
        t1 = pd.to_datetime(st_valid["stage_end_ist"], errors="coerce")
        computed_dur = (t1 - t0).dt.total_seconds() / 60.0
        st_valid["duration_min_clean"] = dur.fillna(computed_dur)
        st_valid["epistemic_class"] = "Measured"
        st_valid = add_display_date_col(st_valid, "date")
        qa.log(
            domain="Sleep",
            source_tab="HealthConnectDaily!Sleep_Stage_Timeline",
            rule_id="SLEEP_STAGE_TIMELINE_FILTERED",
            total_input_rows=len(stage_timeline_raw),
            flagged_or_excluded_rows=int((~keep_mask).sum()),
            valid_output_rows=len(st_valid),
            detail_note="Retained timeline rows belonging to valid nocturnal Fitbit sessions",
        )
    else:
        st_valid = pd.DataFrame()

    return {
        "Silver_Sleep_Sessions": silver_sessions.reset_index(drop=True),
        "Silver_Sleep_Daily": daily_out,
        "Silver_Sleep_Stage_Timeline": st_valid.reset_index(drop=True),
    }
