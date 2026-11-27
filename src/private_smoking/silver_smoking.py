"""Isolated Silver cleaning, missingness tagging, and zero-leakage schema for Smoking_Log.

Strictly separates pre-decision **onset context** (`Silver_Smoking_Onset_Context`) from
**post-event outcomes** (`Silver_Smoking_Post_Outcomes` such as `Delay_Minutes`, `Count`,
`is_resisted_outcome`, and `Barrier_or_Reason`) so downstream models cannot leak outcomes.
"""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import re
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")

ONSET_COLUMNS: list[str] = [
    "event_key",
    "date",
    "display_date",
    "onset_time_ist",
    "onset_hour_ist",
    "work_mode",
    "craving_intensity_onset",
    "craving_intensity_missing_reason",
    "trigger_onset",
    "mood_onset",
    "epistemic_class",
]

POST_OUTCOME_COLUMNS: list[str] = [
    "event_key",
    "date",
    "display_date",
    "event_type",
    "is_resisted_outcome",
    "count_smoked_post",
    "count_status_tag",
    "delay_minutes_post",
    "delay_missing_reason",
    "barrier_or_reason_post",
    "epistemic_class",
]


def _to_iso_date(val: Any) -> str | None:
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return None
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    if len(s) >= 10 and s[2] == "-" and s[5] == "-":
        try:
            return datetime.strptime(s[:10], "%d-%m-%Y").strftime("%Y-%m-%d")
        except ValueError:
            pass
    parsed = pd.to_datetime(s, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _to_display_date(iso_ds: str | None) -> str | None:
    if not iso_ds or len(iso_ds) < 10:
        return None
    return f"{iso_ds[8:10]}-{iso_ds[5:7]}-{iso_ds[0:4]}"


def _norm_time_hhmm(val: Any) -> str:
    if val is None or pd.isna(val):
        return ""
    m = re.search(r"(\d{1,2}):(\d{2})", str(val).strip())
    return f"{int(m.group(1)):02d}:{m.group(2)}" if m else ""


def _work_mode(iso_ds: str) -> str:
    wd = date.fromisoformat(iso_ds).weekday()
    if wd in (0, 2):
        return "Office_Mon_Wed"
    if wd in (1, 3, 4):
        return "WFH_Tue_Thu_Fri"
    return "Weekend"


def _scalar_float_or_nan(val: Any) -> float:
    if val is None or pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _deterministic_event_key(ds: str, tm: str, idx: int) -> str:
    digest = hashlib.sha256(f"{ds}|{tm}|{idx}".encode("utf-8")).hexdigest()[:12]
    return f"SMK-{ds.replace('-', '')}-{digest}"


def process_silver_smoking(
    smoking_raw: pd.DataFrame,
    spine_start_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Produce isolated Onset Context, Post Outcomes, Daily Missingness Spine, and QA summary."""
    run_ts = datetime.now(tz=IST).strftime("%d-%m-%Y %H:%M:%S IST")
    total_in = len(smoking_raw)

    if smoking_raw.empty:
        qa_empty = pd.DataFrame(
            [
                {
                    "run_ts_ist": run_ts,
                    "rule_id": "SMOKING_SILVER_CLEAN",
                    "total_input_rows": 0,
                    "duplicates_removed": 0,
                    "valid_events": 0,
                    "epistemic_class": "Descriptive",
                }
            ]
        )
        return {
            "Silver_Smoking_Onset_Context": pd.DataFrame(columns=ONSET_COLUMNS),
            "Silver_Smoking_Post_Outcomes": pd.DataFrame(columns=POST_OUTCOME_COLUMNS),
            "Silver_Smoking_Daily_Missingness": pd.DataFrame(
                columns=["date", "display_date", "smoking_log_status", "missing_reason", "epistemic_class"]
            ),
            "Smoking_QA_Log": qa_empty,
        }

    df = smoking_raw.copy()
    df["_iso_date"] = df["Date"].map(_to_iso_date) if "Date" in df.columns else None
    df = df.loc[df["_iso_date"].notna()].copy()
    df["_time_ist"] = df["Time"].map(_norm_time_hhmm) if "Time" in df.columns else ""
    df["_ev_norm"] = (
        df["Event_Type"].astype(str).str.strip().str.lower()
        if "Event_Type" in df.columns
        else ""
    )
    df["_cnt_norm"] = (
        df["Count"].astype(str).str.strip() if "Count" in df.columns else ""
    )
    df["_trig_norm"] = (
        df["Trigger"].astype(str).str.strip().str.lower()
        if "Trigger" in df.columns
        else ""
    )

    dup_mask = df.duplicated(
        subset=["_iso_date", "_time_ist", "_ev_norm", "_cnt_norm", "_trig_norm"],
        keep="first",
    )
    dup_cnt = int(dup_mask.sum())
    dedup = df.loc[~dup_mask].sort_values(["_iso_date", "_time_ist"]).reset_index(drop=True)

    onset_rows: list[dict[str, Any]] = []
    post_rows: list[dict[str, Any]] = []

    for idx, r in enumerate(dedup.to_dict(orient="records")):
        ds = str(r["_iso_date"])
        disp = _to_display_date(ds)
        tm = str(r["_time_ist"])
        hr_ist = float(int(tm[:2])) if len(tm) >= 2 and tm[:2].isdigit() else np.nan
        ev_id = str(r.get("Event_ID") or "").strip() or _deterministic_event_key(ds, tm, idx)

        # Onset features ONLY (strictly pre-outcome)
        intensity = _scalar_float_or_nan(r.get("Craving_Intensity"))
        intensity_miss = "" if not np.isnan(intensity) else "Not_Logged"
        trigger = str(r.get("Trigger") or "").strip()
        mood = str(r.get("Mood") or "").strip()

        onset_rows.append(
            {
                "event_key": ev_id,
                "date": ds,
                "display_date": disp,
                "onset_time_ist": tm,
                "onset_hour_ist": hr_ist,
                "work_mode": _work_mode(ds),
                "craving_intensity_onset": intensity,
                "craving_intensity_missing_reason": intensity_miss,
                "trigger_onset": trigger,
                "mood_onset": mood,
                "epistemic_class": "Measured",
            }
        )

        # Post-event outcomes ONLY (never exposed in Onset Context)
        ev_type = str(r.get("Event_Type") or "").strip()
        ev_low = ev_type.lower()
        cnt_val = _scalar_float_or_nan(r.get("Count"))
        delay_val = _scalar_float_or_nan(r.get("Delay_Minutes"))
        barrier = str(r.get("Barrier_or_Reason") or "").strip()

        is_resisted = bool(
            ("resist" in ev_low)
            or ("craving" in ev_low and (np.isnan(cnt_val) or cnt_val == 0.0))
        )
        if is_resisted and (np.isnan(cnt_val) or cnt_val == 0.0):
            cnt_clean = 0.0
            cnt_tag = "True_Zero"
        elif not np.isnan(cnt_val):
            cnt_clean = cnt_val
            cnt_tag = "True_Zero" if cnt_val == 0.0 else "Present"
        else:
            cnt_clean = np.nan
            cnt_tag = "Not_Logged"

        delay_miss = "" if not np.isnan(delay_val) else "Not_Logged"

        post_rows.append(
            {
                "event_key": ev_id,
                "date": ds,
                "display_date": disp,
                "event_type": ev_type,
                "is_resisted_outcome": is_resisted,
                "count_smoked_post": cnt_clean,
                "count_status_tag": cnt_tag,
                "delay_minutes_post": delay_val,
                "delay_missing_reason": delay_miss,
                "barrier_or_reason_post": barrier,
                "epistemic_class": "Measured",
            }
        )

    onset_df = pd.DataFrame(onset_rows, columns=ONSET_COLUMNS)
    post_df = pd.DataFrame(post_rows, columns=POST_OUTCOME_COLUMNS)

    # Daily missingness spine (distinguishes Not_Logged days from True_Zero resisted-only days)
    min_d = spine_start_date or str(onset_df["date"].min())
    max_d = str(onset_df["date"].max())
    cal_dates = pd.date_range(start=min_d, end=max_d, freq="D").strftime("%Y-%m-%d").tolist()

    smoked_sum_by_d = post_df.groupby("date")["count_smoked_post"].sum(min_count=1).to_dict()
    logged_dates = set(onset_df["date"].astype(str))

    miss_rows: list[dict[str, Any]] = []
    for ds in cal_dates:
        if ds not in logged_dates:
            st_tag, m_reason = "Not_Logged", "Not_Logged"
        else:
            tot = smoked_sum_by_d.get(ds, np.nan)
            if pd.notna(tot) and float(tot) == 0.0:
                st_tag, m_reason = "True_Zero", "True_Zero"
            else:
                st_tag, m_reason = "Present", ""
        miss_rows.append(
            {
                "date": ds,
                "display_date": _to_display_date(ds),
                "smoking_log_status": st_tag,
                "missing_reason": m_reason,
                "epistemic_class": "Descriptive",
            }
        )

    miss_df = pd.DataFrame(miss_rows)
    qa_df = pd.DataFrame(
        [
            {
                "run_ts_ist": run_ts,
                "rule_id": "SMOKING_SILVER_CLEAN",
                "total_input_rows": int(total_in),
                "duplicates_removed": int(dup_cnt),
                "valid_events": int(len(onset_df)),
                "epistemic_class": "Descriptive",
            }
        ]
    )

    return {
        "Silver_Smoking_Onset_Context": onset_df,
        "Silver_Smoking_Post_Outcomes": post_df,
        "Silver_Smoking_Daily_Missingness": miss_df,
        "Smoking_QA_Log": qa_df,
    }
